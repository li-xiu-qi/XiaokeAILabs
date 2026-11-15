#!/usr/bin/env python3
"""跨机流水线并行（PP=2）张量通信冒烟脚本。

两个节点各起一个进程，通过 TCPStore 会合、点对点收发，模拟一次
流水线前向：stage 1 在本地设备上算出激活张量 H，发送给 stage 2；
stage 2 接收后本地计算，回传一个固定值张量，stage 1 收下，构成一次
完整往返。逐次记录收发延迟、本地计算耗时、激活吞吐与显存占用，
由 stage 1 汇总写出 JSON 指标文件。

用法（两机各执行一次，--master-addr 指向 stage 1 所在机器）：

    # 节点 A（stage 1，主节点）
    python scripts/pipeline_stage.py --role stage1 --master-addr <stage1-ip>

    # 节点 B（stage 2）
    python scripts/pipeline_stage.py --role stage2 --master-addr <stage1-ip>

依赖：torch。NCCL 后端要求两端各一块 NVIDIA GPU；backend=gloo 时
可用 CPU 在本机起两个进程做协议自测（--device cpu）。不加载任何
模型权重，全部张量为随机合成数据。

指标口径（schema v2）：
- nodes.stage1 / nodes.stage2 各自携带本节点的 host、设备、环境、
  send/recv/compute/roundtrip 计时与本节点显存峰值，互不覆盖；
- results.numerical 取 stage 1 的逐元素校验结果（探针张量原样回传
  后逐元素比对，不是只比总和）；
- 带宽类字段单位为 GB/s（1 GB = 1e9 字节），同名字段附
  bytes_per_s 原始值；
- 阶段同步失败时停止；通信超时由进程组配置控制。
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import socket
import statistics
import sys
import time

import torch
import torch.distributed as dist

GB = 1e9  # 带宽字段统一按 1 GB = 1e9 字节换算


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--role", choices=["stage1", "stage2"], required=True, help="本节点承担的流水线阶段")
    p.add_argument("--master-addr", required=True, help="stage 1 节点的 IP 或主机名（TCPStore 会合点）")
    p.add_argument("--master-port", type=int, default=29500, help="TCPStore 端口（默认 29500）")
    p.add_argument("--backend", choices=["nccl", "gloo"], default="nccl", help="分布式后端（默认 nccl；gloo 可用于 CPU 自测）")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                   help="计算设备（默认 auto：有 GPU 用 cuda，否则 cpu；cpu 需配合 --backend gloo）")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"], help="激活张量精度")
    p.add_argument("--batch", type=int, default=1, help="批大小")
    p.add_argument("--seq", type=int, default=128, help="序列长度（token 数）")
    p.add_argument("--hidden", type=int, default=4096, help="隐层宽度")
    p.add_argument("--local-layers", type=int, default=8, help="本阶段模拟的 Transformer 层数（仅影响本地计算量）")
    p.add_argument("--iters", type=int, default=50, help="正式计时迭代次数")
    p.add_argument("--warmup", type=int, default=10, help="预热迭代次数（不计入指标）")
    p.add_argument("--timeout-min", type=int, default=30, help="分布式初始化超时（分钟）")
    p.add_argument("--barrier-timeout", type=int, default=120,
                   help="Gloo 阶段同步的超时秒数（NCCL 使用进程组超时）")
    p.add_argument("--out", default=None, help="指标 JSON 输出路径（仅 stage 1 写入；默认打印到 stdout）")
    return p.parse_args()


def summarize(values: list[float]) -> dict:
    """给定毫秒列表，返回均值与分位数。"""
    if not values:
        return {}
    s = sorted(values)
    return {
        "mean_ms": round(statistics.fmean(s), 9),
        "p50_ms": round(s[len(s) // 2], 4),
        "p95_ms": round(s[min(len(s) - 1, int(len(s) * 0.95))], 4),
        "min_ms": round(s[0], 4),
        "max_ms": round(s[-1], 4),
        "n": len(s),
    }


def bandwidth_fields(act_bytes: int, seconds: float) -> dict:
    """把 bytes/seconds 换算成 GB/s（1e9 字节）并附原始 bytes/s。"""
    if seconds <= 0:
        return {"bytes_per_s": None, "GB_per_s": None}
    return {
        "bytes_per_s": round(act_bytes / seconds, 2),
        "GB_per_s": round(act_bytes / seconds / GB, 6),
    }


def node_throughput(act_bytes: int, send_ms: list[float], roundtrip_ms: list[float]) -> dict:
    """按计时区间计算有效吞吐；往返时间还包含两端计算与等待。"""
    out: dict = {"activation_bytes_per_iter": act_bytes, "unit": "1 GB = 1e9 bytes"}
    if send_ms:
        out["send_effective"] = bandwidth_fields(act_bytes, statistics.fmean(send_ms) / 1e3)
    if roundtrip_ms:
        out["roundtrip_effective"] = bandwidth_fields(act_bytes * 2, statistics.fmean(roundtrip_ms) / 1e3)
    return out


def gpu_info(device: torch.device) -> dict:
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        free_b, total_b = torch.cuda.mem_get_info()
        return {
            "name": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "sm_count": props.multi_processor_count,
            "gpu_total_mb": round(total_b / 2**20, 1),
            "gpu_free_mb": round(free_b / 2**20, 1),
        }
    return {"name": "cpu", "compute_capability": None, "sm_count": None,
            "gpu_total_mb": None, "gpu_free_mb": None}


def env_info() -> dict:
    return {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "nccl": torch.cuda.nccl.version() if torch.cuda.is_available() else None,
        "python": sys.version.split()[0],
        "nvfp4_dtype_available": hasattr(torch, "float4_e2m1fn_x2"),
    }


def memory_stats(device: torch.device) -> dict:
    if device.type == "cuda":
        return {
            "max_allocated_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
            "reserved_mb": round(torch.cuda.memory_reserved() / 2**20, 1),
        }
    return {"max_allocated_mb": None, "reserved_mb": None}


def local_forward(x: torch.Tensor, weight: torch.Tensor, layers: int = 1) -> torch.Tensor:
    """用多次矩阵乘和激活函数模拟计算负载。"""
    for _ in range(layers):
        x = torch.nn.functional.gelu(x @ weight)
    return x


def merge_peer_payload(local: dict, peer: dict, peer_role: str) -> tuple[dict, list[str]]:
    """把对端回传的节点指标并入本地结果。

    只接收对端以自己角色名命名的节点段与 errors；对端的
    numerical / gpu_memory 等占位值不会覆盖本节点的真实结果。
    返回 (合并后的结果, 对端错误列表)。
    """
    merged = dict(local)
    errors: list[str] = []
    node = peer.get(peer_role) if isinstance(peer, dict) else None
    if isinstance(node, dict):
        merged[peer_role] = node
        for e in peer.get("errors", []) or []:
            errors.append(f"[{peer_role}] {e}")
    else:
        errors.append(f"未收到 {peer_role} 的指标回传（结构不符）")
    return merged, errors


def main() -> int:
    args = parse_args()
    rank = 0 if args.role == "stage1" else 1
    is_stage1 = rank == 0
    world_size = 2
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    if (min(args.batch, args.seq, args.hidden, args.local_layers, args.iters,
            args.timeout_min, args.barrier_timeout) <= 0 or args.warmup < 0
            or not 1 <= args.master_port <= 65535):
        print(f"[{args.role}] FATAL: 维度、层数、迭代次数和超时须大于零，预热次数须非负",
              file=sys.stderr)
        return 2
    if is_stage1 and args.out and os.path.exists(args.out):
        print(f"[{args.role}] FATAL: 输出文件已存在，请指定新路径", file=sys.stderr)
        return 2
    print(f"[{args.role}] 检查计算设备", flush=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        print(f"[{args.role}] FATAL: 指定了 --device cuda 但 torch.cuda.is_available() 为 False", file=sys.stderr)
        return 2
    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(f"{args.device}:0" if args.device == "cuda" else "cpu")

    if args.backend == "nccl" and device.type != "cuda":
        print(f"[{args.role}] FATAL: nccl 后端需要 CUDA 设备", file=sys.stderr)
        return 2
    if device.type == "cuda":
        print(f"[{args.role}] 初始化 CUDA 设备", flush=True)
        torch.cuda.set_device(device)
        print(f"[{args.role}] CUDA 设备已初始化", flush=True)
    os.environ.setdefault("NCCL_DEBUG", "WARN")

    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize()

    def reset_peak() -> None:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

    def safe_barrier() -> None:
        """Gloo 使用指定超时；NCCL 使用进程组超时。失败必须中止。"""
        if args.backend == "gloo":
            dist.monitored_barrier(timeout=datetime.timedelta(seconds=args.barrier_timeout))
        else:
            dist.barrier()

    errors: list[str] = []
    status = "failed"
    results: dict = {}
    peer_metrics: dict = {}

    def emit_report() -> None:
        nonlocal status
        merged, peer_errors = merge_peer_payload(results, peer_metrics, "stage2")
        errors.extend(peer_errors)
        stage1_node = merged.get("stage1", {})
        stage2_node = merged.get("stage2", {})
        if peer_errors or errors:
            status = "failed"
        report = {
            "schema": "xiaokeailabs/smoke-pipeline-metrics/v2",
            "status": status,
            "generated_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "config": {
                "world_size": world_size,
                "pipeline_stages": 2,
                "backend": args.backend,
                "device": str(device),
                "dtype": args.dtype,
                "batch": args.batch,
                "seq": args.seq,
                "hidden": args.hidden,
                "local_layers": args.local_layers,
                "iters": args.iters,
                "warmup": args.warmup,
                "master": f"{args.master_addr}:{args.master_port}",
                "nccl_socket_ifname": os.environ.get("NCCL_SOCKET_IFNAME"),
            },
            "nodes": {"stage1": stage1_node, "stage2": stage2_node},
            "results": {
                "numerical": stage1_node.get("numerical", {}),
                "gpu_memory": stage1_node.get("gpu_memory", {}),
                "throughput": stage1_node.get("throughput", {}),
                "stage1_comm": {k: stage1_node.get(k, {})
                                for k in ("send", "recv", "roundtrip", "compute")},
                "stage2_comm": {k: stage2_node.get(k, {})
                                for k in ("send", "recv", "roundtrip", "compute")},
            },
            "errors": errors,
        }
        text = json.dumps(report, ensure_ascii=False, indent=2)
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, "x", encoding="utf-8") as f:
                f.write(text + "\n")
        print(text)

    try:
        print(f"[{args.role}] 初始化 {args.backend} 进程组", flush=True)
        dist.init_process_group(
            backend=args.backend,
            init_method=f"tcp://{args.master_addr}:{args.master_port}",
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(minutes=args.timeout_min),
        )
        print(f"[{args.role}] 进程组已连接", flush=True)
        config = {key: getattr(args, key) for key in
                  ("backend", "dtype", "batch", "seq", "hidden", "local_layers", "iters", "warmup",
                   "timeout_min", "barrier_timeout")}
        configs = [None, None]
        dist.all_gather_object(configs, config)
        if configs[0] != configs[1]:
            raise ValueError(f"两端配置不一致: {configs}")
        safe_barrier()

        torch.manual_seed(1234 + rank)
        weight = (torch.randn(args.hidden, args.hidden, device=device, dtype=torch.float32) * 0.02).to(dtype)

        send_ms: list[float] = []
        recv_ms: list[float] = []
        compute_ms: list[float] = []
        roundtrip_ms: list[float] = []

        def one_round(stage1_side: bool) -> tuple[float, float, float]:
            """一次前向往返，返回 (send_ms, compute_ms, recv_ms)。"""
            if stage1_side:
                h = torch.randn(args.batch, args.seq, args.hidden, device=device, dtype=dtype)
                t0 = time.perf_counter()
                y = local_forward(h, weight, args.local_layers)
                sync()
                t1 = time.perf_counter()
                dist.send(y.contiguous(), dst=1)
                sync()
                t2 = time.perf_counter()
                grad = torch.empty_like(y)
                dist.recv(grad, src=1)
                sync()
                t3 = time.perf_counter()
                return (t2 - t1) * 1e3, (t1 - t0) * 1e3, (t3 - t2) * 1e3
            else:
                y = torch.empty(args.batch, args.seq, args.hidden, device=device, dtype=dtype)
                r0 = time.perf_counter()
                dist.recv(y, src=0)
                sync()
                r1 = time.perf_counter()
                t0 = time.perf_counter()
                out = local_forward(y, weight, args.local_layers)
                sync()
                t1 = time.perf_counter()
                grad = torch.full_like(out, 0.5)
                dist.send(grad.contiguous(), dst=0)
                sync()
                t2 = time.perf_counter()
                return (t2 - t1) * 1e3, (t1 - t0) * 1e3, (r1 - r0) * 1e3

        # 预热
        reset_peak()
        for _ in range(args.warmup):
            one_round(is_stage1)
        safe_barrier()

        # 正式计时
        reset_peak()
        t_start = time.perf_counter()
        for _ in range(args.iters):
            t0 = time.perf_counter()
            s, c, r = one_round(is_stage1)
            t1 = time.perf_counter()
            if is_stage1:
                send_ms.append(s)
                roundtrip_ms.append((t1 - t0) * 1e3)
            else:
                send_ms.append(s)
            compute_ms.append(c)
            recv_ms.append(r)
        total_s = time.perf_counter() - t_start
        safe_barrier()

        # 逐元素校验：stage 2 把收到的探针张量原样回传，stage 1 逐元素比对，
        # 比总和更能发现传输损坏（README 旧版只比总和，已修正）。
        if is_stage1:
            probe = torch.randn(args.batch, args.seq, args.hidden, device=device, dtype=dtype)
            dist.send(probe.contiguous(), dst=1)
            echo = torch.empty_like(probe)
            dist.recv(echo, src=1)
            max_abs_diff = (echo.double() - probe.double()).abs().max().item()
            element_mismatches = int(torch.count_nonzero(echo != probe).item())
        else:
            probe = torch.empty(args.batch, args.seq, args.hidden, device=device, dtype=dtype)
            dist.recv(probe, src=0)
            dist.send(probe.contiguous(), dst=0)
            max_abs_diff, element_mismatches = None, None

        numerical = {
            "probe_max_abs_diff": max_abs_diff,
            "probe_element_mismatches": element_mismatches,
            "probe_elements": args.batch * args.seq * args.hidden,
            "passed": element_mismatches == 0 if is_stage1 else None,
        }
        # 由执行比对的第一阶段向第二阶段发送判定；第二阶段不能凭占位值报通过。
        verdict = torch.tensor(
            [int(numerical["passed"]) if is_stage1 else 0], dtype=torch.int32, device=device
        )
        if is_stage1:
            dist.send(verdict, dst=1)
        else:
            dist.recv(verdict, src=0)
            numerical["passed"] = bool(verdict.item())

        node = {
            "role": args.role,
            "host": socket.gethostname(),
            "device": str(device),
            **gpu_info(device),
            **env_info(),
        }
        act_bytes = args.batch * args.seq * args.hidden * torch.empty((), dtype=dtype).element_size()
        local_metrics = {
            "send": summarize(send_ms),
            "recv": summarize(recv_ms),
            "compute": summarize(compute_ms),
            "roundtrip": summarize(roundtrip_ms),
            "wall_total_s": round(total_s, 4),
            "throughput": node_throughput(act_bytes, send_ms, roundtrip_ms),
        }
        if is_stage1 and total_s > 0:
            local_metrics["throughput"]["activation_tokens_per_s"] = round(
                args.batch * args.seq * args.iters / total_s, 2)

        results = {
            args.role: {
                **node,
                "gpu_memory": memory_stats(device),
                "numerical": numerical,
                **local_metrics,
            }
        }
        status = "passed" if numerical["passed"] else "failed"
        # stage 2 把自己的指标回传 stage 1 汇总
        if not is_stage1:
            payload = json.dumps(results).encode("utf-8")
            n = torch.tensor([len(payload)], dtype=torch.int64, device=device)
            dist.send(n.contiguous(), dst=0)
            buf = torch.frombuffer(bytearray(payload), dtype=torch.uint8).to(device)
            dist.send(buf.contiguous(), dst=0)
            ack = torch.zeros(1, dtype=torch.int32, device=device)
            dist.recv(ack, src=0)
            if not bool(ack.item()):
                status = "failed"
        else:
            n = torch.zeros(1, dtype=torch.int64, device=device)
            dist.recv(n, src=1)
            buf = torch.zeros(int(n.item()), dtype=torch.uint8, device=device)
            dist.recv(buf, src=1)
            peer_metrics = json.loads(buf.cpu().numpy().tobytes().decode("utf-8"))
            emit_report()
            ack = torch.tensor([int(status == "passed")], dtype=torch.int32, device=device)
            dist.send(ack, dst=1)

    except Exception as exc:  # noqa: BLE001
        errors.append(f"{type(exc).__name__}: {exc}")
        status = "failed"
        print(f"[{args.role}] ERROR: {errors[-1]}", file=sys.stderr)
    finally:
        if dist.is_available() and dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception as exc:  # noqa: BLE001
                print(f"[{args.role}] WARN: destroy_process_group 失败：{exc}", file=sys.stderr)

    if not is_stage1:
        # stage 2 打印自己的本地结果，便于对端失联时单独排查
        print(json.dumps(results, ensure_ascii=False, indent=2))

    return 0 if status == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
