#!/usr/bin/env python3
"""pipeline_stage.py 的轻量自测：不依赖 GPU 的协议级回归。

覆盖三类检查：
1. 纯函数：summarize / bandwidth_fields / merge_peer_payload 的口径
   （GB/s 按 1e9 字节换算；对端指标不会覆盖本地 numerical/gpu_memory）；
2. 端到端：本机起两个 gloo CPU 进程跑一轮正式通信，验证
   - 逐元素探针校验通过（probe_element_mismatches == 0）；
   - stage 2 的 recv 计时有实测值（旧版恒为 0）；
   - 报告里 nodes.stage1 / nodes.stage2 各自独立，results.numerical
     取 stage 1 的真实判定；
   - 带宽字段与「字节数 ÷ 耗时」重算一致（单位修正回归）。
3. 本脚本是单机两进程的协议自测，不是双机通信证据。

用法：python scripts/test_pipeline_stage_gloo.py
依赖：torch（CPU 版即可）。torch 缺失时不执行自测并返回非零状态。
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

ITERS = 3
WARMUP = 1
BATCH, SEQ, HIDDEN = 1, 8, 32
DTYPE = "fp32"
ACT_BYTES = BATCH * SEQ * HIDDEN * 4  # fp32

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_pure_functions() -> None:
    print("== 纯函数检查 ==")
    import pipeline_stage as ps

    check("summarize 空列表返回空", ps.summarize([]) == {})
    s = ps.summarize([3.0, 1.0, 2.0])
    check("summarize 均值/分位", s["mean_ms"] == 2.0 and s["p50_ms"] == 2.0 and s["n"] == 3, str(s))

    bw = ps.bandwidth_fields(1_000_000_000, 1.0)
    check("bandwidth_fields 1e9 字节/秒 = 1 GB/s", bw["GB_per_s"] == 1.0, str(bw))
    check("bandwidth_fields 附带原始 bytes/s", bw["bytes_per_s"] == 1e9)
    check("bandwidth_fields 零耗时返回 None", ps.bandwidth_fields(10, 0.0)["GB_per_s"] is None)

    local = {"stage1": {"numerical": {"passed": True, "probe_max_abs_diff": 0.0},
                        "gpu_memory": {"max_allocated_mb": 111.0}}}
    peer = {"stage2": {"numerical": {"passed": True, "probe_max_abs_diff": 0.0},
                       "gpu_memory": {"max_allocated_mb": 222.0}},
            "errors": ["boom"]}
    merged, errs = ps.merge_peer_payload(local, peer, "stage2")
    check("合并后 stage1 numerical 不被对端覆盖",
          merged["stage1"]["numerical"] == local["stage1"]["numerical"])
    check("合并后 stage1 gpu_memory 不被对端覆盖",
          merged["stage1"]["gpu_memory"]["max_allocated_mb"] == 111.0)
    check("stage2 节点段被并入", merged["stage2"]["gpu_memory"]["max_allocated_mb"] == 222.0)
    check("对端 errors 带角色前缀", errs == ["[stage2] boom"], str(errs))

    merged2, errs2 = ps.merge_peer_payload(local, {}, "stage2")
    check("对端缺失时记录错误且不崩溃", len(errs2) == 1 and "stage2" not in merged2)


def test_end_to_end() -> bool:
    print("== 端到端自测（本机两进程，gloo + CPU）==")
    try:
        import torch  # noqa: F401
    except ImportError:
        print("  SKIP: torch 不可用，跳过端到端自测")
        return True

    port = free_port()
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "metrics.json")
        common = [
            sys.executable, os.path.join(SCRIPT_DIR, "pipeline_stage.py"),
            "--backend", "gloo", "--device", "cpu", "--dtype", DTYPE,
            "--batch", str(BATCH), "--seq", str(SEQ), "--hidden", str(HIDDEN),
            "--iters", str(ITERS), "--warmup", str(WARMUP),
            "--master-addr", "127.0.0.1", "--master-port", str(port),
            "--barrier-timeout", "60",
        ]
        p1 = subprocess.Popen(common + ["--role", "stage1", "--out", out],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        time.sleep(0.5)
        p2 = subprocess.Popen(common + ["--role", "stage2"],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            o1, e1 = p1.communicate(timeout=180)
            o2, e2 = p2.communicate(timeout=180)
        except subprocess.TimeoutExpired:
            for p in (p1, p2):
                p.kill()
            check("两进程在超时前结束", False, "180s 超时")
            return False

        check("stage1 退出码为 0", p1.returncode == 0, f"rc={p1.returncode} err={e1[-300:]}")
        check("stage2 退出码为 0", p2.returncode == 0, f"rc={p2.returncode} err={e2[-300:]}")
        check("stage1 输出报告存在", os.path.isfile(out))
        if not (os.path.isfile(out) and p1.returncode == 0 and p2.returncode == 0):
            return False

        with open(out, "r", encoding="utf-8") as f:
            rep = json.load(f)

        check("status == passed", rep.get("status") == "passed", str(rep.get("status")))
        check("errors 为空", rep.get("errors") == [], str(rep.get("errors")))
        num = rep["results"]["numerical"]
        check("逐元素校验无不匹配", num.get("probe_element_mismatches") == 0, str(num))
        check("逐元素校验最大差为 0", num.get("probe_max_abs_diff") == 0.0, str(num))
        check("探针元素数正确", num.get("probe_elements") == BATCH * SEQ * HIDDEN)

        n1, n2 = rep["nodes"]["stage1"], rep["nodes"]["stage2"]
        check("两端节点段都在", bool(n1.get("host")) and bool(n2.get("host")))
        check("stage1 send/recv/compute 均有 %d 次实测" % ITERS,
              n1["send"]["n"] == ITERS and n1["recv"]["n"] == ITERS and n1["compute"]["n"] == ITERS)
        check("stage2 recv 有实测值（非旧版恒 0）",
              n2["recv"].get("n") == ITERS and n2["recv"].get("mean_ms", 0) >= 0)
        check("stage1 numerical 与 results.numerical 一致",
              n1.get("numerical") == num)

        send = n1["throughput"]["send_effective"]
        expected = round(ACT_BYTES / (n1["send"]["mean_ms"] / 1e3) / 1e9, 6)
        check("stage1 发送带宽与字节数÷耗时重算一致（GB/s 口径）",
              abs(send["GB_per_s"] - expected) <= max(1e-9, abs(expected) * 1e-3),
              f"reported={send['GB_per_s']} expected={expected}")
        check("吞吐附带原始 bytes/s",
              abs(send["bytes_per_s"] - send["GB_per_s"] * 1e9) <= 500.0)
        check("schema 为 v2", rep.get("schema") == "xiaokeailabs/smoke-pipeline-metrics/v2")
    return not failures


def main() -> int:
    try:
        import torch  # noqa: F401
    except ImportError:
        print("SKIP: torch 不可用，未执行自测")
        return 1
    test_pure_functions()
    test_end_to_end()
    if failures:
        print(f"\n结果：FAIL（{len(failures)} 项未过）")
        return 1
    print("\n结果：PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
