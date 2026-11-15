"""显存、数据规模与批次吞吐基准探测与可视化。

测量同一模型配置 (`d_model=320, n_layers=6, n_heads=8, d_ff=1280, vocab=8192`) 在目标 GPU 上：
1) 数据量级 (`max_chars`: 100K ~ 80M) 对 CPU 数据张量内存与 GPU 峰值显存的影响（证明单步显存与总语料无关，仅受 batch*block 与单次 eval 批量控制）；
2) `batch_size x block_size` 扫描的真实峰值显存 (allocated/reserved)、tokens/s 吞吐、步耗时与常见消费级显卡 (`8GB / 16GB / 24GB / 64GB+`) 适配边界；
3) 等效全局批 `effective_batch=64` 下 `grad_accum` (`bs=64/1` vs `32/2` vs `16/4` vs `8/8`) 的显存收益与速度代价。

用法：
    python scripts/scaling/profile_vram_batch.py --device cuda --amp bf16 --out "$XL_OUT_DIR/scaling"
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

HERE = Path(__file__).resolve().parent
SCRIPTS_DIR = HERE.parent
REPO_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

import llm_common as lc  # noqa: E402


def _cleanup_cuda(dev: torch.device) -> None:
    gc.collect()
    if dev.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)


def bench_step_config(
    cfg_dict: dict,
    vocab_size: int,
    block_size: int,
    batch_size: int,
    grad_accum: int,
    amp_mode: str,
    device: torch.device,
    warmup: int = 3,
    iters: int = 10,
) -> dict:
    """测量单组 (block_size, batch_size, grad_accum) 的真实峰值显存与稳态吞吐。"""
    _cleanup_cuda(device)
    cfg = lc.ModelConfig(
        vocab_size=vocab_size,
        d_model=cfg_dict["d_model"],
        n_layers=cfg_dict["n_layers"],
        n_heads=cfg_dict["n_heads"],
        n_kv_heads=cfg_dict["n_heads"],
        d_ff=cfg_dict["d_ff"],
        max_len=block_size,
    )
    row = {
        "block_size": block_size,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "effective_batch": batch_size * grad_accum,
        "tokens_per_update": batch_size * grad_accum * block_size,
        "amp": amp_mode,
        "oom": False,
    }
    try:
        model = lc.build_model(cfg, seed=0).to(device)
        optimizers, _ = lc.build_optimizer(model, cfg, lr=3e-4, optimizer="muon")
        amp = lc.AmpCtx(amp_mode, device)
        fake_ids = torch.randint(0, vocab_size, (max(batch_size * 4, 512) * block_size,), dtype=torch.long)
        gen = torch.Generator().manual_seed(42)

        def one_step():
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)
            for _ in range(grad_accum):
                x, y = lc.get_batch(fake_ids, batch_size, block_size, gen, device=device)
                with amp.autocast():
                    _, loss = model(x, targets=y)
                amp.backward(loss / grad_accum)
            amp.step(optimizers, model=model, grad_clip=1.0)

        model.train()
        for _ in range(warmup):
            one_step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        for _ in range(iters):
            one_step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        dt = time.perf_counter() - t0
        mem = lc.memory_stats(device)
        tok_per_s = (iters * batch_size * grad_accum * block_size) / max(dt, 1e-6)
        ms_per_upd = 1000.0 * dt / iters
        eta_10m_min = (10_000_000 / tok_per_s) / 60.0 if tok_per_s > 0 else None
        row.update({
            "model_params": model.num_params(),
            "cpu_stream_mb": lc.tensor_size_mb(fake_ids),
            "peak_vram_allocated_mb": mem["peak_vram_allocated_mb"],
            "peak_vram_reserved_mb": mem["peak_vram_reserved_mb"],
            "gpu_total_vram_mb": mem["gpu_total_vram_mb"],
            "tokens_per_second": round(tok_per_s, 1),
            "ms_per_update": round(ms_per_upd, 2),
            "eta_10m_tokens_min": round(eta_10m_min, 2) if eta_10m_min else None,
        })
        del model, optimizers, fake_ids
    except torch.cuda.OutOfMemoryError:
        row["oom"] = True
        row["peak_vram_allocated_mb"] = None
        row["peak_vram_reserved_mb"] = None
        row["tokens_per_second"] = 0.0
    _cleanup_cuda(device)
    return row


def measure_corpus_footprint(
    train_source: str | None, tokenizer_path: str | None, max_chars_list: list[int]
) -> list[dict]:
    """实测不同语料规模 (`max_chars`) 的 token 数、CPU 张量大小与加载时间。"""
    if not train_source or not tokenizer_path or not Path(train_source).exists() or not Path(tokenizer_path).exists():
        # 若未提供外部 parquet，按约 3.94 chars/token + int64 (8B/token) 精确换算
        return [
            {
                "max_chars": mc,
                "estimated_tokens": int(mc / 3.94),
                "cpu_int64_mb": round((mc / 3.94) * 8 / (1024**2), 2),
                "cpu_uint16_mb": round((mc / 3.94) * 2 / (1024**2), 2),
                "measured": False,
            }
            for mc in max_chars_list
        ]
    tok = lc.load_tokenizer(tokenizer_path)
    eos = lc.eos_id(tok)
    rows = []
    for mc in max_chars_list:
        t0 = time.perf_counter()
        ids, stats = lc.build_ids_from_source(train_source, tok, mc, 10_000, eos)
        dt = time.perf_counter() - t0
        n_tok = int(ids.numel())
        rows.append({
            "max_chars": mc,
            "docs": stats.get("docs"),
            "tokens": n_tok,
            "cpu_int64_mb": lc.tensor_size_mb(ids),
            "cpu_uint16_mb": round(n_tok * 2 / (1024**2), 2),
            "prep_seconds": round(dt, 2),
            "measured": True,
        })
        del ids
    return rows


def plot_profile(data: dict, out_dir: Path) -> None:
    sweep = [r for r in data["batch_block_sweep"] if not r["oom"]]
    accum_rows = [r for r in data["grad_accum_sweep"] if not r["oom"]]
    if not sweep:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    for blk in sorted({r["block_size"] for r in sweep}):
        sub = sorted([r for r in sweep if r["block_size"] == blk], key=lambda x: x["batch_size"])
        axes[0].plot(
            [r["batch_size"] for r in sub],
            [r["peak_vram_allocated_mb"] for r in sub],
            "o-",
            label=f"block={blk}",
        )
        axes[1].plot(
            [r["batch_size"] for r in sub],
            [r["tokens_per_second"] for r in sub],
            "s-",
            label=f"block={blk}",
        )
    axes[0].axhline(8192, color="tab:red", ls="--", lw=1.0, alpha=0.6, label="8GB limit")
    axes[0].set_title("Peak VRAM vs Batch & Block")
    axes[0].set_xlabel("batch_size (per rank)")
    axes[0].set_ylabel("Peak Allocated VRAM (MB)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].set_title("Throughput vs Batch & Block")
    axes[1].set_xlabel("batch_size (per rank)")
    axes[1].set_ylabel("tokens / s")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    if accum_rows:
        labels = [f"bs={r['batch_size']}\nacc={r['grad_accum']}" for r in accum_rows]
        x = range(len(labels))
        ax_v = axes[2]
        ax_t = ax_v.twinx()
        ax_v.bar([i - 0.18 for i in x], [r["peak_vram_allocated_mb"] for r in accum_rows],
                 width=0.36, color="tab:blue", alpha=0.8, label="Peak VRAM (MB)")
        ax_t.plot(list(x), [r["tokens_per_second"] for r in accum_rows],
                  "o-", color="tab:orange", lw=2, label="tokens/s")
        ax_v.set_xticks(list(x))
        ax_v.set_xticklabels(labels, fontsize=8)
        ax_v.set_ylabel("Peak VRAM (MB)", color="tab:blue")
        ax_t.set_ylabel("tokens / s", color="tab:orange")
        ax_v.set_title("Equal Effective Batch=64 Tradeoff")
        ax_v.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Memory & Throughput Profile ({data['gpu_name']}, {data['amp']})")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_vram_batch_tradeoff.png", dpi=140)
    plt.close(fig)


def write_table(data: dict, path: Path) -> None:
    lines = [
        f"# 显存、批次与语料内存基准 ({data['gpu_name']})",
        "",
        f"模型参数量: {data.get('model_params', 0):,} ({data['model']}), 精度: {data['amp']}, 词表: {data['vocab_size']}",
        f"显存容量: {data.get('gpu_total_vram_mb', 0):.0f} MB",
        "",
        "## 1. 语料规模 vs CPU 张量内存（GPU 训练显存与语料总大小无关）",
        "",
        "| max_chars | docs | tokens | int64 CPU 内存 (MB) | uint16 压缩 (MB) | 切词耗时 (s) |",
        "|---|---|---|---|---|---|",
    ]
    for r in data["corpus_footprint"]:
        tok_v = r.get("tokens") or r.get("estimated_tokens")
        lines.append(
            f"| {r['max_chars']:,} | {r.get('docs', 'est')} | {tok_v:,} | "
            f"{r['cpu_int64_mb']} | {r['cpu_uint16_mb']} | {r.get('prep_seconds', 'n/a')} |"
        )
    lines += [
        "",
        "## 2. Batch Size x Context Length (block_size) 显存与吞吐矩阵",
        "",
        "| block_size | batch_size | 单步 tokens | Peak Alloc (MB) | Peak Reserved (MB) | tokens/s | ms/step | 10M tokens 预计耗时 (min) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in data["batch_block_sweep"]:
        if r["oom"]:
            lines.append(f"| {r['block_size']} | {r['batch_size']} | {r['tokens_per_update']:,} | OOM | OOM | 0 | n/a | n/a |")
        else:
            lines.append(
                f"| {r['block_size']} | {r['batch_size']} | {r['tokens_per_update']:,} | "
                f"{r['peak_vram_allocated_mb']} | {r['peak_vram_reserved_mb']} | "
                f"{r['tokens_per_second']:,} | {r['ms_per_update']} | {r['eta_10m_tokens_min']} |"
            )
    lines += [
        "",
        "## 3. 等效全局批 (effective_batch=64, block_size=256) 的 grad_accum 取舍",
        "",
        "| batch_size | grad_accum | Peak Alloc (MB) | Peak Reserved (MB) | tokens/s | 相对峰值显存 |",
        "|---|---|---|---|---|---|",
    ]
    base_v = next((r["peak_vram_allocated_mb"] for r in data["grad_accum_sweep"] if not r["oom"]), None)
    for r in data["grad_accum_sweep"]:
        if r["oom"]:
            lines.append(f"| {r['batch_size']} | {r['grad_accum']} | OOM | OOM | 0 | n/a |")
        else:
            ratio = f"{100.0 * r['peak_vram_allocated_mb'] / base_v:.1f}%" if base_v else "100%"
            lines.append(
                f"| {r['batch_size']} | {r['grad_accum']} | {r['peak_vram_allocated_mb']} | "
                f"{r['peak_vram_reserved_mb']} | {r['tokens_per_second']:,} | {ratio} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="显存、数据规模与批次吞吐基准探测（模型尺寸走参数，不依赖集群配置）")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--amp", default="bf16", choices=["none", "bf16", "fp16"])
    ap.add_argument("--vocab-size", type=int, default=8192)
    ap.add_argument("--d-model", type=int, default=320, dest="d_model")
    ap.add_argument("--n-layers", type=int, default=6, dest="n_layers")
    ap.add_argument("--n-heads", type=int, default=8, dest="n_heads")
    ap.add_argument("--d-ff", type=int, default=1280, dest="d_ff")
    ap.add_argument("--train-source", default=None, help="可选 parquet/txt 路径以实测各档 CPU token 内存")
    ap.add_argument("--tokenizer", default=None, help="tokenizer.json 路径")
    ap.add_argument("--out", default=os.environ.get("XL_OUT_DIR") or str(REPO_ROOT / "results" / "scaling"),
                    help="输出目录（缺省 $XL_OUT_DIR/scaling）")
    ap.add_argument("--tag", default=None, help="输出文件名后缀（如 gb10 / rtx3080）")
    args = ap.parse_args()

    # 模型尺寸：缺省与数据量级扩展实验一致（d_model=320, n_layers=6, n_heads=8, d_ff=1280）
    model_cfg = {
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "d_ff": args.d_ff,
    }
    dev = lc.resolve_device(args.device)
    gpu_name = torch.cuda.get_device_name(dev) if dev.type == "cuda" else "CPU"
    total_mb = (
        round(torch.cuda.get_device_properties(dev).total_memory / (1024**2), 1)
        if dev.type == "cuda"
        else 0.0
    )

    sweep = []
    for blk in (128, 256, 512):
        batches = (8, 16, 32, 64) if blk <= 256 else (8, 16, 32)
        for bs in batches:
            res = bench_step_config(model_cfg, args.vocab_size, blk, bs, 1, args.amp, dev)
            sweep.append(res)
            print(f"[prof] blk={blk} bs={bs} -> alloc={res['peak_vram_allocated_mb']}MB tok/s={res['tokens_per_second']} oom={res['oom']}")

    accum_sweep = []
    for bs, acc in ((64, 1), (32, 2), (16, 4), (8, 8)):
        res = bench_step_config(model_cfg, args.vocab_size, 256, bs, acc, args.amp, dev)
        accum_sweep.append(res)
        print(f"[accum] bs={bs}x{acc}=64 -> alloc={res['peak_vram_allocated_mb']}MB tok/s={res['tokens_per_second']} oom={res['oom']}")

    corpus = measure_corpus_footprint(args.train_source, args.tokenizer, [100_000, 2_000_000, 20_000_000, 80_000_000])
    params = next((r["model_params"] for r in sweep if "model_params" in r), 0)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    payload = {
        "gpu_name": gpu_name,
        "gpu_total_vram_mb": total_mb,
        "amp": args.amp,
        "vocab_size": args.vocab_size,
        "model": model_cfg,
        "model_params": params,
        "corpus_footprint": corpus,
        "batch_block_sweep": sweep,
        "grad_accum_sweep": accum_sweep,
    }
    (out_dir / f"vram_batch_profile{suffix}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_table(payload, out_dir / f"vram_batch_table{suffix}.md")
    if not suffix or args.tag == "gb10":
        plot_profile(payload, out_dir)
    print(f"[ok] 基准写入 {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
