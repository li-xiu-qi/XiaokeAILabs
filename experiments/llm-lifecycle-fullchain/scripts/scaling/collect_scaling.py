#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""collect_scaling.py — 汇总数据量级扩展实验指标，产出 JSON / 对比表 / 绘图。

机器无关版：只读本地结果目录（`--results-dir`，缺省 `$XL_OUT_DIR/scaling`），
不做任何 ssh。目录结构约定：

    <results-dir>/<mode>/<tier>/pretrain_metrics.json
    <results-dir>/<mode>/<tier>/sft_metrics.json        （可选）
    <results-dir>/<mode>/<tier>/failover_events.json    （可选）

mode 如 single / dual / failover，tier 如 t1_micro_100k。产物：

    <out>/scaling_summary.json     结构化实测结果（含单双卡对照）
    <out>/comparison_table.md      人读对比表
    <out>/fig_data_scaling_<mode>.png
    <out>/fig_speedup_single_vs_dual.png
    <out>/fig_sft_downstream_<mode>.png

多机编排（跨节点 torchrun）需结合具体集群网络环境配置，本仓只保留与机器无关的汇总与分析脚本。

用法：
    python scripts/scaling/collect_scaling.py --results-dir "$XL_OUT_DIR/scaling"
    python scripts/scaling/collect_scaling.py --results-dir /path/to/results --modes single,dual
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent


def scan_results(results_dir: Path) -> dict[str, dict[str, Path]]:
    """扫描 <results-dir>/<mode>/<tier>/ 目录，返回 {mode: {tier: dir}}。"""
    out: dict[str, dict[str, Path]] = {}
    if not results_dir.exists():
        return out
    for mode_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        tiers = {p.name: p for p in sorted(mode_dir.iterdir()) if p.is_dir()}
        if tiers:
            out[mode_dir.name] = tiers
    return out


def read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def tier_rows(mode: str, tier: str, tier_dir: Path) -> dict:
    pre = read_json(tier_dir / "pretrain_metrics.json")
    sft = read_json(tier_dir / "sft_metrics.json")
    failover = read_json(tier_dir / "failover_events.json")
    row = {"tier": tier, "mode": mode, "pretrain": None, "sft": None}
    if pre:
        mu = pre.get("muon", {})
        bp = mu.get("val_bpb_ppl", {}) or {}
        cfg = pre.get("config", {})
        dist = cfg.get("dist", {})
        world = int(dist.get("world_size", 1) or 1)
        tokens_seen = mu.get("tokens_seen")
        row["pretrain"] = {
            "docs": pre.get("train_records"),
            "train_tokens": pre.get("train_tokens"),
            "val_tokens": pre.get("val_tokens"),
            "model_params": pre.get("model_params"),
            "steps": cfg.get("steps"),
            "batch_size": cfg.get("batch_size"),
            "block_size": cfg.get("block_size"),
            "world_size": world,
            "wall_seconds": mu.get("wall_seconds"),
            "tokens_seen": tokens_seen,
            "tokens_per_sec": round(tokens_seen / mu["wall_seconds"], 1)
            if tokens_seen and mu.get("wall_seconds") else None,
            "final_val_loss": mu.get("final_val_loss"),
            "best_val_loss": mu.get("best_val_loss"),
            "best_val_step": mu.get("best_val_step"),
            "bpb": bp.get("bpb"),
            "ppl": bp.get("ppl"),
            "eval_tokens": bp.get("eval_tokens"),
            "train_data_cpu_mb": mu.get("train_data_cpu_mb"),
            "peak_vram_allocated_mb": (mu.get("memory") or {}).get("peak_vram_allocated_mb"),
            "peak_vram_reserved_mb": (mu.get("memory") or {}).get("peak_vram_reserved_mb"),
            "train_source": pre.get("train_source"),
            "val_source": pre.get("val_source"),
            "run_id": pre.get("run_id"),
        }
    if sft:
        row["sft"] = {
            "train_pairs": sft.get("train_pairs"),
            "val_pairs": sft.get("val_pairs"),
            "wall_seconds": sft.get("wall_seconds"),
            "final_val_loss": sft.get("final_val_loss"),
            "val_answer_hit_rate": sft.get("val_answer_hit_rate"),
            "train_answer_hit_rate": sft.get("train_answer_hit_rate"),
            "test_bpb": (sft.get("test_bpb_ppl") or {}).get("bpb"),
            "test_ppl": (sft.get("test_bpb_ppl") or {}).get("ppl"),
            "peak_vram_allocated_mb": (sft.get("memory") or {}).get("peak_vram_allocated_mb"),
            "peak_vram_reserved_mb": (sft.get("memory") or {}).get("peak_vram_reserved_mb"),
        }
    row["failover_events"] = failover if isinstance(failover, list) else []
    return row


def speedup_table(rows_by_mode: dict[str, list[dict]]) -> list[dict]:
    single = {r["tier"]: r for r in rows_by_mode.get("single", [])}
    dual = {r["tier"]: r for r in rows_by_mode.get("dual", [])}
    out = []
    for tier in sorted(set(single) & set(dual)):
        s, d = single[tier], dual[tier]
        if not s.get("pretrain") or not d.get("pretrain"):
            continue
        sw = s["pretrain"]["wall_seconds"]
        dw = d["pretrain"]["wall_seconds"]
        stp = s["pretrain"]["tokens_per_sec"]
        dtp = d["pretrain"]["tokens_per_sec"]
        out.append({
            "tier": tier,
            "single_wall_seconds": sw,
            "dual_wall_seconds": dw,
            "single_tokens_per_sec": stp,
            "dual_tokens_per_sec": dtp,
            "speedup_wall": round(sw / dw, 3) if sw and dw else None,
            "speedup_throughput": round(dtp / stp, 3) if stp and dtp else None,
            "single_final_val_loss": s["pretrain"]["final_val_loss"],
            "dual_final_val_loss": d["pretrain"]["final_val_loss"],
            "single_bpb": s["pretrain"]["bpb"],
            "dual_bpb": d["pretrain"]["bpb"],
        })
    return out


def write_markdown(summary: dict, path: Path) -> None:
    lines = [
        "# 数据量级扩展对比训练实测表",
        "",
        f"生成时间: {summary['generated_at']}",
        "",
        "## 各量级预训练指标",
        "",
        "| 模式/量级 | docs | train_tokens | tokens_seen | world_size | wall_s | tokens/s | final_val_loss | best_val_loss | BPB | PPL | Peak VRAM (MB) | CPU Data (MB) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for mode, rows in summary["modes"].items():
        for r in rows:
            p = r.get("pretrain")
            if not p:
                continue
            vram = p.get("peak_vram_allocated_mb")
            dmb = p.get("train_data_cpu_mb")
            tseen = p.get("tokens_seen")
            lines.append(
                f"| {mode}/{r['tier']} | {p['docs']} | {p['train_tokens']:,} | "
                f"{f'{tseen:,}' if tseen else 'n/a'} | "
                f"{p['world_size']} | {p['wall_seconds']} | {p['tokens_per_sec']:,} | "
                f"{p['final_val_loss']} | {p['best_val_loss']} | {p['bpb']} | {p['ppl']} | "
                f"{vram if vram is not None else 'n/a'} | {dmb if dmb is not None else 'n/a'} |"
            )
    lines += ["", "## 接续 SFT 下游效果", "",
              "| 模式/量级 | SFT wall_s | SFT val_loss | 答案命中率 | SFT 后 BPB | SFT 后 PPL | SFT Peak VRAM (MB) |", "|---|---|---|---|---|---|---|"]
    for mode, rows in summary["modes"].items():
        for r in rows:
            s = r.get("sft")
            if not s:
                continue
            svram = s.get("peak_vram_allocated_mb")
            lines.append(
                f"| {mode}/{r['tier']} | {s['wall_seconds']} | {s['final_val_loss']} | "
                f"{s['val_answer_hit_rate']} | {s['test_bpb']} | {s['test_ppl']} | "
                f"{svram if svram is not None else 'n/a'} |"
            )
    if summary.get("single_vs_dual"):
        lines += ["", "## 单卡 vs 双卡同配速对照", "",
                  "| 量级 | 单卡 wall_s | 双卡 wall_s | 单卡 tokens/s | 双卡 tokens/s | 加速比(耗时) | 加速比(吞吐) |",
                  "|---|---|---|---|---|---|---|"]
        for r in summary["single_vs_dual"]:
            lines.append(
                f"| {r['tier']} | {r['single_wall_seconds']} | {r['dual_wall_seconds']} | "
                f"{r['single_tokens_per_sec']:,} | {r['dual_tokens_per_sec']:,} | "
                f"{r['speedup_wall']} | {r['speedup_throughput']} |"
            )
    fail_rows = [
        (mode, r["tier"], ev)
        for mode, rows in summary["modes"].items()
        for r in rows
        for ev in (r.get("failover_events") or [])
    ]
    if fail_rows:
        lines += [
            "",
            "## 故障转移与断点续训记录",
            "",
            "| 模式/量级 | 阶段 | 原因 | 故障前 world_size | 恢复位置 | 续训起始 step |",
            "|---|---|---|---|---|---|",
        ]
        for mode, tname, ev in fail_rows:
            lines.append(
                f"| {mode}/{tname} | {ev.get('stage')} | {ev.get('reason')} | "
                f"{ev.get('from_world_size')} | {ev.get('recovered_on')} (world_size={ev.get('to_world_size')}) | "
                f"{ev.get('resumed_step', 0)} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_scaling(rows: list[dict], out: Path, mode: str) -> None:
    rows = [r for r in rows if r.get("pretrain")]
    if len(rows) < 2:
        return
    xs = [r["pretrain"]["train_tokens"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, key, label in [
        (axes[0], "final_val_loss", "final val loss"),
        (axes[1], "bpb", "BPB (bits per byte)"),
    ]:
        ys = [r["pretrain"][key] for r in rows]
        bs = [r["pretrain"]["best_val_loss"] for r in rows] if key == "final_val_loss" else None
        ax.plot(xs, ys, "o-", label=f"{mode} final")
        if bs:
            ax.plot(xs, bs, "s--", alpha=0.7, label=f"{mode} best")
        ax.set_xscale("log")
        ax.set_xlabel("training tokens (log)")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle(f"Data scaling on {mode}")
    fig.tight_layout()
    fig.savefig(out / f"fig_data_scaling_{mode}.png", dpi=140)
    plt.close(fig)


def plot_speedup(sp: list[dict], out: Path) -> None:
    if not sp:
        return
    tiers = [r["tier"] for r in sp]
    x = range(len(tiers))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].bar([i - 0.2 for i in x], [r["single_wall_seconds"] for r in sp], width=0.4, label="single")
    axes[0].bar([i + 0.2 for i in x], [r["dual_wall_seconds"] for r in sp], width=0.4, label="dual")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(tiers, rotation=20, ha="right")
    axes[0].set_ylabel("wall seconds")
    axes[0].set_title("Wall time, same config")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="y")
    axes[1].bar([i - 0.2 for i in x], [r["single_tokens_per_sec"] for r in sp], width=0.4, label="single")
    axes[1].bar([i + 0.2 for i in x], [r["dual_tokens_per_sec"] for r in sp], width=0.4, label="dual")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(tiers, rotation=20, ha="right")
    axes[1].set_ylabel("tokens / s")
    axes[1].set_title("Throughput")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out / "fig_speedup_single_vs_dual.png", dpi=140)
    plt.close(fig)


def plot_sft(rows: list[dict], out: Path, mode: str) -> None:
    rows = [r for r in rows if r.get("sft") and r.get("pretrain")]
    if len(rows) < 2:
        return
    xs = [r["pretrain"]["train_tokens"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(xs, [r["sft"]["val_answer_hit_rate"] for r in rows], "o-", color="tab:green")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("pretraining tokens (log)")
    axes[0].set_ylabel("SFT answer hit rate")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(xs, [r["pretrain"]["bpb"] for r in rows], "o-", label="after pretrain")
    axes[1].plot(xs, [r["sft"]["test_bpb"] for r in rows], "s--", label="after SFT")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("pretraining tokens (log)")
    axes[1].set_ylabel("BPB on fixed heldout set")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.suptitle(f"SFT downstream effect on {mode}")
    fig.tight_layout()
    fig.savefig(out / f"fig_sft_downstream_{mode}.png", dpi=140)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="汇总数据量级扩展实验指标（机器无关，只读本地目录）")
    ap.add_argument("--results-dir", dest="results_dir", default=None,
                    help="结果目录（缺省 $XL_OUT_DIR/scaling）")
    ap.add_argument("--modes", default=None, help="逗号分隔的 mode 过滤（缺省全部）")
    ap.add_argument("--out", default=None, help="输出目录（缺省与 --results-dir 相同）")
    args = ap.parse_args()

    results_dir = Path(args.results_dir or os.environ.get("XL_OUT_DIR") or ".")
    if args.results_dir is None and os.environ.get("XL_OUT_DIR"):
        results_dir = Path(os.environ["XL_OUT_DIR"]) / "scaling"
    out = Path(args.out) if args.out else results_dir
    out.mkdir(parents=True, exist_ok=True)

    scanned = scan_results(results_dir)
    if args.modes:
        wanted = {m.strip() for m in args.modes.split(",") if m.strip()}
        scanned = {m: t for m, t in scanned.items() if m in wanted}
    if not scanned:
        print(f"[warn] {results_dir} 下未找到 <mode>/<tier>/ 结构的结果目录")

    rows_by_mode = {
        mode: [tier_rows(mode, tier, tdir) for tier, tdir in tiers.items()]
        for mode, tiers in scanned.items()
    }
    summary = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "results_dir": str(results_dir),
        "modes": rows_by_mode,
        "single_vs_dual": speedup_table(rows_by_mode),
    }
    (out / "scaling_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_markdown(summary, out / "comparison_table.md")
    for mode, rows in rows_by_mode.items():
        plot_scaling(rows, out, mode)
        plot_sft(rows, out, mode)
    plot_speedup(summary["single_vs_dual"], out)
    print(f"[ok] 汇总写入 {out}")
    for mode, rows in rows_by_mode.items():
        got = sum(1 for r in rows if r.get("pretrain"))
        print(f"  {mode}: {got}/{len(rows)} 个量级有预训练指标")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
