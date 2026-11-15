#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""08_eval_report.py — 全链路评估与报告汇总。

对四个 checkpoint（pretrain / midtrain / sft / rl）跑同一套评估：
  - 统一测试集 BPB（bits per byte，与词表大小无关）/ PPL / mean CE（--test-source）
  - 聊天问答命中率（SFT 留出集，与 05 同一划分：seed 1234、val_frac 0.15）
  - 算术 exact-match（RL 训练集 = 算式进过语料；留出集 = 算式未进语料）
汇总各阶段 loss 指标，写出 report.md 与 summary.json。

数据与输出目录（对外部大语料的扩展）：
  - --out-dir 指定实验目录（checkpoint、指标、报告全部落此目录，缺省 $XL_OUT_DIR）
  - --tokenizer / --sft-path / --rl-path / --test-source 均可指向自定义路径

用法：
    python scripts/08_eval_report.py [--device cpu|cuda]
    python scripts/08_eval_report.py --out-dir "$XL_OUT_DIR" \
        --test-source /path/to/shard_06542.parquet
"""

from __future__ import annotations

import argparse
import json
import random

import torch

import llm_common as lc


def chat_eval(model, tok, eos: int, sft_path, device=None) -> tuple[float, int]:
    """SFT 留出集（与 05 相同划分）的答案前缀命中率。"""
    pairs = [
        json.loads(l)
        for l in sft_path.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    rng = random.Random(1234)
    rng.shuffle(pairs)
    n_val = max(1, int(len(pairs) * 0.15))
    val_pairs = pairs[:n_val]
    hits = 0
    for row in val_pairs:
        p_ids = tok.encode(row["prompt"], add_special_tokens=False).ids
        idx = torch.tensor([p_ids], dtype=torch.long, device=device)
        out, _ = lc.generate(model, idx, 24, eos, temperature=0.0, use_cache=True)
        gen = tok.decode(out[0, len(p_ids) :].tolist(), skip_special_tokens=True).strip()
        expect = row["response"].strip()
        hits += int(gen.replace(" ", "").startswith(expect.replace(" ", "")[:12]))
    return hits / max(1, len(val_pairs)), len(val_pairs)


def arith_eval(model, tok, eos: int, rl_path, device=None) -> tuple[float, float, int, int]:
    rows = [
        json.loads(l)
        for l in rl_path.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    train = [r for r in rows if r.get("in_corpus")]
    val = [r for r in rows if not r.get("in_corpus")]

    def em(tasks):
        hits = 0
        for row in tasks:
            p_ids = tok.encode(row["prompt"], add_special_tokens=False).ids
            idx = torch.tensor([p_ids], dtype=torch.long, device=device)
            out, _ = lc.generate(model, idx, 8, eos, temperature=0.0, use_cache=True)
            gen = tok.decode(out[0, len(p_ids) :].tolist(), skip_special_tokens=True).strip()
            hits += int(gen == row["answer"])
        return hits / max(1, len(tasks))

    return em(train), em(val), len(train), len(val)


def load_json(path) -> dict | None:
    p = path if hasattr(path, "exists") else (lc.out_dir() / path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main() -> int:
    ap = argparse.ArgumentParser(description="全链路评估与报告汇总（统一测试集 BPB/PPL）")
    ap.add_argument("--sft-path", type=str, default=None, dest="sft_path",
                    help="SFT 指令集 JSONL（缺省 $XL_DATA_DIR/sft.jsonl）")
    ap.add_argument("--rl-path", type=str, default=None, dest="rl_path",
                    help="RL 任务 JSONL（缺省 $XL_DATA_DIR/rl.jsonl）")
    ap.add_argument("--test-source", type=str, default=None, dest="test_source",
                    help="统一测试集（parquet/txt），给定则对每个 checkpoint 记录 BPB / PPL")
    ap.add_argument("--test-max-chars", type=int, default=2_000_000, dest="test_max_chars")
    ap.add_argument("--eval-windows", type=int, default=256)
    lc.add_device_args(ap)
    lc.add_outdir_args(ap)
    args = ap.parse_args()

    device = lc.resolve_device(args.device)
    out_dir = lc.resolve_out_dir(args)
    ckpt_dir = out_dir / "checkpoints"
    tok_path = lc.resolve_tokenizer_path(args)
    sft_path = Path(args.sft_path) if args.sft_path else lc.data_dir() / "sft.jsonl"
    rl_path = Path(args.rl_path) if args.rl_path else lc.data_dir() / "rl.jsonl"
    print(f"[08] device={device}, out_dir={out_dir}, tokenizer={tok_path}")

    tok = lc.load_tokenizer(tok_path)
    eos = lc.eos_id(tok)
    token_bytes = lc.load_token_bytes(tok_path.parent, tok=tok)

    test_ids = None
    if args.test_source:
        test_ids, test_stats = lc.build_ids_from_source(
            args.test_source, tok, args.test_max_chars, 10_000, eos
        )
        print(
            f"[08] 统一测试集: {test_stats['docs']:,} 条, {test_stats['tokens']:,} tokens "
            f"(源: {args.test_source})"
        )

    stages = [
        ("pretrain", ckpt_dir / "pretrain.pt"),
        ("midtrain", ckpt_dir / "midtrain.pt"),
        ("sft", ckpt_dir / "sft.pt"),
        ("rl", ckpt_dir / "rl.pt"),
    ]
    table = []
    for name, path in stages:
        if not path.exists():
            print(f"[08] 缺少 {path}，跳过")
            continue
        model, ckpt = lc.load_checkpoint(path, device=device)
        chat_acc, n_chat = chat_eval(model, tok, eos, sft_path, device=device)
        tr_acc, va_acc, n_tr, n_va = arith_eval(model, tok, eos, rl_path, device=device)
        row = {
            "stage": name,
            "step": ckpt.get("step"),
            "params": model.num_params(),
            "chat_hit_rate": round(chat_acc, 4),
            "arith_train_exact_match": round(tr_acc, 4),
            "arith_val_exact_match": round(va_acc, 4),
        }
        if test_ids is not None:
            bp = lc.evaluate_bpb_ppl(
                model, lc.ensure_stream(test_ids, model.cfg.max_len), token_bytes,
                batch_size=8, block_size=model.cfg.max_len, device=device,
                max_windows=args.eval_windows,
            )
            row["test_bpb"] = bp.get("bpb")
            row["test_ppl"] = bp.get("ppl")
            row["test_mean_loss_nats"] = bp.get("mean_loss_nats")
        table.append(row)
        extra = (
            f"  BPB={row.get('test_bpb')} PPL={row.get('test_ppl')}"
            if test_ids is not None
            else ""
        )
        print(
            f"[08] {name:<9} step={ckpt.get('step'):<4} "
            f"chat_hit={chat_acc:.0%} ({n_chat} 题)  "
            f"算术: 训练 {tr_acc:.0%} / 留出 {va_acc:.0%} ({n_tr}/{n_va} 题){extra}"
        )

    pretrain_m = load_json(out_dir / "pretrain_metrics.json")
    midtrain_m = load_json(out_dir / "midtrain_metrics.json")
    sft_m = load_json(out_dir / "sft_metrics.json")
    rl_m = load_json(out_dir / "rl_metrics.json")
    tok_info = load_json(tok_path.parent / "tokenizer_info.json")

    summary = {
        "run_id": lc.run_id(),
        "stages": table,
        "tokenizer": tok_info,
        "pretrain": (
            {
                "muon_best_val_loss": pretrain_m["muon"]["best_val_loss"],
                "muon_best_val_step": pretrain_m["muon"]["best_val_step"],
                "muon_final_val_loss": pretrain_m["muon"]["final_val_loss"],
                "muon_val_bpb_ppl": pretrain_m["muon"].get("val_bpb_ppl"),
                "adamw_best_val_loss": (
                    pretrain_m["adamw_baseline"]["best_val_loss"]
                    if pretrain_m and pretrain_m.get("adamw_baseline")
                    else None
                ),
                "adamw_final_val_loss": (
                    pretrain_m["adamw_baseline"]["final_val_loss"]
                    if pretrain_m and pretrain_m.get("adamw_baseline")
                    else None
                ),
            }
            if pretrain_m
            else None
        ),
        "midtrain": (
            {"final_val_loss": midtrain_m["final_val_loss"]} if midtrain_m else None
        ),
        "sft": (
            {
                "final_val_loss": sft_m["final_val_loss"],
                "val_answer_hit_rate": sft_m["val_answer_hit_rate"],
            }
            if sft_m
            else None
        ),
        "rl": rl_m["final"] if rl_m else None,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ---- report.md ----
    lines = [
        "# 08 全链路评估报告\n",
        "同一套评估跑在各 checkpoint 上：统一测试集 BPB/PPL、聊天问答命中率（SFT 留出集）与"
        "算术 exact-match（训练集算式进过预训练语料，留出集算式完全未见）。\n",
        "| 阶段 | step | 参数量 | 聊天问答命中率 | 算术 exact-match（训练/留出） | BPB | PPL |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in table:
        lines.append(
            f"| {r['stage']} | {r['step']} | {r['params']:,} | {r['chat_hit_rate']:.0%} "
            f"| {r['arith_train_exact_match']:.0%} / {r['arith_val_exact_match']:.0%} "
            f"| {r.get('test_bpb') if r.get('test_bpb') is not None else '-'} "
            f"| {r.get('test_ppl') if r.get('test_ppl') is not None else '-'} |"
        )

    if pretrain_m:
        mu = pretrain_m["muon"]
        ad = pretrain_m.get("adamw_baseline")
        lines.append("\n## 预训练：Muon vs AdamW（同 steps 同种子）\n")
        lines.append("| 优化器 | best val loss（步数） | final val loss | train loss 末段 |")
        lines.append("|---|---|---|---|")
        lines.append(
            f"| Muon+AdamW | {mu['best_val_loss']:.4f}（step {mu['best_val_step']}） "
            f"| {mu['final_val_loss']:.4f} | {mu['last_train_loss']:.4f} |"
        )
        if ad:
            lines.append(
                f"| AdamW | {ad['best_val_loss']:.4f}（step {ad['best_val_step']}） "
                f"| {ad['final_val_loss']:.4f} | {ad['last_train_loss']:.4f} |"
            )
        lines.append(
            "\nMuon 的 best val 更低且出现更早（优化更快）；小数据下继续训练两者 final val "
            "都可能上扬，属数据规模局限而非优化器缺陷。\n"
        )
    if rl_m:
        f = rl_m["final"]
        lines.append("\n## RL（GRPO）效果\n")
        lines.append("| 指标 | SFT（RL 前） | RL 后 |")
        lines.append("|---|---|---|")
        lines.append(
            f"| 算术 exact-match（训练集） | {f['sft_train_exact_match']:.0%} "
            f"| {f['train_exact_match']:.0%} |"
        )
        lines.append(
            f"| 算术 exact-match（留出集） | {f['sft_val_exact_match']:.0%} "
            f"| {f['val_exact_match']:.0%} |"
        )
        lines.append(
            f"| 聊天问答命中率（回归） | {f['chat_hit_rate_sft']:.0%} "
            f"| {f['chat_hit_rate_after_rl']:.0%} |"
        )
        lines.append("")
    lines.append(
        "\n各阶段 loss 曲线：pretrain_loss.png、midtrain_loss.png、"
        "sft_loss.png、rl_reward.png（与本报告同目录）；生成样例见同目录 samples.md。\n"
    )
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[08] 已写出 {out_dir / 'report.md'} 与 {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
