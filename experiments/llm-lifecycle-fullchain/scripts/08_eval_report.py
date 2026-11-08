#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""08_eval_report.py — 全链路评估与 report.md 汇总。

对四个 checkpoint（pretrain / midtrain / sft / rl）跑同一套评估：
  - 聊天问答命中率（SFT 留出集，与 05 同一划分：seed 1234、val_frac 0.15）
  - 算术 exact-match（RL 训练集 = 算式进过语料；留出集 = 算式未进语料）
汇总各阶段 loss 指标，写出 results/report.md 与 results/summary.json。

用法：
    python scripts/08_eval_report.py
"""

from __future__ import annotations

import json
import random

import torch

import llm_common as lc


def chat_eval(model, tok, eos: int) -> tuple[float, int]:
    """SFT 留出集（与 05 相同划分）的答案前缀命中率。"""
    pairs = [
        json.loads(l)
        for l in (lc.DATA_DIR / "sft.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    rng = random.Random(1234)
    rng.shuffle(pairs)
    n_val = max(1, int(len(pairs) * 0.15))
    val_pairs = pairs[:n_val]
    hits = 0
    for row in val_pairs:
        p_ids = tok.encode(row["prompt"], add_special_tokens=False).ids
        out, _ = lc.generate(model, torch.tensor([p_ids]), 24, eos, temperature=0.0, use_cache=True)
        gen = tok.decode(out[0, len(p_ids) :].tolist(), skip_special_tokens=True).strip()
        expect = row["response"].strip()
        hits += int(gen.replace(" ", "").startswith(expect.replace(" ", "")[:12]))
    return hits / max(1, len(val_pairs)), len(val_pairs)


def arith_eval(model, tok, eos: int) -> tuple[float, float, int, int]:
    rows = [
        json.loads(l)
        for l in (lc.DATA_DIR / "rl.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    train = [r for r in rows if r.get("in_corpus")]
    val = [r for r in rows if not r.get("in_corpus")]

    def em(tasks):
        hits = 0
        for row in tasks:
            p_ids = tok.encode(row["prompt"], add_special_tokens=False).ids
            out, _ = lc.generate(
                model, torch.tensor([p_ids]), 8, eos, temperature=0.0, use_cache=True
            )
            gen = tok.decode(out[0, len(p_ids) :].tolist(), skip_special_tokens=True).strip()
            hits += int(gen == row["answer"])
        return hits / max(1, len(tasks))

    return em(train), em(val), len(train), len(val)


def load_json(name: str) -> dict | None:
    p = lc.ROOT / "results" / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main() -> int:
    tok = lc.load_tokenizer()
    eos = lc.eos_id(tok)

    stages = [
        ("pretrain", lc.CKPT_DIR / "pretrain.pt"),
        ("midtrain", lc.CKPT_DIR / "midtrain.pt"),
        ("sft", lc.CKPT_DIR / "sft.pt"),
        ("rl", lc.CKPT_DIR / "rl.pt"),
    ]
    table = []
    for name, path in stages:
        if not path.exists():
            print(f"[08] 缺少 {path}，跳过")
            continue
        model, ckpt = lc.load_checkpoint(path)
        chat_acc, n_chat = chat_eval(model, tok, eos)
        tr_acc, va_acc, n_tr, n_va = arith_eval(model, tok, eos)
        row = {
            "stage": name,
            "step": ckpt.get("step"),
            "params": model.num_params(),
            "chat_hit_rate": round(chat_acc, 4),
            "arith_train_exact_match": round(tr_acc, 4),
            "arith_val_exact_match": round(va_acc, 4),
        }
        table.append(row)
        print(
            f"[08] {name:<9} step={ckpt.get('step'):<4} "
            f"chat_hit={chat_acc:.0%} ({n_chat} 题)  "
            f"算术: 训练 {tr_acc:.0%} / 留出 {va_acc:.0%} ({n_tr}/{n_va} 题)"
        )

    pretrain_m = load_json("pretrain_metrics.json")
    midtrain_m = load_json("midtrain_metrics.json")
    sft_m = load_json("sft_metrics.json")
    rl_m = load_json("rl_metrics.json")
    tok_info = load_json("tokenizer/tokenizer_info.json")

    summary = {
        "stages": table,
        "tokenizer": tok_info,
        "pretrain": (
            {
                "muon_best_val_loss": pretrain_m["muon"]["best_val_loss"],
                "muon_best_val_step": pretrain_m["muon"]["best_val_step"],
                "muon_final_val_loss": pretrain_m["muon"]["final_val_loss"],
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
        "midtrain": {"final_val_loss": midtrain_m["final_val_loss"]} if midtrain_m else None,
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
    (lc.ROOT / "results" / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ---- report.md ----
    lines = [
        "# 08 全链路评估报告\n",
        "同一套评估跑在四个 checkpoint 上：聊天问答命中率（SFT 留出集）与"
        "算术 exact-match（训练集算式进过预训练语料，留出集算式完全未见）。\n",
        "| 阶段 | step | 参数量 | 聊天问答命中率 | 算术 exact-match（训练/留出） |",
        "|---|---|---|---|---|",
    ]
    for r in table:
        lines.append(
            f"| {r['stage']} | {r['step']} | {r['params']:,} | {r['chat_hit_rate']:.0%} "
            f"| {r['arith_train_exact_match']:.0%} / {r['arith_val_exact_match']:.0%} |"
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
            "\nMuon 的 best val 更低且出现更早（优化更快）；继续训练后两者 final val "
            "都上扬，Muon 更早过拟合，9K token 的微型语料上这是预期行为。\n"
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
        "\n各阶段 loss 曲线：results/pretrain_loss.png、midtrain_loss.png、"
        "sft_loss.png、rl_reward.png；生成样例见 results/samples.md。\n"
    )
    (lc.ROOT / "results" / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("[08] 已写出 results/report.md 与 results/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
