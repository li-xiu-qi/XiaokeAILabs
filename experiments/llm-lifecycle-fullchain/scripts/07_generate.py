#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""07_generate.py — 自回归生成：KV cache 引擎 + 贪心/温度采样 + 提速对比。

加载 RL（或 SFT）权重，对一组 prompt 做解码：
  - greedy：argmax，确定性
  - temperature：softmax(logits/T) 后多项式采样，可配 top_k / top_p
  - KV cache 与 naive 全前向的提速对比（同 prompt、同参数、逐 token 计时）
默认同时用 midtrain / pretrain 模型对照同一批 prompt，展示各阶段行为变化。
产物：results/samples.md、results/samples.json。

用法：
    python scripts/07_generate.py [--checkpoint results/checkpoints/rl.pt]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import llm_common as lc

PROMPTS = [
    "User: What is the capital of Japan?\nAssistant:",
    "User: Which city is the capital of Egypt?\nAssistant:",
    "User: What color is a banana?\nAssistant:",
    "User: What does Carol do?\nAssistant:",
    "User: What is 7 plus 5?\nAssistant:",
    "User: Hello\nAssistant:",
    "User: Who are you?\nAssistant:",
    "The capital of Peru is",
    "The small robot reads the",
]

CONT_PROMPTS = [
    "The capital of Peru is",
    "The small robot reads the",
    "The quiet student finds the",
]


def run_prompts(model, tok, prompts, temperature, top_k, top_p, max_new, seed, use_cache=True):
    eos = lc.eos_id(tok)
    rows = []
    for i, p in enumerate(prompts):
        torch.manual_seed(seed + i)  # 每个 prompt 固定采样种子，结果可复现
        ids = tok.encode(p, add_special_tokens=False).ids
        idx = torch.tensor([ids], dtype=torch.long)
        t0 = time.time()
        out, n_new = lc.generate(
            model, idx, max_new, eos, temperature, top_k, top_p, use_cache=use_cache, seed=seed + i
        )
        dt = time.time() - t0
        text = tok.decode(out[0, len(ids) :].tolist(), skip_special_tokens=True).strip()
        rows.append(
            {
                "prompt": p.replace("\n", "\\n"),
                "mode": (
                    "greedy"
                    if temperature <= 0
                    else f"temperature={temperature}, top_k={top_k}, top_p={top_p}"
                ),
                "generated": text,
                "new_tokens": n_new,
                "seconds": round(dt, 3),
                "tokens_per_sec": round(n_new / max(dt, 1e-6), 1),
            }
        )
    return rows


def speed_benchmark(model, tok, max_new: int = 96) -> dict:
    """同一 prompt 分别用 KV cache 与 naive 前向解码，对比 tokens/s。

    两个细节保证基准有效：
    - 用续写 prompt 并传 eos=-1（永不停止），两边都跑满 max_new 步；聊天 prompt 会
      早早吐出 eos，生成太短则 cache 的节省被每步固定开销吃掉
    - 分 batch=1 与 batch=8：batch 越大，naive 每步要重算的全长序列越多，cache 优势越明显
    """
    prompt = "The capital of Peru is"
    ids = tok.encode(prompt, add_special_tokens=False).ids
    out = {}
    for batch in (1, 8):
        idx = torch.tensor([ids] * batch, dtype=torch.long)
        row = {}
        for use_cache in (True, False):
            torch.manual_seed(0)
            t0 = time.time()
            _, n_new = lc.generate(
                model, idx.clone(), max_new, -1, temperature=0.0, use_cache=use_cache
            )
            dt = time.time() - t0
            total_new = n_new * batch
            row["kv_cache" if use_cache else "naive"] = {
                "new_tokens": n_new,
                "seconds": round(dt, 3),
                "tokens_per_sec": round(total_new / max(dt, 1e-6), 1),
            }
        row["speedup"] = round(
            row["kv_cache"]["tokens_per_sec"] / max(row["naive"]["tokens_per_sec"], 1e-6), 2
        )
        out[f"batch{batch}"] = row
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="贪心 + 温度采样生成")
    ap.add_argument("--checkpoint", default=str(lc.CKPT_DIR / "rl.pt"))
    ap.add_argument(
        "--compare", default=str(lc.CKPT_DIR / "midtrain.pt"), help="对照模型，传空串跳过"
    )
    ap.add_argument(
        "--compare2", default=str(lc.CKPT_DIR / "pretrain.pt"), help="第二对照模型，传空串跳过"
    )
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    tok = lc.load_tokenizer()
    model, ckpt = lc.load_checkpoint(Path(args.checkpoint))
    print(
        f"[07] 载入 {args.checkpoint}（stage={ckpt.get('stage')}, "
        f"step={ckpt.get('step')}, {model.num_params():,} 参数）"
    )

    result = {
        "checkpoint": str(Path(args.checkpoint).relative_to(lc.ROOT)),
        "tokenizer": str(lc.TOK_PATH.relative_to(lc.ROOT)),
        "decode_params": {
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "max_new": args.max_new,
            "seed": args.seed,
        },
        "greedy": [],
        "sampling": [],
        "compare_midtrain_greedy": [],
        "compare_pretrain_greedy": [],
        "temperature_effect": {},
        "speed": {},
    }

    result["greedy"] = run_prompts(model, tok, PROMPTS, 0.0, 0, 1.0, args.max_new, args.seed)
    result["sampling"] = run_prompts(
        model, tok, PROMPTS, args.temperature, args.top_k, args.top_p, args.max_new, args.seed
    )

    if args.compare and Path(args.compare).exists():
        base, _ = lc.load_checkpoint(Path(args.compare))
        result["compare_midtrain_greedy"] = run_prompts(
            base, tok, PROMPTS, 0.0, 0, 1.0, args.max_new, args.seed
        )
        print(f"[07] 对照模型 {args.compare} 生成完成")
    if args.compare2 and Path(args.compare2).exists():
        base2, _ = lc.load_checkpoint(Path(args.compare2))
        result["compare_pretrain_greedy"] = run_prompts(
            base2, tok, PROMPTS, 0.0, 0, 1.0, args.max_new, args.seed
        )
        print(f"[07] 对照模型 {args.compare2} 生成完成")

    cont_low = run_prompts(model, tok, CONT_PROMPTS, 0.8, 0, 1.0, 12, args.seed)
    cont_high = run_prompts(model, tok, CONT_PROMPTS, 2.5, 0, 1.0, 12, args.seed)
    result["temperature_effect"] = {"low": cont_low, "high": cont_high}

    result["speed"] = speed_benchmark(model, tok)
    for b in ("batch1", "batch8"):
        s = result["speed"][b]
        print(
            f"[07] 解码速度 {b}: KV cache {s['kv_cache']['tokens_per_sec']} tok/s "
            f"vs naive {s['naive']['tokens_per_sec']} tok/s (提速 {s['speedup']}x)"
        )

    # ---- 输出与落盘 ----
    def cell(s: str) -> str:
        return s.replace("|", "\\|").replace("\n", " ")

    lines = [
        "# 07 生成结果\n",
        f"- 主模型：`{result['checkpoint']}`（stage={ckpt.get('stage')}）",
        f"- 采样参数：temperature={args.temperature}, top_k={args.top_k}, "
        f"top_p={args.top_p}, max_new={args.max_new}, seed={args.seed}\n",
    ]
    for section, title in [
        ("greedy", "贪心解码（temperature=0）"),
        ("sampling", "温度采样"),
        ("compare_midtrain_greedy", "对照：midtrain 模型贪心解码"),
        ("compare_pretrain_greedy", "对照：预训练模型贪心解码"),
    ]:
        if not result[section]:
            continue
        lines.append(f"\n## {title}\n")
        lines.append("| prompt | 生成 | 新 token 数 | tok/s |")
        lines.append("|---|---|---|---|")
        for r in result[section]:
            lines.append(
                f"| `{r['prompt']}` | {cell(r['generated'])} "
                f"| {r['new_tokens']} | {r['tokens_per_sec']} |"
            )
    lines.append("\n## 温度效应对照（续写 prompt，T=0.8 vs T=2.5）\n")
    lines.append("| prompt | T=0.8 | T=2.5 |")
    lines.append("|---|---|---|")
    for lo, hi in zip(cont_low, cont_high):
        lines.append(
            f"| `{lo['prompt']}` | {cell(lo['generated'])} " f"| {cell(hi['generated'])} |"
        )
    lines.append("\n## KV cache 提速对比（生成长度 96 token）\n")
    lines.append("| 场景 | KV cache tokens/s | naive tokens/s | 提速 |")
    lines.append("|---|---|---|---|")
    for b, label in (("batch1", "batch=1"), ("batch8", "batch=8（服务场景）")):
        s = result["speed"][b]
        lines.append(
            f"| {label} | {s['kv_cache']['tokens_per_sec']} "
            f"| {s['naive']['tokens_per_sec']} | {s['speedup']}x |"
        )
    lines.append("\n")

    md = "\n".join(lines) + "\n"
    (lc.ROOT / "results" / "samples.md").write_text(md, encoding="utf-8")
    (lc.ROOT / "results" / "samples.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n[07] 贪心解码示例：")
    for r in result["greedy"][:5]:
        print(f"  {r['prompt']:<48} -> {r['generated'][:44]!r}")
    print("[07] 温度采样示例：")
    for r in result["sampling"][:5]:
        print(f"  {r['prompt']:<48} -> {r['generated'][:44]!r}")
    if result["compare_midtrain_greedy"]:
        print("[07] midtrain 模型贪心（对照）：")
        for r in result["compare_midtrain_greedy"][:5]:
            print(f"  {r['prompt']:<48} -> {r['generated'][:44]!r}")
    if result["compare_pretrain_greedy"]:
        print("[07] 预训练模型贪心（对照）：")
        for r in result["compare_pretrain_greedy"][:5]:
            print(f"  {r['prompt']:<48} -> {r['generated'][:44]!r}")
    print("[07] 温度效应对照（T=0.8 vs T=2.5）：")
    for lo, hi in zip(cont_low, cont_high):
        print(f"  {lo['prompt']:<30} | {lo['generated'][:24]!r} | {hi['generated'][:24]!r}")
    print("[07] 已写出 results/samples.md 与 results/samples.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
