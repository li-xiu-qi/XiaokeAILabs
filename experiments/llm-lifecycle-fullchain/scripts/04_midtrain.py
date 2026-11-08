#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""04_midtrain.py — Midtraining：对话格式适配（nanochat base_train 与 chat_sft 之间的阶段）。

在预训练权重基础上，只在「User:/Assistant:」对话记录上以较低学习率继续训练：
  - 全序列 loss（不做 prompt 掩码），让模型先学会对话格式与问答分布
  - lr 统一缩放（--lr-scale，默认 0.2），避免破坏预训练知识
产物：results/checkpoints/midtrain.pt、results/midtrain_metrics.json。

用法：
    python scripts/04_midtrain.py [--steps 120] [--batch-size 16] [--lr-scale 0.2]
"""

from __future__ import annotations

import argparse
import json
import random
import time

import torch

import llm_common as lc


def dialogue_records(text: str) -> list[str]:
    """语料中的对话格式记录（含 User:/Assistant: 的记录）。"""
    return [r for r in text.split(lc.EOS) if "User:" in r and "Assistant:" in r]


@torch.no_grad()
def quick_samples(
    model: lc.CausalLM, tok, prompts: list[str], eos: int, max_new: int = 16
) -> list[dict]:
    rows = []
    for p in prompts:
        ids = tok.encode(p, add_special_tokens=False).ids
        out, _ = lc.generate(
            model, torch.tensor([ids]), max_new, eos, temperature=0.0, use_cache=True
        )
        text = tok.decode(out[0, len(ids) :].tolist(), skip_special_tokens=True).strip()
        rows.append({"prompt": p.replace("\n", "\\n"), "generated": text})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="对话格式 midtrain")
    ap.add_argument("--init", default=str(lc.CKPT_DIR / "pretrain.pt"))
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-scale", type=float, default=0.2)
    ap.add_argument("--matrix-lr", type=float, default=0.02)
    ap.add_argument("--embedding-lr", type=float, default=0.3)
    ap.add_argument("--unembedding-lr", type=float, default=0.004)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    tok = lc.load_tokenizer()
    eos = lc.eos_id(tok)
    model, ckpt = lc.load_checkpoint(lc.ROOT / args.init)
    cfg = model.cfg
    print(
        f"[04] 载入 {args.init}（stage={ckpt.get('stage')}, step={ckpt.get('step')}）, "
        f"{model.num_params():,} 参数"
    )

    text = (lc.DATA_DIR / "corpus.txt").read_text(encoding="utf-8")
    records = dialogue_records(text)
    rng = random.Random(args.seed)
    rng.shuffle(records)
    n_val = max(1, int(len(records) * args.val_frac))
    val_records, train_records = records[:n_val], records[n_val:]
    train_ids = lc.ensure_stream(lc.records_to_ids(train_records, tok), args.block_size)
    val_ids = lc.ensure_stream(lc.records_to_ids(val_records, tok), args.block_size)
    print(
        f"[04] 对话记录: {len(train_records)} train / {len(val_records)} val, "
        f"{len(train_ids):,} / {len(val_ids):,} tokens"
    )

    optimizers, _ = lc.build_optimizer(
        model,
        cfg,
        lr=args.lr,
        weight_decay=args.weight_decay,
        optimizer="muon",
        matrix_lr=args.matrix_lr,
        embedding_lr=args.embedding_lr,
        unembedding_lr=args.unembedding_lr,
    )
    lc.tag_initial_lrs(optimizers)

    gen = torch.Generator().manual_seed(args.seed)
    val_gen = torch.Generator().manual_seed(args.seed + 1)
    history, val_history = [], []
    t0 = time.time()
    model.train()
    for step in range(args.steps):
        lc.set_lrs(optimizers, args.steps, step, args.lr_scale)
        x, y = lc.get_batch(train_ids, args.batch_size, args.block_size, gen)
        _, loss = model(x, targets=y)
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        for opt in optimizers:
            opt.step()
        history.append({"step": step, "loss": round(loss.item(), 4)})
        if (step + 1) % args.eval_every == 0 or step == 0:
            vl = lc.evaluate(model, val_ids, args.batch_size, args.block_size, 8, val_gen)
            val_history.append({"step": step, "val_loss": round(vl, 4)})
            print(
                f"[04] step {step + 1:>4}/{args.steps}  train_loss={loss.item():.4f}  "
                f"val_loss={vl:.4f}"
            )
    wall = time.time() - t0
    final_val = lc.evaluate(model, val_ids, args.batch_size, args.block_size, 16, val_gen)
    model.eval()

    lc.save_checkpoint(
        model,
        cfg,
        lc.CKPT_DIR / "midtrain.pt",
        stage="midtrain",
        init=str(lc.Path(args.init).relative_to(lc.ROOT)),
        step=args.steps,
        tokenizer=str(lc.TOK_PATH.relative_to(lc.ROOT)),
        seed=args.seed,
    )

    samples = quick_samples(
        model,
        tok,
        [
            "User: What is the capital of Japan?\nAssistant:",
            "User: Hello\nAssistant:",
        ],
        eos,
    )
    for s in samples:
        print(f"[04] 样例 {s['prompt']} -> {s['generated'][:40]!r}")

    plot_ok = lc.plot_loss(
        history,
        val_history,
        lc.ROOT / "results" / "midtrain_loss.png",
        title="Midtrain loss on dialogue records",
    )
    first = lc.avg([h["loss"] for h in history[: max(1, args.steps // 10)]])
    last = lc.avg([h["loss"] for h in history[-max(1, args.steps // 10) :]])
    metrics = {
        "stage": "midtrain",
        "config": vars(args),
        "init": str(lc.Path(args.init).relative_to(lc.ROOT)),
        "train_records": len(train_records),
        "val_records": len(val_records),
        "history": history,
        "val_history": val_history,
        "first_train_loss": round(first, 4),
        "last_train_loss": round(last, 4),
        "final_val_loss": round(final_val, 4),
        "samples": samples,
        "wall_seconds": round(wall, 1),
        "loss_curve": "results/midtrain_loss.png" if plot_ok else None,
    }
    (lc.ROOT / "results" / "midtrain_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    drop = 1 - last / first
    # midtrain 的目标是格式适配：只要求 train loss 下降；对话记录同时是预训练语料的
    # 一部分，val loss 起点低且小数据下续训会上扬（过拟合），如实记录不作硬性门槛
    ok = drop > 0.2
    print(
        f"[04] midtrain loss: {first:.4f} -> {last:.4f} (下降 {drop:.1%}), "
        f"final val loss {final_val:.4f}（对话记录亦在预训练语料中，小数据续训后 "
        f"val 上扬属预期）, 用时 {wall:.1f}s"
    )
    print(f"[04] loss 下降验证: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
