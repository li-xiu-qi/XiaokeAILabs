#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""05_sft.py — 指令微调（Supervised Fine-Tuning）。

在 midtrain 权重基础上，用 data/sft.jsonl 的指令对做 SFT：
  - 序列拼成 "User: {问题}\nAssistant:" + " {回答}" + eos
  - loss 只在回答（含 eos）上计算，prompt 位置置 -100 忽略
  - 训练后在验证集 prompt 上贪心生成，按答案前缀命中率评估
产物：results/checkpoints/sft.pt、results/sft_metrics.json、results/sft_loss.png。

用法：
    python scripts/05_sft.py [--steps 250] [--batch-size 16] [--lr-scale 0.2]
"""

from __future__ import annotations

import argparse
import json
import random
import time

import torch
import torch.nn.functional as F

import llm_common as lc


@torch.no_grad()
def hit_rate(
    model: lc.CausalLM, tok, pairs: list[dict], eos: int, max_new: int = 24
) -> tuple[float, list[dict]]:
    """贪心生成并按答案前缀命中率统计。"""
    hits, samples = 0, []
    for row in pairs:
        p_ids = tok.encode(row["prompt"], add_special_tokens=False).ids
        out, _ = lc.generate(
            model, torch.tensor([p_ids]), max_new, eos, temperature=0.0, use_cache=True
        )
        gen = tok.decode(out[0, len(p_ids) :].tolist(), skip_special_tokens=True).strip()
        expect = row["response"].strip()
        hit = gen.replace(" ", "").startswith(expect.replace(" ", "")[:12])
        hits += int(hit)
        samples.append(
            {"prompt": row["prompt"], "expected": expect, "generated": gen, "hit": bool(hit)}
        )
    return hits / max(1, len(pairs)), samples


def main() -> int:
    ap = argparse.ArgumentParser(description="指令微调 SFT")
    ap.add_argument("--init", default=str(lc.CKPT_DIR / "midtrain.pt"))
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-scale", type=float, default=0.2)
    ap.add_argument("--matrix-lr", type=float, default=0.02)
    ap.add_argument("--embedding-lr", type=float, default=0.3)
    ap.add_argument("--unembedding-lr", type=float, default=0.004)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    tok = lc.load_tokenizer()
    eos = lc.eos_id(tok)
    pad_id = tok.token_to_id(lc.PAD)
    model, ckpt = lc.load_checkpoint(lc.ROOT / args.init)
    cfg = model.cfg
    print(f"[05] 载入 {args.init}（stage={ckpt.get('stage')}）, {model.num_params():,} 参数")

    pairs = [
        json.loads(l)
        for l in (lc.DATA_DIR / "sft.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    n_val = max(1, int(len(pairs) * args.val_frac))
    val_pairs, train_pairs = pairs[:n_val], pairs[n_val:]

    def encode(row):
        p = tok.encode(row["prompt"], add_special_tokens=False).ids
        r = tok.encode(row["response"], add_special_tokens=False).ids + [eos]
        return p, r

    train_data = [encode(r) for r in train_pairs]
    val_data = [encode(r) for r in val_pairs]
    print(f"[05] SFT 数据: {len(train_data)} train / {len(val_data)} val 条指令对")

    def make_batch(data):
        seqs, labels = [], []
        for p, r in data:
            ids = (p + r)[: args.block_size]
            lab = ([-100] * len(p) + list(r))[: args.block_size]
            seqs.append(ids)
            labels.append(lab)
        maxlen = max(len(s) for s in seqs)
        x = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
        y = torch.full((len(seqs), maxlen), -100, dtype=torch.long)
        for i, (s, l) in enumerate(zip(seqs, labels)):
            x[i, : len(s)] = torch.tensor(s)
            y[i, : len(l)] = torch.tensor(l)
        return x, y

    def masked_loss(x, y):
        logits, _ = model(x)
        return F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)), y[:, 1:].reshape(-1), ignore_index=-100
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

    history, val_history = [], []
    t0 = time.time()
    model.train()
    for step in range(args.steps):
        lc.set_lrs(optimizers, args.steps, step, args.lr_scale)
        batch = rng.sample(train_data, k=min(args.batch_size, len(train_data)))
        x, y = make_batch(batch)
        loss = masked_loss(x, y)
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        for opt in optimizers:
            opt.step()
        history.append({"step": step, "loss": round(loss.item(), 4)})
        if (step + 1) % args.eval_every == 0 or step == 0:
            model.eval()
            with torch.no_grad():
                vx, vy = make_batch(val_data)
                vl = masked_loss(vx, vy).item()
            model.train()
            val_history.append({"step": step, "val_loss": round(vl, 4)})
            print(
                f"[05] step {step + 1:>4}/{args.steps}  sft_loss={loss.item():.4f}  "
                f"val_loss={vl:.4f}"
            )
    wall = time.time() - t0
    model.eval()

    acc, samples = hit_rate(model, tok, val_pairs, eos)
    train_acc, _ = hit_rate(model, tok, train_pairs[:20], eos)
    print(f"[05] 训练集命中率: {train_acc:.0%}, 留出集命中率: {acc:.0%}")
    for s in samples[:5]:
        print(
            f"[05] 样例 Q: {s['prompt'].splitlines()[0].replace('User: ', '')}  ->  "
            f"{s['generated'][:40]!r} (期望 {s['expected'][:30]!r}, "
            f"{'✓' if s['hit'] else '✗'})"
        )

    lc.save_checkpoint(
        model,
        cfg,
        lc.CKPT_DIR / "sft.pt",
        stage="sft",
        init=str(lc.Path(args.init).relative_to(lc.ROOT)),
        step=args.steps,
        tokenizer=str(lc.TOK_PATH.relative_to(lc.ROOT)),
        seed=args.seed,
    )

    plot_ok = lc.plot_loss(
        history,
        val_history,
        lc.ROOT / "results" / "sft_loss.png",
        title="SFT loss on instruction pairs",
        ylabel="cross entropy (response only)",
    )
    first = lc.avg([h["loss"] for h in history[: max(1, args.steps // 20)]])
    last = lc.avg([h["loss"] for h in history[-max(1, args.steps // 20) :]])
    final_val = val_history[-1]["val_loss"] if val_history else float("nan")
    metrics = {
        "stage": "sft",
        "config": vars(args),
        "init": str(lc.Path(args.init).relative_to(lc.ROOT)),
        "train_pairs": len(train_data),
        "val_pairs": len(val_data),
        "history": history,
        "val_history": val_history,
        "first_train_loss": round(first, 4),
        "last_train_loss": round(last, 4),
        "final_val_loss": round(final_val, 4),
        "val_answer_hit_rate": round(acc, 4),
        "train_answer_hit_rate": round(train_acc, 4),
        "val_samples": samples,
        "wall_seconds": round(wall, 1),
        "loss_curve": "results/sft_loss.png" if plot_ok else None,
    }
    (lc.ROOT / "results" / "sft_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    drop = 1 - last / first
    ok = drop > 0.2 and final_val < first
    print(
        f"[05] sft loss: {first:.4f} -> {last:.4f} (下降 {drop:.1%}), "
        f"final val loss {final_val:.4f}, 用时 {wall:.1f}s"
    )
    print(f"[05] loss 下降验证: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
