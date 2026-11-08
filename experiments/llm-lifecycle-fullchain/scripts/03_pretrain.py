#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""03_pretrain.py — 微型语料预训练（Muon + AdamW 混合优化器，附 AdamW 基线对照）。

对齐 nanochat 的优化器策略（scripts/base_train.py + nanochat/optim.py）：
  - 隐藏矩阵（blocks 内全部参数）用 Muon：动量 + Newton-Schulz 正交化
  - 词嵌入 / lm_head 用 AdamW，lr 按组分别设置
  - 对照组：同样步数、同种子全部用 AdamW，用于量化 Muon 的收益

产物：results/checkpoints/pretrain.pt（Muon，链路主线）、
      results/checkpoints/pretrain_adamw.pt（基线）、
      results/pretrain_metrics.json、results/pretrain_loss.png

用法：
    python scripts/03_pretrain.py [--steps 300] [--batch-size 32] [--lr-scale 1.0]
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

import llm_common as lc


def train_one(
    optimizer_kind: str,
    train_ids: torch.Tensor,
    val_ids: torch.Tensor,
    cfg: lc.ModelConfig,
    args,
    vocab_size: int,
) -> tuple[lc.CausalLM, dict]:
    """训练一个模型并返回 (model, metrics)。optimizer_kind: 'muon' | 'adamw'。"""
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    model = lc.build_model(cfg, seed=0)
    optimizers, groups_info = lc.build_optimizer(
        model,
        cfg,
        lr=args.lr,
        weight_decay=args.weight_decay,
        optimizer=optimizer_kind,
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
                f"[03:{optimizer_kind}] step {step + 1:>4}/{args.steps}  "
                f"train_loss={loss.item():.4f}  val_loss={vl:.4f}"
            )

    wall = time.time() - t0
    final_val = lc.evaluate(model, val_ids, args.batch_size, args.block_size, 16, val_gen)
    model.eval()

    first = lc.avg([h["loss"] for h in history[: max(1, args.steps // 20)]])
    last = lc.avg([h["loss"] for h in history[-max(1, args.steps // 20) :]])
    best_entry = min(val_history, key=lambda v: v["val_loss"])
    metrics = {
        "optimizer": optimizer_kind,
        "lr_groups": groups_info,
        "history": history,
        "val_history": val_history,
        "first_train_loss": round(first, 4),
        "last_train_loss": round(last, 4),
        "best_val_loss": round(best_entry["val_loss"], 4),
        "best_val_step": best_entry["step"],
        "final_val_loss": round(final_val, 4),
        "wall_seconds": round(wall, 1),
        "steps_per_second": round(args.steps / max(wall, 1e-6), 2),
    }
    return model, metrics


def main() -> int:
    ap = argparse.ArgumentParser(description="微型语料预训练")
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4, help="AdamW 基线学习率")
    ap.add_argument("--lr-scale", type=float, default=1.0, help="统一缩放各组 lr")
    ap.add_argument("--matrix-lr", type=float, default=0.02, help="Muon 矩阵 lr")
    ap.add_argument("--embedding-lr", type=float, default=0.3, help="词嵌入 lr")
    ap.add_argument("--unembedding-lr", type=float, default=0.004, help="lm_head lr")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--no-baseline", action="store_true", help="跳过 AdamW 基线对照")
    args = ap.parse_args()

    tok = lc.load_tokenizer()
    vocab_size = tok.get_vocab_size()
    eos = lc.eos_id(tok)

    cfg = lc.ModelConfig(vocab_size=vocab_size, max_len=args.block_size)
    text = (lc.DATA_DIR / "corpus.txt").read_text(encoding="utf-8")
    train_records, val_records = lc.split_records(text, args.val_frac, seed=args.seed)
    train_ids = lc.ensure_stream(lc.records_to_ids(train_records, tok), args.block_size)
    val_ids = lc.ensure_stream(lc.records_to_ids(val_records, tok), args.block_size)
    print(
        f"[03] 语料: {len(train_records)} train / {len(val_records)} val 条记录, "
        f"{len(train_ids):,} / {len(val_ids):,} tokens, vocab={vocab_size}"
    )
    print(
        f"[03] 模型: {cfg.n_layers} 层, d_model={cfg.d_model}, heads={cfg.n_heads}, "
        f"mlp={cfg.mlp_type}, qk_norm={cfg.qk_norm}, softcap={cfg.softcap}"
    )

    # ---- 主线：Muon + AdamW ----
    model, muon_metrics = train_one("muon", train_ids, val_ids, cfg, args, vocab_size)
    lc.save_checkpoint(
        model,
        cfg,
        lc.CKPT_DIR / "pretrain.pt",
        stage="pretrain",
        step=args.steps,
        tokenizer=str(lc.TOK_PATH.relative_to(lc.ROOT)),
        seed=args.seed,
        optimizer="muon",
    )
    print(
        f"[03:muon] train loss: {muon_metrics['first_train_loss']:.4f} -> "
        f"{muon_metrics['last_train_loss']:.4f}, final val loss "
        f"{muon_metrics['final_val_loss']:.4f}, 用时 {muon_metrics['wall_seconds']}s"
    )

    # ---- 基线：纯 AdamW（同 steps 同种子）----
    adamw_metrics = None
    if not args.no_baseline:
        base_model, adamw_metrics = train_one("adamw", train_ids, val_ids, cfg, args, vocab_size)
        lc.save_checkpoint(
            base_model,
            cfg,
            lc.CKPT_DIR / "pretrain_adamw.pt",
            stage="pretrain",
            step=args.steps,
            tokenizer=str(lc.TOK_PATH.relative_to(lc.ROOT)),
            seed=args.seed,
            optimizer="adamw",
        )
        print(
            f"[03:adamw] train loss: {adamw_metrics['first_train_loss']:.4f} -> "
            f"{adamw_metrics['last_train_loss']:.4f}, final val loss "
            f"{adamw_metrics['final_val_loss']:.4f}, 用时 {adamw_metrics['wall_seconds']}s"
        )

    # ---- 曲线与指标 ----
    plot_ok = lc.plot_loss(
        muon_metrics["history"],
        muon_metrics["val_history"],
        lc.ROOT / "results" / "pretrain_loss.png",
        title="Pretrain loss on toy corpus (Muon+AdamW)",
    )
    metrics = {
        "stage": "pretrain",
        "config": vars(args) | {"model": cfg.to_dict()},
        "vocab_size": vocab_size,
        "model_params": model.num_params(),
        "param_groups": model.num_param_groups(),
        "train_records": len(train_records),
        "val_records": len(val_records),
        "train_tokens": len(train_ids),
        "val_tokens": len(val_ids),
        "muon": muon_metrics,
        "adamw_baseline": adamw_metrics,
        "loss_curve": "results/pretrain_loss.png" if plot_ok else None,
    }
    if adamw_metrics:
        metrics["val_gain_muon_vs_adamw"] = round(
            adamw_metrics["final_val_loss"] - muon_metrics["final_val_loss"], 4
        )
    (lc.ROOT / "results" / "pretrain_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    drop = 1 - muon_metrics["last_train_loss"] / muon_metrics["first_train_loss"]
    ok = drop > 0.3
    print(f"[03] Muon 下降 {drop:.1%}（阈值 30%）: {'PASS' if ok else 'FAIL'}")
    if adamw_metrics:
        mu, ad = muon_metrics, adamw_metrics
        print(
            f"[03] Muon   best val {mu['best_val_loss']:.4f}@step{mu['best_val_step']} "
            f"/ final {mu['final_val_loss']:.4f}"
        )
        print(
            f"[03] AdamW  best val {ad['best_val_loss']:.4f}@step{ad['best_val_step']} "
            f"/ final {ad['final_val_loss']:.4f}"
        )
        print(f"[03] 解读: best val 更低且出现更早 = 优化更快; final val 反映小语料上的过拟合程度")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
