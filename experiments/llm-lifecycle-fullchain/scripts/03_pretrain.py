#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""03_pretrain.py — 预训练（Muon + AdamW 混合优化器，附 AdamW 基线对照）。

对齐 nanochat 的优化器策略（scripts/base_train.py + nanochat/optim.py）：
  - 隐藏矩阵（blocks 内全部参数）用 Muon：动量 + Newton-Schulz 正交化
  - 词嵌入 / lm_head 用 AdamW，lr 按组分别设置
  - 对照组：同样步数、同种子全部用 AdamW，用于量化 Muon 的收益

数据接入（对外部大语料的扩展）：
  - --train-source / --val-source 支持 parquet 分片文件、parquet 目录或 txt 文件
    （如 karpathy/climbmix-400b-shuffle 的 shard_XXXXX.parquet）
  - --max-chars / --doc-cap 控制训练字符上限与单文档截断，与 01 分词器口径一致
  - 不给 --val-source 时沿用原行为：从 $XL_DATA_DIR/corpus.txt 按 --val-frac 切分
  - --d-model/--n-layers/--n-heads/--d-ff/--block-size 自定义模型尺寸
  - --out-dir 独立实验输出目录（指标/checkpoint/曲线全部落此目录，不覆盖原 results/）
  - 统一测试集评估：除 val_loss 外记录 BPB（bits per byte，跨词表比较须对齐原文）与 PPL

设备与分布式：
  - --device auto/cpu/cuda，--amp none/bf16/fp16，--grad-accum 梯度累积，--scale 放大每卡 batch
  - torchrun 多进程时自动 DDP：各 rank 用 rank 偏移种子抽不同 batch，梯度 all_reduce 平均；
    评估、保存、画图只在 rank0 执行，checkpoint 保存时剥离 module. 前缀

产物（--out-dir 缺省为 $XL_OUT_DIR）：checkpoints/pretrain.pt（Muon，链路主线）、
      checkpoints/pretrain_adamw.pt（基线）、pretrain_metrics.json、pretrain_loss.png

用法：
    python scripts/03_pretrain.py [--steps 300] [--batch-size 32] [--lr-scale 1.0]
    python scripts/03_pretrain.py --device cuda --amp bf16 --scale 2
    torchrun --nproc_per_node=2 scripts/03_pretrain.py --device cpu --backend gloo
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
    ctx: lc.DistContext,
    token_bytes: torch.Tensor | None = None,
    ckpt_dir: Path | None = None,
) -> tuple[lc.CausalLM, dict]:
    """训练一个模型并返回 (model, metrics)。optimizer_kind: 'muon' | 'adamw'。

    单进程与 DDP 走同一段代码：net 在单进程时就是 model 本身；各 rank 的 batch 种子
    按 rank 偏移，因此抽到不同数据，DDP 反向时自动做梯度平均。
    """
    device = ctx.device
    amp = lc.AmpCtx(args.amp, device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    model = lc.build_model(cfg, seed=0).to(device)
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

    bs = lc.effective_batch(args.batch_size, args.scale)
    accum = max(1, args.grad_accum)
    gen = lc.batch_generator(args.seed, ctx)  # 各 rank 不同 batch
    val_gen = torch.Generator().manual_seed(args.seed + 1)  # 仅 rank0 评估用
    resume_file = (
        "pretrain_resume.pt" if optimizer_kind == "muon" else f"pretrain_{optimizer_kind}_resume.pt"
    )
    if ckpt_dir is not None:
        start_step, history, val_history, elapsed_prev = lc.resume_training_if_needed(
            args,
            ckpt_dir,
            resume_file,
            model,
            optimizers,
            device,
            ctx,
            scaler=amp.scaler,
            train_gen=gen,
            val_gen=val_gen,
            tag=f"03:{optimizer_kind}",
        )
    else:
        start_step, history, val_history, elapsed_prev = 0, [], [], 0.0
    net = lc.wrap_ddp(model, ctx)

    lc.reset_peak_memory(device)
    t0 = time.time()
    model.train()
    for step in range(start_step, args.steps):
        lc.check_available_memory()
        lc.set_lrs(optimizers, args.steps, step, args.lr_scale)
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        for micro in range(accum):
            x, y = lc.get_batch(train_ids, bs, args.block_size, gen, device=device)
            with amp.autocast():
                _, loss = net(x, targets=y)
            amp.backward(loss / accum)
        amp.step(optimizers, model=net, grad_clip=args.grad_clip)

        history.append({"step": step, "loss": round(loss.item(), 4)})
        if lc.is_rank0(ctx) and ((step + 1) % args.eval_every == 0 or step == 0):
            vl = lc.evaluate(model, val_ids, bs, args.block_size, 8, val_gen, device=device)
            val_history.append({"step": step, "val_loss": round(vl, 4)})
            print(
                f"[03:{optimizer_kind}] step {step + 1:>4}/{args.steps}  "
                f"train_loss={loss.item():.4f}  val_loss={vl:.4f}"
            )
        if ckpt_dir is not None:
            lc.checkpoint_step_hook(
                args,
                ckpt_dir,
                resume_file,
                model,
                cfg,
                optimizers,
                step + 1,
                ctx,
                scaler=amp.scaler,
                history=history,
                val_history=val_history,
                t0=t0,
                elapsed_offset=elapsed_prev,
                train_gen=gen,
                val_gen=val_gen,
                stage="pretrain",
                tag=f"03:{optimizer_kind}",
            )

    wall = elapsed_prev + (time.time() - t0)
    final_val = (
        lc.evaluate(model, val_ids, bs, args.block_size, 16, val_gen, device=device)
        if lc.is_rank0(ctx)
        else float("nan")
    )
    # 统一测试集 BPB / PPL（固定 token 流连续窗口；不同词表需额外对齐原文）
    bpb_ppl = (
        lc.evaluate_bpb_ppl(
            model, val_ids, token_bytes,
            batch_size=bs, block_size=args.block_size, device=device,
            max_windows=args.eval_windows,
        )
        if lc.is_rank0(ctx)
        else {}
    )
    model.eval()

    first = lc.avg([h["loss"] for h in history[: max(1, args.steps // 20)]])
    last = lc.avg([h["loss"] for h in history[-max(1, args.steps // 20) :]])
    best_entry = min(val_history, key=lambda v: v["val_loss"]) if val_history else {"val_loss": float("nan"), "step": -1}
    tokens_seen = args.steps * bs * accum * args.block_size * ctx.world_size
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
        "val_bpb_ppl": bpb_ppl,
        "wall_seconds": round(wall, 1),
        "tokens_seen": tokens_seen,
        "tokens_per_second": round(tokens_seen / max(wall, 1e-6), 1),
        "steps_per_second": round(args.steps / max(wall, 1e-6), 2),
        "batch_size_per_rank": bs,
        "grad_accum": accum,
        "train_data_cpu_mb": lc.tensor_size_mb(train_ids),
        "val_data_cpu_mb": lc.tensor_size_mb(val_ids),
        "memory": lc.memory_stats(device),
    }
    return model, metrics


def main() -> int:
    ap = argparse.ArgumentParser(description="预训练（外部大语料 + 数据量级扩展）")
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4, help="AdamW 基线学习率")
    ap.add_argument("--lr-scale", type=float, default=1.0, help="统一缩放各组 lr")
    ap.add_argument("--matrix-lr", type=float, default=0.02, help="Muon 矩阵 lr")
    ap.add_argument("--embedding-lr", type=float, default=0.3, help="词嵌入 lr")
    ap.add_argument("--unembedding-lr", type=float, default=0.004, help="lm_head lr")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--eval-windows", type=int, default=256, help="BPB/PPL 评估的连续窗口数")
    ap.add_argument("--val-frac", type=float, default=0.05, help="不给 --val-source 时的验证比例")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--no-baseline", action="store_true", help="跳过 AdamW 基线对照")
    # 外部语料
    ap.add_argument("--train-source", type=str, default=None, dest="train_source",
                    help="训练语料：parquet 分片文件 / parquet 目录 / txt（缺省 data/corpus.txt）")
    ap.add_argument("--val-source", type=str, default=None, dest="val_source",
                    help="固定留出验证集：parquet 分片 / 目录 / txt（给定则不再从训练集切分）")
    ap.add_argument("--max-chars", type=int, default=None, dest="max_chars",
                    help="训练语料最大字符上限（数据量级控制，缺省不限）")
    ap.add_argument("--val-max-chars", type=int, default=2_000_000, dest="val_max_chars",
                    help="验证集最大字符上限（默认 2M 字符）")
    ap.add_argument("--doc-cap", type=int, default=10_000, help="单篇文档截断字符数（与 01 一致）")
    lc.add_device_args(ap)
    lc.add_model_args(ap)
    lc.add_outdir_args(ap)
    lc.add_resume_args(ap)
    args = ap.parse_args()

    lc.check_available_memory()
    ctx = lc.init_distributed(args.device, args.backend)
    device = ctx.device
    out_dir = lc.resolve_out_dir(args)
    ckpt_dir = out_dir / "checkpoints"
    tok_path = lc.resolve_tokenizer_path(args)
    if lc.is_rank0(ctx):
        print(
            f"[03] device={device}, distributed={ctx.enabled}, world_size={ctx.world_size}, "
            f"backend={ctx.backend}, amp={args.amp}, grad_accum={args.grad_accum}, scale={args.scale}"
        )
        print(f"[03] out_dir={out_dir}, tokenizer={tok_path}")
    try:
        tok = lc.load_tokenizer(tok_path)
        vocab_size = tok.get_vocab_size()
        eos = lc.eos_id(tok)
        token_bytes = lc.load_token_bytes(tok_path.parent, tok=tok)

        cfg = lc.model_config_from_args(args, vocab_size)
        if args.val_source:
            # 外部固定留出验证集（各量级共用，保证可比）
            val_ids, val_stats = lc.build_ids_from_source(
                args.val_source, tok, args.val_max_chars, args.doc_cap, eos
            )
            train_ids, train_stats = lc.build_ids_from_source(
                args.train_source or (lc.data_dir() / "corpus.txt"),
                tok, args.max_chars, args.doc_cap, eos,
            )
            train_records, val_records = train_stats["docs"], val_stats["docs"]
        else:
            text = (lc.data_dir() / "corpus.txt").read_text(encoding="utf-8")
            train_records_l, val_records_l = lc.split_records(text, args.val_frac, seed=args.seed)
            train_ids = lc.records_to_ids(train_records_l, tok)
            val_ids = lc.records_to_ids(val_records_l, tok)
            train_records, val_records = len(train_records_l), len(val_records_l)
            train_stats = {"source": str(lc.data_dir() / "corpus.txt"), "docs": train_records,
                           "tokens": len(train_ids)}
            val_stats = {"source": "corpus.txt split", "docs": val_records, "tokens": len(val_ids)}
        train_ids = lc.ensure_stream(train_ids, args.block_size)
        val_ids = lc.ensure_stream(val_ids, args.block_size)
        if lc.is_rank0(ctx):
            print(
                f"[03] 训练语料: {train_stats['docs']:,} 条记录, {train_stats['tokens']:,} tokens "
                f"(源: {train_stats['source']}, max_chars={args.max_chars})"
            )
            print(
                f"[03] 验证语料: {val_stats['docs']:,} 条记录, {val_stats['tokens']:,} tokens "
                f"(源: {val_stats['source']})"
            )
            print(
                f"[03] 模型: {cfg.n_layers} 层, d_model={cfg.d_model}, heads={cfg.n_heads}, "
                f"d_ff={cfg.d_ff}, block={cfg.max_len}, vocab={vocab_size}, "
                f"mlp={cfg.mlp_type}, qk_norm={cfg.qk_norm}, softcap={cfg.softcap}"
            )

        # ---- 主线：Muon + AdamW ----
        model, muon_metrics = train_one(
            "muon", train_ids, val_ids, cfg, args, vocab_size, ctx, token_bytes, ckpt_dir=ckpt_dir
        )
        if lc.is_rank0(ctx):
            lc.save_checkpoint(
                model,
                cfg,
                ckpt_dir / "pretrain.pt",
                stage="pretrain",
                step=args.steps,
                tokenizer=str(tok_path),
                seed=args.seed,
                optimizer="muon",
                dist=lc.dist_info(ctx, args),
            )
            print(
                f"[03:muon] train loss: {muon_metrics['first_train_loss']:.4f} -> "
                f"{muon_metrics['last_train_loss']:.4f}, final val loss "
                f"{muon_metrics['final_val_loss']:.4f}, 用时 {muon_metrics['wall_seconds']}s"
            )

        # ---- 基线：纯 AdamW（同 steps 同种子）----
        adamw_metrics = None
        if not args.no_baseline:
            base_model, adamw_metrics = train_one(
                "adamw", train_ids, val_ids, cfg, args, vocab_size, ctx, token_bytes, ckpt_dir=ckpt_dir
            )
            if lc.is_rank0(ctx):
                lc.save_checkpoint(
                    base_model,
                    cfg,
                    ckpt_dir / "pretrain_adamw.pt",
                    stage="pretrain",
                    step=args.steps,
                    tokenizer=str(tok_path),
                    seed=args.seed,
                    optimizer="adamw",
                    dist=lc.dist_info(ctx, args),
                )
                print(
                    f"[03:adamw] train loss: {adamw_metrics['first_train_loss']:.4f} -> "
                    f"{adamw_metrics['last_train_loss']:.4f}, final val loss "
                    f"{adamw_metrics['final_val_loss']:.4f}, 用时 {adamw_metrics['wall_seconds']}s"
                )

        if not lc.is_rank0(ctx):
            return 0

        # ---- 曲线与指标（仅 rank0）----
        plot_ok = lc.plot_loss(
            muon_metrics["history"],
            muon_metrics["val_history"],
            out_dir / "pretrain_loss.png",
            title="Pretrain loss (Muon+AdamW)",
        )
        metrics = {
            "stage": "pretrain",
            "run_id": lc.run_id(),
            "config": vars(args) | {"model": cfg.to_dict(), "dist": lc.dist_info(ctx, args)},
            "vocab_size": vocab_size,
            "model_params": model.num_params(),
            "param_groups": model.num_param_groups(),
            "train_records": train_records,
            "val_records": val_records,
            "train_tokens": train_stats["tokens"],
            "val_tokens": val_stats["tokens"],
            "train_source": train_stats["source"],
            "val_source": val_stats["source"],
            "muon": muon_metrics,
            "adamw_baseline": adamw_metrics,
            "loss_curve": "pretrain_loss.png" if plot_ok else None,
        }
        if adamw_metrics:
            metrics["val_gain_muon_vs_adamw"] = round(
                adamw_metrics["final_val_loss"] - muon_metrics["final_val_loss"], 4
            )
        (out_dir / "pretrain_metrics.json").write_text(
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
            print("[03] 解读: best val 更低且出现更早 = 优化更快; final val 反映小语料上的过拟合程度")
        if muon_metrics.get("val_bpb_ppl"):
            bp = muon_metrics["val_bpb_ppl"]
            print(
                f"[03] 统一测试集: BPB={bp.get('bpb')}, PPL={bp.get('ppl')}, "
                f"mean_loss={bp.get('mean_loss_nats')} (eval_tokens={bp.get('eval_tokens')})"
            )
        return 0 if ok else 1
    finally:
        lc.cleanup_distributed()


if __name__ == "__main__":
    raise SystemExit(main())
