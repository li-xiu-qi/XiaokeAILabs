#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""04_midtrain.py — Midtraining：对话格式适配（nanochat base_train 与 chat_sft 之间的阶段）。

在预训练权重基础上，只在「User:/Assistant:」对话记录上以较低学习率继续训练：
  - 全序列 loss（不做 prompt 掩码），让模型先学会对话格式与问答分布
  - lr 统一缩放（--lr-scale，默认 0.2），避免破坏预训练知识

设备与分布式：--device / --amp / --backend / --grad-accum / --scale 同 03；torchrun 多进程时
各 rank 抽不同 batch，评估、样例、保存、画图只在 rank0，checkpoint 无 module. 前缀。

产物：$XL_OUT_DIR/checkpoints/midtrain.pt、$XL_OUT_DIR/midtrain_metrics.json。

用法：
    python scripts/04_midtrain.py [--steps 120] [--batch-size 16] [--lr-scale 0.2]
    python scripts/04_midtrain.py --device cuda --amp bf16
    torchrun --nproc_per_node=2 scripts/04_midtrain.py --device cpu
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time

import torch

import llm_common as lc


def dialogue_records(text: str) -> list[str]:
    """语料中的对话格式记录（含 User:/Assistant: 的记录）。"""
    return [r for r in text.split(lc.EOS) if "User:" in r and "Assistant:" in r]


@torch.no_grad()
def quick_samples(
    model: lc.CausalLM, tok, prompts: list[str], eos: int, max_new: int = 16, device=None
) -> list[dict]:
    rows = []
    for p in prompts:
        ids = tok.encode(p, add_special_tokens=False).ids
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        out, _ = lc.generate(
            model, idx, max_new, eos, temperature=0.0, use_cache=True
        )
        text = tok.decode(out[0, len(ids) :].tolist(), skip_special_tokens=True).strip()
        rows.append({"prompt": p.replace("\n", "\\n"), "generated": text})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="对话格式 midtrain（外部语料 + 独立输出目录）")
    ap.add_argument("--init", default=None,
                    help="初始 checkpoint（缺省 $XL_OUT_DIR/checkpoints/pretrain.pt）")
    ap.add_argument("--dialogue-source", type=str, default=None, dest="dialogue_source",
                    help="对话记录语料：parquet/txt（缺省 $XL_DATA_DIR/corpus.txt，按 User:/Assistant: 过滤）")
    ap.add_argument("--max-chars", type=int, default=None, dest="max_chars",
                    help="对话语料最大字符上限（缺省不限）")
    ap.add_argument("--test-source", type=str, default=None, dest="test_source",
                    help="统一测试集（parquet/txt），给定则记录 BPB / PPL")
    ap.add_argument("--test-max-chars", type=int, default=2_000_000, dest="test_max_chars")
    ap.add_argument("--eval-windows", type=int, default=256)
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
    lc.add_device_args(ap)
    lc.add_outdir_args(ap)
    lc.add_resume_args(ap)
    args = ap.parse_args()

    lc.check_available_memory()
    ctx = lc.init_distributed(args.device, args.backend)
    device = ctx.device
    amp = lc.AmpCtx(args.amp, device)
    out_dir = lc.resolve_out_dir(args)
    if not args.init:
        args.init = str(out_dir / "checkpoints" / "pretrain.pt")
    ckpt_dir = out_dir / "checkpoints"
    tok_path = lc.resolve_tokenizer_path(args)
    if lc.is_rank0(ctx):
        print(
            f"[04] device={device}, distributed={ctx.enabled}, world_size={ctx.world_size}, "
            f"backend={ctx.backend}, amp={args.amp}"
        )
        print(f"[04] out_dir={out_dir}, tokenizer={tok_path}")
    try:
        torch.manual_seed(args.seed)
        random.seed(lc.rank_seed(args.seed, ctx))

        tok = lc.load_tokenizer(tok_path)
        eos = lc.eos_id(tok)
        token_bytes = lc.load_token_bytes(tok_path.parent, tok=tok)
        model, ckpt = lc.load_checkpoint(lc.ROOT / args.init, device=device)
        cfg = model.cfg
        if lc.is_rank0(ctx):
            print(
                f"[04] 载入 {args.init}（stage={ckpt.get('stage')}, step={ckpt.get('step')}）, "
                f"{model.num_params():,} 参数"
            )

        if args.dialogue_source:
            # 外部对话语料：逐文档过滤 User:/Assistant: 记录
            import dataset as _ds

            records = []
            total_chars = 0
            for doc in _ds.iter_texts(data_source=args.dialogue_source, split="train",
                                      max_chars=args.max_chars, doc_cap=100_000):
                for rec in doc.split(lc.EOS):
                    if "User:" in rec and "Assistant:" in rec:
                        records.append(rec)
                        total_chars += len(rec)
            if not records:
                print(f"[04] 错误: {args.dialogue_source} 中未找到对话记录", file=sys.stderr)
                return 1
        else:
            text = (lc.data_dir() / "corpus.txt").read_text(encoding="utf-8")
            records = dialogue_records(text)
        rng = random.Random(args.seed)
        rng.shuffle(records)
        n_val = max(1, int(len(records) * args.val_frac))
        val_records, train_records = records[:n_val], records[n_val:]
        train_ids = lc.ensure_stream(lc.records_to_ids(train_records, tok), cfg.max_len)
        val_ids = lc.ensure_stream(lc.records_to_ids(val_records, tok), cfg.max_len)
        if lc.is_rank0(ctx):
            print(
                f"[04] 对话记录: {len(train_records)} train / {len(val_records)} val, "
                f"{len(train_ids):,} / {len(val_ids):,} tokens"
            )

        net = lc.wrap_ddp(model, ctx)
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

        bs = lc.effective_batch(args.batch_size, args.scale)
        accum = max(1, args.grad_accum)
        gen = lc.batch_generator(args.seed, ctx)
        val_gen = torch.Generator().manual_seed(args.seed + 1)
        start_step, history, val_history, elapsed_prev = lc.resume_training_if_needed(
            args,
            ckpt_dir,
            "midtrain_resume.pt",
            model,
            optimizers,
            device,
            ctx,
            scaler=amp.scaler,
            train_gen=gen,
            val_gen=val_gen,
            tag="04",
        )
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
                    f"[04] step {step + 1:>4}/{args.steps}  train_loss={loss.item():.4f}  "
                    f"val_loss={vl:.4f}"
                )
            lc.checkpoint_step_hook(
                args,
                ckpt_dir,
                "midtrain_resume.pt",
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
                stage="midtrain",
                tag="04",
            )
        wall = elapsed_prev + (time.time() - t0)
        final_val = (
            lc.evaluate(model, val_ids, bs, args.block_size, 16, val_gen, device=device)
            if lc.is_rank0(ctx)
            else float("nan")
        )
        model.eval()

        if not lc.is_rank0(ctx):
            return 0

        lc.save_checkpoint(
            model,
            cfg,
            ckpt_dir / "midtrain.pt",
            stage="midtrain",
            init=str(lc.Path(args.init).relative_to(lc.ROOT)) if str(args.init).startswith(str(lc.ROOT)) else args.init,
            step=args.steps,
            tokenizer=str(tok_path),
            seed=args.seed,
            dist=lc.dist_info(ctx, args),
        )

        samples = quick_samples(
            model,
            tok,
            [
                "User: What is the capital of Japan?\nAssistant:",
                "User: Hello\nAssistant:",
            ],
            eos,
            device=device,
        )
        for s in samples:
            print(f"[04] 样例 {s['prompt']} -> {s['generated'][:40]!r}")

        plot_ok = lc.plot_loss(
            history,
            val_history,
            out_dir / "midtrain_loss.png",
            title="Midtrain loss on dialogue records",
        )
        # 统一测试集 BPB / PPL（给定 --test-source 时）
        test_bpb_ppl = {}
        if args.test_source:
            test_ids, _ = lc.build_ids_from_source(args.test_source, tok, args.test_max_chars, 10_000, eos)
            test_bpb_ppl = lc.evaluate_bpb_ppl(
                model, lc.ensure_stream(test_ids, cfg.max_len), token_bytes,
                batch_size=bs, block_size=cfg.max_len, device=device,
                max_windows=args.eval_windows,
            )
            print(f"[04] 统一测试集: BPB={test_bpb_ppl.get('bpb')}, PPL={test_bpb_ppl.get('ppl')}")
        first = lc.avg([h["loss"] for h in history[: max(1, args.steps // 10)]])
        last = lc.avg([h["loss"] for h in history[-max(1, args.steps // 10) :]])
        metrics = {
            "stage": "midtrain",
            "run_id": lc.run_id(),
            "config": vars(args) | {"dist": lc.dist_info(ctx, args)},
            "init": args.init,
            "train_records": len(train_records),
            "val_records": len(val_records),
            "history": history,
            "val_history": val_history,
            "first_train_loss": round(first, 4),
            "last_train_loss": round(last, 4),
            "final_val_loss": round(final_val, 4),
            "test_bpb_ppl": test_bpb_ppl,
            "samples": samples,
            "wall_seconds": round(wall, 1),
            "loss_curve": "midtrain_loss.png" if plot_ok else None,
        }
        (out_dir / "midtrain_metrics.json").write_text(
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
    finally:
        lc.cleanup_distributed()


if __name__ == "__main__":
    raise SystemExit(main())
