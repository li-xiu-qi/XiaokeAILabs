#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""05_sft.py — 指令微调（Supervised Fine-Tuning）。

在预训练（或 midtrain）权重基础上，用指令对做 SFT：
  - 序列拼成 "User: {问题}\nAssistant:" + " {回答}" + eos
  - loss 只在回答（含 eos）上计算，prompt 位置置 -100 忽略
  - 训练后在验证集 prompt 上贪心生成，按答案前缀命中率评估

数据接入（对外部大语料的扩展）：
  - --sft-path 支持任意 JSONL 指令集（含 HuggingFaceTB/smol-smoltalk 子集转换产物）
  - --test-source 给定统一测试集（parquet 分片/目录/txt）时，额外记录 BPB / PPL
  - --out-dir 独立实验输出目录；--tokenizer 自定义分词器

设备与分布式：--device / --amp / --backend / --grad-accum / --scale 同 03；torchrun 多进程时
各 rank 用 rank 偏移的 rng 抽不同指令对，命中率评估与保存只在 rank0。

产物（--out-dir 缺省为 $XL_OUT_DIR）：checkpoints/sft.pt、sft_metrics.json、sft_loss.png。

用法：
    python scripts/05_sft.py [--steps 250] [--batch-size 16] [--lr-scale 0.2]
    python scripts/05_sft.py --device cuda --amp bf16 --scale 2
    torchrun --nproc_per_node=2 scripts/05_sft.py --device cpu
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import llm_common as lc


@torch.no_grad()
def hit_rate(
    model: lc.CausalLM, tok, pairs: list[dict], eos: int, max_new: int = 24, device=None
) -> tuple[float, list[dict]]:
    """贪心生成并按答案前缀命中率统计。"""
    hits, samples = 0, []
    for row in pairs:
        p_ids = tok.encode(row["prompt"], add_special_tokens=False).ids
        idx = torch.tensor([p_ids], dtype=torch.long, device=device)
        out, _ = lc.generate(
            model, idx, max_new, eos, temperature=0.0, use_cache=True
        )
        gen = tok.decode(out[0, len(p_ids) :].tolist(), skip_special_tokens=True).strip()
        expect = row["response"].strip()
        hit = gen.replace(" ", "").startswith(expect.replace(" ", "")[:12])
        hits += int(hit)
        samples.append(
            {"prompt": row["prompt"], "expected": expect, "generated": gen, "hit": bool(hit)}
        )
    return hits / max(1, len(pairs)), samples


def load_pairs(path: str | Path) -> list[dict]:
    """加载指令对：JSONL 或 Parquet（HuggingFaceTB/smol-smoltalk 的 messages 列）。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"未检测到 SFT 数据文件: {p}\n请指定 --sft-path 或先准备指令微调 JSONL。"
        )
    if p.suffix.lower() == ".parquet":
        return pairs_from_smoltalk(p)
    return [
        json.loads(l)
        for l in p.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]


def pairs_from_smoltalk(path: str | Path, limit: int | None = None) -> list[dict]:
    """把 smol-smoltalk 风格 Parquet（messages 列）转成 prompt/response 指令对。

    取每条对话最后一个 user 回合作 prompt、最后一个 assistant 回合作 response；
    多轮上下文拼进 prompt，保持与 JSONL 指令集同一 schema。
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(str(path))
    out: list[dict] = []
    for rg in range(pf.num_row_groups):
        table = pf.read_row_group(rg, columns=["messages"])
        for row in table.column("messages").to_pylist():
            msgs = row if isinstance(row, list) else []
            user_turns = [m for m in msgs if m.get("role") == "user"]
            asst_turns = [m for m in msgs if m.get("role") == "assistant"]
            if not user_turns or not asst_turns:
                continue
            prompt_parts, response = [], None
            for m in msgs:
                if m.get("role") == "user":
                    prompt_parts.append(str(m.get("content", "")).strip())
                elif m.get("role") == "assistant":
                    response = str(m.get("content", "")).strip()
            if not response:
                continue
            out.append({"prompt": "\n".join(x for x in prompt_parts if x), "response": response})
            if limit and len(out) >= limit:
                return out
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="指令微调 SFT（外部指令集 + 统一测试集评估）")
    ap.add_argument("--init", default=None,
                    help="初始 checkpoint（缺省 $XL_OUT_DIR/checkpoints/midtrain.pt）")
    ap.add_argument("--sft-path", type=str, default=None, dest="sft_path",
                    help="SFT 指令集 JSONL 路径（缺省 $XL_DATA_DIR/sft.jsonl）")
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--batch-size", type=int, default=16)
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
    ap.add_argument("--test-source", type=str, default=None, dest="test_source",
                    help="统一测试集（parquet/txt），给定则记录 BPB / PPL")
    ap.add_argument("--test-max-chars", type=int, default=2_000_000, dest="test_max_chars")
    ap.add_argument("--eval-windows", type=int, default=256)
    ap.add_argument("--sft-max-rows", type=int, default=20_000, dest="sft_max_rows",
                    help="从指令集（含 Parquet）中最多加载的条数，控制大语料下的加载耗时")
    ap.add_argument("--hit-eval-rows", type=int, default=200, dest="hit_eval_rows",
                    help="命中率评估最多贪心生成多少条（控制耗时，缺省 200）")
    ap.add_argument("--val-eval-rows", type=int, default=256, dest="val_eval_rows",
                    help="val_loss 分批评估最多条数（防止全量 eval OOM，缺省 256）")
    lc.add_device_args(ap)
    lc.add_model_args(ap)
    lc.add_outdir_args(ap)
    lc.add_resume_args(ap)
    args = ap.parse_args()

    lc.check_available_memory()
    ctx = lc.init_distributed(args.device, args.backend)
    device = ctx.device
    amp = lc.AmpCtx(args.amp, device)
    out_dir = lc.resolve_out_dir(args)
    if not args.init:
        args.init = str(out_dir / "checkpoints" / "midtrain.pt")
    ckpt_dir = out_dir / "checkpoints"
    tok_path = lc.resolve_tokenizer_path(args)
    sft_path = Path(args.sft_path) if args.sft_path else (lc.data_dir() / "sft.jsonl")
    if lc.is_rank0(ctx):
        print(
            f"[05] device={device}, distributed={ctx.enabled}, world_size={ctx.world_size}, "
            f"backend={ctx.backend}, amp={args.amp}"
        )
        print(f"[05] out_dir={out_dir}, tokenizer={tok_path}, sft_path={sft_path}")
    try:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

        tok = lc.load_tokenizer(tok_path)
        eos = lc.eos_id(tok)
        pad_id = tok.token_to_id(lc.PAD)
        token_bytes = lc.load_token_bytes(tok_path.parent, tok=tok)
        model, ckpt = lc.load_checkpoint(lc.ROOT / args.init, device=device)
        cfg = model.cfg
        if lc.is_rank0(ctx):
            print(f"[05] 载入 {args.init}（stage={ckpt.get('stage')}）, {model.num_params():,} 参数")

        pairs = load_pairs(sft_path)
        if args.sft_max_rows and len(pairs) > args.sft_max_rows:
            rng0 = random.Random(args.seed)
            rng0.shuffle(pairs)
            pairs = pairs[: args.sft_max_rows]
            if lc.is_rank0(ctx):
                print(f"[05] 指令集截断到 {args.sft_max_rows:,} 条（--sft-max-rows）")
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
        if lc.is_rank0(ctx):
            print(f"[05] SFT 数据: {len(train_data)} train / {len(val_data)} val 条指令对")

        def make_batch(data):
            seqs, labels = [], []
            for p, r in data:
                ids = (p + r)[: cfg.max_len]
                lab = ([-100] * len(p) + list(r))[: cfg.max_len]
                seqs.append(ids)
                labels.append(lab)
            maxlen = max(len(s) for s in seqs)
            x = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
            y = torch.full((len(seqs), maxlen), -100, dtype=torch.long)
            for i, (s, l) in enumerate(zip(seqs, labels)):
                x[i, : len(s)] = torch.tensor(s)
                y[i, : len(l)] = torch.tensor(l)
            return x.to(device), y.to(device)

        net = lc.wrap_ddp(model, ctx)

        def masked_loss(x, y, m):
            logits, _ = m(x)
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

        bs = lc.effective_batch(args.batch_size, args.scale)
        accum = max(1, args.grad_accum)
        # 各 rank 用不同 rng：DDP 下两个 rank 抽同一批指令对只会浪费一半算力
        sample_rng = random.Random(lc.rank_seed(args.seed, ctx))
        # 分批评估 val_loss（按 token 数加权），避免全量 val_data 一次前向分配数十 GiB logits 导致 OOM
        val_eval_subset = val_data[: max(1, min(len(val_data), args.val_eval_rows))]

        def eval_val_loss() -> float:
            total_loss = 0.0
            total_toks = 0
            step_bs = max(1, bs)
            with torch.no_grad():
                for idx in range(0, len(val_eval_subset), step_bs):
                    chunk = val_eval_subset[idx : idx + step_bs]
                    vx, vy = make_batch(chunk)
                    n_tok = int((vy[:, 1:] != -100).sum().item())
                    if n_tok == 0:
                        continue
                    with amp.autocast():
                        l_val = masked_loss(vx, vy, model).item()
                    total_loss += l_val * n_tok
                    total_toks += n_tok
            return total_loss / max(1, total_toks)

        start_step, history, val_history, elapsed_prev = lc.resume_training_if_needed(
            args,
            ckpt_dir,
            "sft_resume.pt",
            model,
            optimizers,
            device,
            ctx,
            scaler=amp.scaler,
            sample_rng=sample_rng,
            tag="05",
        )
        lc.reset_peak_memory(device)
        t0 = time.time()
        model.train()
        for step in range(start_step, args.steps):
            lc.check_available_memory()
            lc.set_lrs(optimizers, args.steps, step, args.lr_scale)
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)
            for micro in range(accum):
                batch = sample_rng.sample(train_data, k=min(bs, len(train_data)))
                x, y = make_batch(batch)
                with amp.autocast():
                    loss = masked_loss(x, y, net)
                amp.backward(loss / accum)
            amp.step(optimizers, model=net, grad_clip=args.grad_clip)
            history.append({"step": step, "loss": round(loss.item(), 4)})
            if lc.is_rank0(ctx) and ((step + 1) % args.eval_every == 0 or step == 0):
                model.eval()
                vl = eval_val_loss()
                model.train()
                val_history.append({"step": step, "val_loss": round(vl, 4)})
                print(
                    f"[05] step {step + 1:>4}/{args.steps}  sft_loss={loss.item():.4f}  "
                    f"val_loss={vl:.4f}"
                )
            lc.checkpoint_step_hook(
                args,
                ckpt_dir,
                "sft_resume.pt",
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
                sample_rng=sample_rng,
                stage="sft",
                tag="05",
            )
        wall = elapsed_prev + (time.time() - t0)
        model.eval()

        if not lc.is_rank0(ctx):
            return 0

        acc, samples = hit_rate(model, tok, val_pairs[: args.hit_eval_rows], eos, device=device)
        train_acc, _ = hit_rate(model, tok, train_pairs[: min(20, args.hit_eval_rows)], eos, device=device)
        print(f"[05] 训练集命中率: {train_acc:.0%}, 留出集命中率: {acc:.0%}")
        for s in samples[:5]:
            print(
                f"[05] 样例 Q: {s['prompt'].splitlines()[0].replace('User: ', '')}  ->  "
                f"{s['generated'][:40]!r} (期望 {s['expected'][:30]!r}, "
                f"{'✓' if s['hit'] else '✗'})"
            )

        # 统一测试集 BPB / PPL（给定 --test-source 时）
        test_bpb_ppl = {}
        if args.test_source:
            test_ids, test_stats = lc.build_ids_from_source(
                args.test_source, tok, args.test_max_chars, 10_000, eos
            )
            test_bpb_ppl = lc.evaluate_bpb_ppl(
                model, lc.ensure_stream(test_ids, cfg.max_len), token_bytes,
                batch_size=bs, block_size=cfg.max_len, device=device,
                max_windows=args.eval_windows,
            )
            print(
                f"[05] 统一测试集: BPB={test_bpb_ppl.get('bpb')}, "
                f"PPL={test_bpb_ppl.get('ppl')} (源: {args.test_source})"
            )

        lc.save_checkpoint(
            model,
            cfg,
            ckpt_dir / "sft.pt",
            stage="sft",
            init=str(lc.Path(args.init).relative_to(lc.ROOT)) if str(args.init).startswith(str(lc.ROOT)) else args.init,
            step=args.steps,
            tokenizer=str(tok_path),
            seed=args.seed,
            dist=lc.dist_info(ctx, args),
        )

        plot_ok = lc.plot_loss(
            history,
            val_history,
            out_dir / "sft_loss.png",
            title="SFT loss on instruction pairs",
            ylabel="cross entropy (response only)",
        )
        first = lc.avg([h["loss"] for h in history[: max(1, args.steps // 20)]])
        last = lc.avg([h["loss"] for h in history[-max(1, args.steps // 20) :]])
        final_val = val_history[-1]["val_loss"] if val_history else float("nan")
        metrics = {
            "stage": "sft",
            "run_id": lc.run_id(),
            "config": vars(args) | {"dist": lc.dist_info(ctx, args)},
            "init": args.init,
            "sft_path": str(sft_path),
            "train_pairs": len(train_data),
            "val_pairs": len(val_data),
            "history": history,
            "val_history": val_history,
            "first_train_loss": round(first, 4),
            "last_train_loss": round(last, 4),
            "final_val_loss": round(final_val, 4),
            "val_answer_hit_rate": round(acc, 4),
            "train_answer_hit_rate": round(train_acc, 4),
            "test_bpb_ppl": test_bpb_ppl,
            "val_samples": samples,
            "wall_seconds": round(wall, 1),
            "batch_size_per_rank": bs,
            "grad_accum": accum,
            "memory": lc.memory_stats(device),
            "loss_curve": "sft_loss.png" if plot_ok else None,
        }
        (out_dir / "sft_metrics.json").write_text(
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
    finally:
        lc.cleanup_distributed()


if __name__ == "__main__":
    raise SystemExit(main())
