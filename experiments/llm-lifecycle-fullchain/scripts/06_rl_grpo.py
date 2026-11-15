#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""06_rl_grpo.py — GRPO 强化学习（可验证奖励）。

对齐 nanochat scripts/chat_rl.py 的思路，落到本实验的玩具规模：
  - 策略模型：05 产出的 sft.pt；参考模型：同权重的冻结副本（KL 锚点）
  - 每个 prompt 采样一组（group）回答，reward 为可验证的精确匹配
  - advantage = 组内 (r - mean) / std；组内无差异（std≈0）时跳过，不产生梯度
  - loss = -advantage * logprob(回答token) + beta * KL(pi || ref)
  - reward 整形：精确匹配 1.0；首字符为数字 0.2（否则冷启动时全 0 奖励无梯度）
RL 任务为 $XL_DATA_DIR/rl.jsonl 的个位数加减法：in_corpus=true 的算式进过预训练语料
（训练集），false 的完全留出（验证集），用于观察 RL 的记忆与泛化边界。

设备与分布式（--device / --amp / --backend / --scale / --grad-accum）：
  - 采样阶段（no_grad）对 unwrap_model 剥离 DDP 后的策略跑 generate，不触发梯度同步
  - 策略前向反向逐 group 走 DDP 包装（一次 forward 配一次 backward，避免 DDP
    "mark a variable ready only once"），梯度自动跨 rank 平均
  - 多卡时 prompt 按 rank 交错分片（并行采样），组内无差异被跳过的 group 不产生梯度；
    某 rank 本 step 无有效 group 时用零梯度 backward 参与 all_reduce，防止挂死
  - 评估、保存、画图只在 rank0；mean_reward 等指标跨 rank 平均

产物：$XL_OUT_DIR/checkpoints/rl.pt、$XL_OUT_DIR/rl_metrics.json、$XL_OUT_DIR/rl_reward.png。

用法：
    python scripts/06_rl_grpo.py [--steps 30] [--group-size 8] [--prompts-per-step 6]
    python scripts/06_rl_grpo.py --device cuda --scale 2
    torchrun --nproc_per_node=2 scripts/06_rl_grpo.py --device cpu
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

import llm_common as lc

DIGITS = set("0123456789")


def load_rl_tasks(path: str | Path | None = None) -> tuple[list[dict], list[dict]]:
    p = Path(path) if path else (lc.data_dir() / "rl.jsonl")
    if not p.exists():
        raise FileNotFoundError(f"未检测到 RL 任务文件: {p}，请指定 --rl-path")
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    train = [r for r in rows if r.get("in_corpus")]
    val = [r for r in rows if not r.get("in_corpus")]
    return train, val


def reward_of(generated: str, answer: str) -> float:
    """可验证奖励：精确匹配 1.0；答案后仅跟标点 0.6；首字符为数字 0.2；其余 0。

    后两档是冷启动整形：全 0 奖励时组内无差异、GRPO 无梯度。
    """
    g = generated.strip()
    if g == answer:
        return 1.0
    if g.startswith(answer):
        rest = g[len(answer) :].strip()
        if rest and all(ch in ".,!?;:" for ch in rest):
            return 0.6
    if g and g[0] in DIGITS:
        return 0.2
    return 0.0


@torch.no_grad()
def exact_match(
    model: lc.CausalLM, tok, tasks: list[dict], eos: int, max_new: int = 8, device=None
) -> float:
    hits = 0
    for row in tasks:
        p_ids = tok.encode(row["prompt"], add_special_tokens=False).ids
        idx = torch.tensor([p_ids], dtype=torch.long, device=device)
        out, _ = lc.generate(
            model, idx, max_new, eos, temperature=0.0, use_cache=True
        )
        gen = tok.decode(out[0, len(p_ids) :].tolist(), skip_special_tokens=True).strip()
        hits += int(gen == row["answer"])
    return hits / max(1, len(tasks))


@torch.no_grad()
def chat_hit_rate(model: lc.CausalLM, tok, pairs: list[dict], eos: int, device=None) -> float:
    """贪心生成 + 答案前缀命中率（与 05 同口径，用于 RL 后的回归检查）。"""
    hits = 0
    for row in pairs:
        p_ids = tok.encode(row["prompt"], add_special_tokens=False).ids
        idx = torch.tensor([p_ids], dtype=torch.long, device=device)
        out, _ = lc.generate(model, idx, 24, eos, temperature=0.0, use_cache=True)
        gen = tok.decode(out[0, len(p_ids) :].tolist(), skip_special_tokens=True).strip()
        expect = row["response"].strip()
        hits += int(gen.replace(" ", "").startswith(expect.replace(" ", "")[:12]))
    return hits / max(1, len(pairs))


def sample_group(
    model: lc.CausalLM,
    tok,
    prompt: str,
    eos: int,
    group_size: int,
    max_new: int,
    temperature: float,
    seed: int,
    device=None,
):
    """对一个 prompt 采样 group_size 个回答，返回 (prompt_ids, [gen_ids...], [reward...])。

    采样在 no_grad 下对 unwrap 后的策略执行：DDP 包装只用于训练前向反向。
    """
    p_ids = tok.encode(prompt, add_special_tokens=False).ids
    idx = torch.tensor([p_ids] * group_size, dtype=torch.long, device=device)
    torch.manual_seed(seed)
    out, _ = lc.generate(
        model,
        idx,
        max_new,
        eos,
        temperature=temperature,
        top_k=0,
        top_p=1.0,
        use_cache=False,
        seed=seed,
    )
    gens, rewards = [], []
    for i in range(group_size):
        gen_ids = out[i, len(p_ids) :].tolist()
        text = tok.decode(gen_ids, skip_special_tokens=True)
        gens.append(gen_ids)
        rewards.append(reward_of(text, row_answer(prompt)))
    return p_ids, gens, rewards


_ANSWER_CACHE: dict[str, str] = {}
_RL_ROWS: list[dict] = []  # 当前 RL 任务全集（row_answer 查表用）


def row_answer(prompt: str) -> str:
    """从 RL 任务文件查 prompt 对应的标准答案（建缓存避免重复读文件）。"""
    if not _ANSWER_CACHE:
        for r in _RL_ROWS:
            _ANSWER_CACHE[r["prompt"]] = r["answer"]
    return _ANSWER_CACHE[prompt]


def valid_token_counts(gens: list[list[int]], p_len: int, eos: int) -> list[int]:
    """每个回答的有效 token 数（从 p_len 到首个 eos 含 eos），纯序列计算不依赖模型。"""
    counts = []
    for g in gens:
        end = p_len + len(g)
        for j, t in enumerate(g):
            if t == eos:
                end = p_len + j + 1
                break
        counts.append(max(end - p_len, 0))
    return counts


def group_logprob_and_kl(
    model: lc.CausalLM,
    ref: lc.CausalLM,
    p_ids: list[int],
    gens: list[list[int]],
    pad_id: int,
    eos: int,
):
    """对一个 prompt 的一组回答做 teacher forcing，返回 (sum_logprob, kl, n_valid)。

    - sum_logprob[i]：第 i 个回答全部有效 token 的 logprob 之和（eos 计入，其后截断）
    - kl[i]：第 i 个回答有效 token 的平均 KL(pi || ref)
    """
    group = len(gens)
    seqs = [list(p_ids) + g for g in gens]
    maxlen = max(len(s) for s in seqs)
    x = torch.full((group, maxlen), pad_id, dtype=torch.long, device=next(model.parameters()).device)
    for i, s in enumerate(seqs):
        x[i, : len(s)] = torch.tensor(s)

    logits, _ = model(x)  # (G, L, V) —— DDP 下这是策略前向（每次调用配一次 backward）
    with torch.no_grad():
        ref_logits, _ = ref(x)
    logp = F.log_softmax(logits.float(), dim=-1)
    logp_ref = F.log_softmax(ref_logits.float(), dim=-1)

    p_len = len(p_ids)
    sum_lp = torch.zeros(group, device=x.device)
    kl_sum = torch.zeros(group, device=x.device)
    n_valid = torch.zeros(group, device=x.device)
    for i, s in enumerate(seqs):
        # 有效回答位置：p_len .. len(s)-1；遇到 eos 后停止
        end = len(s)
        for j in range(p_len, len(s)):
            if s[j] == eos:
                end = j + 1
                break
        for j in range(p_len, end):
            tok_id = s[j]
            sum_lp[i] += logp[i, j - 1, tok_id]
            p = logp[i, j - 1].exp()
            kl_sum[i] += (p * (logp[i, j - 1] - logp_ref[i, j - 1])).sum()
            n_valid[i] += 1
    return sum_lp, kl_sum, n_valid


def main() -> int:
    ap = argparse.ArgumentParser(description="GRPO 可验证奖励强化学习")
    ap.add_argument("--init", default=None,
                    help="初始 checkpoint（缺省 $XL_OUT_DIR/checkpoints/sft.pt）")
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--group-size", type=int, default=10)
    ap.add_argument("--prompts-per-step", type=int, default=6)
    ap.add_argument("--max-new", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.2)
    ap.add_argument("--kl-beta", type=float, default=0.1)
    ap.add_argument("--lr-scale", type=float, default=1.0)
    ap.add_argument("--matrix-lr", type=float, default=0.002)
    ap.add_argument("--embedding-lr", type=float, default=0.05)
    ap.add_argument("--unembedding-lr", type=float, default=0.002)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--rl-path", type=str, default=None, dest="rl_path",
                    help="RL 任务 JSONL 路径（缺省 $XL_DATA_DIR/rl.jsonl）")
    ap.add_argument("--sft-path", type=str, default=None, dest="sft_path",
                    help="SFT 回归检查 JSONL 路径（缺省 $XL_DATA_DIR/sft.jsonl）")
    ap.add_argument("--test-source", type=str, default=None, dest="test_source",
                    help="统一测试集（parquet/txt），给定则记录 BPB / PPL")
    ap.add_argument("--test-max-chars", type=int, default=2_000_000, dest="test_max_chars")
    ap.add_argument("--eval-windows", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1234)
    lc.add_device_args(
        ap, scale_help="规模系数：按比例放大每步 prompt 数（默认 1.0 保持 CPU 冒烟口径）"
    )
    lc.add_outdir_args(ap)
    lc.add_resume_args(ap)
    args = ap.parse_args()

    lc.check_available_memory()
    ctx = lc.init_distributed(args.device, args.backend)
    device = ctx.device
    amp = lc.AmpCtx(args.amp, device)
    out_dir = lc.resolve_out_dir(args)
    if not args.init:
        args.init = str(out_dir / "checkpoints" / "sft.pt")
    ckpt_dir = out_dir / "checkpoints"
    tok_path = lc.resolve_tokenizer_path(args)
    rl_path = Path(args.rl_path) if args.rl_path else (lc.data_dir() / "rl.jsonl")
    if lc.is_rank0(ctx):
        print(
            f"[06] device={device}, distributed={ctx.enabled}, world_size={ctx.world_size}, "
            f"backend={ctx.backend}, amp={args.amp}, scale={args.scale}"
        )
        print(f"[06] out_dir={out_dir}, tokenizer={tok_path}, rl_path={rl_path}")
    try:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

        tok = lc.load_tokenizer(tok_path)
        lc_eos_id = lc.eos_id(tok)
        pad_id = tok.token_to_id(lc.PAD)
        token_bytes = lc.load_token_bytes(tok_path.parent, tok=tok)
        model, ckpt = lc.load_checkpoint(lc.ROOT / args.init, device=device)
        cfg = model.cfg
        ref = copy.deepcopy(model)
        for p in ref.parameters():
            p.requires_grad_(False)
        ref.eval()
        if lc.is_rank0(ctx):
            print(
                f"[06] 载入 {args.init}（stage={ckpt.get('stage')}）, {model.num_params():,} 参数, "
                f"参考模型为同权重冻结副本"
            )

        train_tasks, val_tasks = load_rl_tasks(rl_path)
        _RL_ROWS.clear()
        _RL_ROWS.extend(train_tasks + val_tasks)
        # 多卡并行采样：prompt 按 rank 交错分片，各 rank 采不同的 prompt 组
        my_tasks = lc.shard_list(train_tasks, ctx)
        prompts_per_step = max(1, int(round(args.prompts_per_step * (args.scale or 1.0))))
        if lc.is_rank0(ctx):
            print(
                f"[06] RL 任务: {len(train_tasks)} 训练（算式进过语料）/ "
                f"{len(val_tasks)} 留出（算式未进语料）; 本 rank 分片 {len(my_tasks)} 条, "
                f"每步 {prompts_per_step} prompt"
            )

        net = lc.wrap_ddp(model, ctx)
        policy = lc.unwrap_model(net)  # 采样与评估都用剥离 DDP 的策略
        optimizers, _ = lc.build_optimizer(
            model,
            cfg,
            lr=1e-3,
            weight_decay=0.0,
            optimizer="muon",
            matrix_lr=args.matrix_lr,
            embedding_lr=args.embedding_lr,
            unembedding_lr=args.unembedding_lr,
        )
        lc.tag_initial_lrs(optimizers)

        rng = random.Random(lc.rank_seed(args.seed, ctx))
        start_step, history, eval_history, elapsed_prev = lc.resume_training_if_needed(
            args,
            ckpt_dir,
            "rl_resume.pt",
            model,
            optimizers,
            device,
            ctx,
            scaler=amp.scaler,
            sample_rng=rng,
            tag="06",
        )
        t0 = time.time()
        model.train()
        for step in range(start_step, args.steps):
            lc.check_available_memory()
            lc.set_lrs(optimizers, args.steps, step, args.lr_scale)
            pool = my_tasks if my_tasks else train_tasks
            prompts = rng.sample(pool, k=min(prompts_per_step, len(pool)))
            step_rewards, skipped = [], 0
            groups = []  # (adv, p_ids, gens)
            n_tok_total = 0
            for row in prompts:
                p_ids, gens, rewards = sample_group(
                    policy,
                    tok,
                    row["prompt"],
                    lc_eos_id,
                    args.group_size,
                    args.max_new,
                    args.temperature,
                    args.seed + step * 1000 + len(step_rewards),
                    device=device,
                )
                step_rewards.extend(rewards)
                r = torch.tensor(rewards, dtype=torch.float32, device=device)
                std = r.std(unbiased=False)
                if std < 1e-6:  # 组内无差异 → advantage 全 0，跳过
                    skipped += 1
                    continue
                adv = (r - r.mean()) / (std + 1e-4)
                n_valid = valid_token_counts(gens, len(p_ids), lc_eos_id)
                n_tok_total += sum(n_valid)
                groups.append((adv, p_ids, gens))

            # 所有 rank 对齐本 step 是否做参数更新：只要任一 rank 有有效 group（m_groups > 0）
            # 即协同更新；有效 group 较少的 rank 用零梯度 dummy backward 补齐次数，避免死锁
            n_groups = len(groups)
            m_groups = n_groups
            if ctx.enabled:
                # 分母取全局有效 token 数；DDP 反向会自动除以 world_size，因此下面需按
                # world_size 抵消平均系数，保证有效梯度与单进程「总 loss / 总 token」一致
                n_tok_total = int(lc.all_reduce_sum(n_tok_total, ctx))
                m_t = torch.tensor([n_groups], device=device)
                dist.all_reduce(m_t, op=dist.ReduceOp.MAX)
                m_groups = int(m_t.item())
            if m_groups > 0:
                denom = max(n_tok_total, 1)
                grad_scale = ctx.world_size if ctx.enabled else 1
                for opt in optimizers:
                    opt.zero_grad(set_to_none=True)
                for adv, p_ids, gens in groups:
                    # 逐 group forward + backward：与单进程「求和后一次 backward」
                    # 数学等价（DDP 对每次 backward 的梯度取跨 rank 平均），且避开
                    # 多前向一次反向触发 DDP 的 ready-once 错误
                    sum_lp, kl_sum, _ = group_logprob_and_kl(
                        net, ref, p_ids, gens, pad_id, lc_eos_id
                    )
                    loss_g = (-(adv * sum_lp).sum() + args.kl_beta * kl_sum.sum()) * (grad_scale / denom)
                    amp.backward(loss_g)
                for _ in range(m_groups - n_groups):
                    # 补齐 backward 次数：经 DDP 包装的零梯度前向反向，不改变参数
                    dummy = torch.zeros((1, 2), dtype=torch.long, device=device)
                    logits0, _ = net(dummy)
                    amp.backward(logits0.sum() * 0.0)
                amp.step(optimizers, model=net, grad_clip=1.0)

            mean_r = lc.all_reduce_mean(
                sum(step_rewards) / max(len(step_rewards), 1), ctx
            )
            skipped_all = int(lc.all_reduce_sum(skipped, ctx)) if ctx.enabled else skipped
            history.append(
                {
                    "step": step,
                    "mean_reward": round(mean_r, 4),
                    "exact_reward_frac": round(
                        sum(1 for r in step_rewards if r >= 1.0) / max(len(step_rewards), 1), 4
                    ),
                    "skipped_groups": skipped_all,
                }
            )
            if lc.is_rank0(ctx) and ((step + 1) % args.eval_every == 0 or step == 0):
                model.eval()
                tr_acc = exact_match(model, tok, train_tasks[:40], lc_eos_id, args.max_new, device=device)
                va_acc = exact_match(model, tok, val_tasks, lc_eos_id, args.max_new, device=device)
                model.train()
                eval_history.append(
                    {
                        "step": step,
                        "train_exact_match": round(tr_acc, 4),
                        "val_exact_match": round(va_acc, 4),
                    }
                )
                print(
                    f"[06] step {step + 1:>3}/{args.steps}  mean_reward={mean_r:.3f}  "
                    f"train_exact={tr_acc:.0%}  val_exact={va_acc:.0%}  "
                    f"skipped={skipped_all}  groups={len(groups)}"
                )
            lc.checkpoint_step_hook(
                args,
                ckpt_dir,
                "rl_resume.pt",
                model,
                cfg,
                optimizers,
                step + 1,
                ctx,
                scaler=amp.scaler,
                history=history,
                val_history=eval_history,
                t0=t0,
                elapsed_offset=elapsed_prev,
                sample_rng=rng,
                stage="rl_grpo",
                tag="06",
            )

        wall = elapsed_prev + (time.time() - t0)
        model.eval()

        if not lc.is_rank0(ctx):
            return 0

        # ---- 最终评估：算术 + 聊天问答回归（仅 rank0）----
        train_acc = exact_match(model, tok, train_tasks, lc_eos_id, args.max_new, device=device)
        val_acc = exact_match(model, tok, val_tasks, lc_eos_id, args.max_new, device=device)
        base_train = exact_match(ref, tok, train_tasks, lc_eos_id, args.max_new, device=device)
        base_val = exact_match(ref, tok, val_tasks, lc_eos_id, args.max_new, device=device)

        sft_file = Path(args.sft_path) if getattr(args, "sft_path", None) else (lc.data_dir() / "sft.jsonl")
        sft_pairs = [
            json.loads(l)
            for l in sft_file.read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]
        rng2 = random.Random(args.seed)
        rng2.shuffle(sft_pairs)
        chat_val = sft_pairs[:10]
        chat_acc = chat_hit_rate(model, tok, chat_val, lc_eos_id, device=device)
        ref_chat_acc = chat_hit_rate(ref, tok, chat_val, lc_eos_id, device=device)

        print(
            f"[06] 算术 exact-match: 训练集 {base_train:.0%} -> {train_acc:.0%}, "
            f"留出集 {base_val:.0%} -> {val_acc:.0%}"
        )
        print(f"[06] 聊天问答命中率（回归检查）: {ref_chat_acc:.0%} -> {chat_acc:.0%}")

        # 统一测试集 BPB / PPL（给定 --test-source 时）
        test_ids = None
        if args.test_source:
            test_ids, test_stats = lc.build_ids_from_source(
                args.test_source, tok, args.test_max_chars, 10_000, lc_eos_id
            )
            bp = lc.evaluate_bpb_ppl(
                model, lc.ensure_stream(test_ids, cfg.max_len), token_bytes,
                batch_size=8, block_size=cfg.max_len, device=device,
                max_windows=args.eval_windows,
            )
            print(f"[06] 统一测试集: BPB={bp.get('bpb')}, PPL={bp.get('ppl')}")

        lc.save_checkpoint(
            model,
            cfg,
            ckpt_dir / "rl.pt",
            stage="rl_grpo",
            init=str(lc.Path(args.init).relative_to(lc.ROOT)) if str(args.init).startswith(str(lc.ROOT)) else args.init,
            step=args.steps,
            tokenizer=str(tok_path),
            seed=args.seed,
            dist=lc.dist_info(ctx, args),
        )

        # ---- reward 曲线 ----
        plot_ok = False
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
            steps = [h["step"] for h in history]
            ax.plot(
                steps,
                [h["mean_reward"] for h in history],
                lw=1.6,
                color="tab:green",
                label="mean reward (shaped)",
            )
            ax.plot(
                steps,
                [h["exact_reward_frac"] for h in history],
                lw=1.6,
                color="tab:blue",
                label="exact-match fraction",
            )
            if eval_history:
                ax.plot(
                    [e["step"] for e in eval_history],
                    [e["train_exact_match"] for e in eval_history],
                    "o-",
                    ms=3,
                    color="tab:orange",
                    label="train exact-match (greedy)",
                )
                ax.plot(
                    [e["step"] for e in eval_history],
                    [e["val_exact_match"] for e in eval_history],
                    "s--",
                    ms=3,
                    color="tab:red",
                    label="val exact-match (greedy)",
                )
            ax.set_xlabel("step")
            ax.set_ylabel("reward / accuracy")
            ax.set_title("GRPO on verifiable arithmetic")
            ax.set_ylim(-0.05, 1.05)
            ax.legend()
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(out_dir / "rl_reward.png")
            plt.close(fig)
            plot_ok = True
        except Exception:
            pass

        metrics = {
            "stage": "rl_grpo",
            "run_id": lc.run_id(),
            "config": vars(args) | {"dist": lc.dist_info(ctx, args)},
            "init": args.init,
            "n_train_tasks": len(train_tasks),
            "n_val_tasks": len(val_tasks),
            "history": history,
            "eval_history": eval_history,
            "final": {
                "train_exact_match": round(train_acc, 4),
                "val_exact_match": round(val_acc, 4),
                "sft_train_exact_match": round(base_train, 4),
                "sft_val_exact_match": round(base_val, 4),
                "chat_hit_rate_after_rl": round(chat_acc, 4),
                "chat_hit_rate_sft": round(ref_chat_acc, 4),
            },
            "test_bpb_ppl": (
                lc.evaluate_bpb_ppl(
                    model, lc.ensure_stream(test_ids, cfg.max_len), token_bytes,
                    batch_size=8, block_size=cfg.max_len, device=device,
                    max_windows=args.eval_windows,
                )
                if args.test_source
                else {}
            ),
            "wall_seconds": round(wall, 1),
            "reward_curve": "rl_reward.png" if plot_ok else None,
        }
        (out_dir / "rl_metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[06] 用时 {wall:.1f}s，产物见 {out_dir / 'rl_metrics.json'}")
        return 0
    finally:
        lc.cleanup_distributed()


if __name__ == "__main__":
    raise SystemExit(main())
