#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""06_rl_grpo.py — GRPO 强化学习（可验证奖励）。

对齐 nanochat scripts/chat_rl.py 的思路，落到本实验的玩具规模：
  - 策略模型：05 产出的 sft.pt；参考模型：同权重的冻结副本（KL 锚点）
  - 每个 prompt 采样一组（group）回答，reward 为可验证的精确匹配
  - advantage = 组内 (r - mean) / std；组内无差异（std≈0）时跳过，不产生梯度
  - loss = -advantage * logprob(回答token) + beta * KL(pi || ref)
  - reward 整形：精确匹配 1.0；首字符为数字 0.2（否则冷启动时全 0 奖励无梯度）
RL 任务为 data/rl.jsonl 的个位数加减法：in_corpus=true 的算式进过预训练语料
（训练集），false 的完全留出（验证集），用于观察 RL 的记忆与泛化边界。

产物：results/checkpoints/rl.pt、results/rl_metrics.json、results/rl_reward.png。

用法：
    python scripts/06_rl_grpo.py [--steps 30] [--group-size 8] [--prompts-per-step 6]
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time

import torch
import torch.nn.functional as F

import llm_common as lc

DIGITS = set("0123456789")


def load_rl_tasks() -> tuple[list[dict], list[dict]]:
    rows = [
        json.loads(l)
        for l in (lc.DATA_DIR / "rl.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
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
def exact_match(model: lc.CausalLM, tok, tasks: list[dict], eos: int, max_new: int = 8) -> float:
    hits = 0
    for row in tasks:
        p_ids = tok.encode(row["prompt"], add_special_tokens=False).ids
        out, _ = lc.generate(
            model, torch.tensor([p_ids]), max_new, eos, temperature=0.0, use_cache=True
        )
        gen = tok.decode(out[0, len(p_ids) :].tolist(), skip_special_tokens=True).strip()
        hits += int(gen == row["answer"])
    return hits / max(1, len(tasks))


@torch.no_grad()
def chat_hit_rate(model: lc.CausalLM, tok, pairs: list[dict], eos: int) -> float:
    """贪心生成 + 答案前缀命中率（与 05 同口径，用于 RL 后的回归检查）。"""
    hits = 0
    for row in pairs:
        p_ids = tok.encode(row["prompt"], add_special_tokens=False).ids
        out, _ = lc.generate(model, torch.tensor([p_ids]), 24, eos, temperature=0.0, use_cache=True)
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
):
    """对一个 prompt 采样 group_size 个回答，返回 (prompt_ids, [gen_ids...], [reward...])。"""
    p_ids = tok.encode(prompt, add_special_tokens=False).ids
    idx = torch.tensor([p_ids] * group_size, dtype=torch.long)
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


def row_answer(prompt: str) -> str:
    """从 rl.jsonl 查 prompt 对应的标准答案（建缓存避免重复读文件）。"""
    if not _ANSWER_CACHE:
        for l in (lc.DATA_DIR / "rl.jsonl").read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                _ANSWER_CACHE[r["prompt"]] = r["answer"]
    return _ANSWER_CACHE[prompt]


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
    x = torch.full((group, maxlen), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, : len(s)] = torch.tensor(s)

    logits, _ = model(x)  # (G, L, V)
    with torch.no_grad():
        ref_logits, _ = ref(x)
    logp = F.log_softmax(logits.float(), dim=-1)
    logp_ref = F.log_softmax(ref_logits.float(), dim=-1)

    p_len = len(p_ids)
    sum_lp = torch.zeros(group)
    kl_sum = torch.zeros(group)
    n_valid = torch.zeros(group)
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
    ap.add_argument("--init", default=str(lc.CKPT_DIR / "sft.pt"))
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
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    tok = lc.load_tokenizer()
    lc_eos_id = lc.eos_id(tok)
    pad_id = tok.token_to_id(lc.PAD)
    model, ckpt = lc.load_checkpoint(lc.ROOT / args.init)
    cfg = model.cfg
    ref = copy.deepcopy(model)
    for p in ref.parameters():
        p.requires_grad_(False)
    ref.eval()
    print(
        f"[06] 载入 {args.init}（stage={ckpt.get('stage')}）, {model.num_params():,} 参数, "
        f"参考模型为同权重冻结副本"
    )

    train_tasks, val_tasks = load_rl_tasks()
    print(
        f"[06] RL 任务: {len(train_tasks)} 训练（算式进过语料）/ "
        f"{len(val_tasks)} 留出（算式未进语料）"
    )

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

    rng = random.Random(args.seed)
    history, eval_history = [], []
    t0 = time.time()
    model.train()
    for step in range(args.steps):
        lc.set_lrs(optimizers, args.steps, step, args.lr_scale)
        prompts = rng.sample(train_tasks, k=min(args.prompts_per_step, len(train_tasks)))
        pg_terms, kl_terms, n_tok = [], [], 0
        step_rewards, skipped = [], 0
        for row in prompts:
            p_ids, gens, rewards = sample_group(
                model,
                tok,
                row["prompt"],
                lc_eos_id,
                args.group_size,
                args.max_new,
                args.temperature,
                args.seed + step * 1000 + len(step_rewards),
            )
            step_rewards.extend(rewards)
            r = torch.tensor(rewards, dtype=torch.float32)
            std = r.std(unbiased=False)
            if std < 1e-6:  # 组内无差异 → advantage 全 0，跳过
                skipped += 1
                continue
            adv = (r - r.mean()) / (std + 1e-4)
            sum_lp, kl_sum, n_valid = group_logprob_and_kl(
                model, ref, p_ids, gens, pad_id, lc_eos_id
            )
            pg_terms.append(-(adv * sum_lp).sum())
            kl_terms.append(kl_sum.sum())
            n_tok += int(n_valid.sum())

        if pg_terms:
            loss = (sum(pg_terms) + args.kl_beta * sum(kl_terms)) / max(n_tok, 1)
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for opt in optimizers:
                opt.step()
        else:
            loss = None

        mean_r = sum(step_rewards) / max(len(step_rewards), 1)
        history.append(
            {
                "step": step,
                "mean_reward": round(mean_r, 4),
                "exact_reward_frac": round(
                    sum(1 for r in step_rewards if r >= 1.0) / max(len(step_rewards), 1), 4
                ),
                "skipped_groups": skipped,
                "loss": round(loss.item(), 4) if loss is not None else None,
            }
        )
        if (step + 1) % args.eval_every == 0 or step == 0:
            model.eval()
            tr_acc = exact_match(model, tok, train_tasks[:40], lc_eos_id, args.max_new)
            va_acc = exact_match(model, tok, val_tasks, lc_eos_id, args.max_new)
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
                f"skipped={skipped}  loss={loss.item() if loss is not None else float('nan'):.4f}"
            )

    wall = time.time() - t0
    model.eval()

    # ---- 最终评估：算术 + 聊天问答回归 ----
    train_acc = exact_match(model, tok, train_tasks, lc_eos_id, args.max_new)
    val_acc = exact_match(model, tok, val_tasks, lc_eos_id, args.max_new)
    base_train = exact_match(ref, tok, train_tasks, lc_eos_id, args.max_new)
    base_val = exact_match(ref, tok, val_tasks, lc_eos_id, args.max_new)

    sft_pairs = [
        json.loads(l)
        for l in (lc.DATA_DIR / "sft.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    rng2 = random.Random(args.seed)
    rng2.shuffle(sft_pairs)
    chat_val = sft_pairs[:10]
    chat_acc = chat_hit_rate(model, tok, chat_val, lc_eos_id)
    ref_chat_acc = chat_hit_rate(ref, tok, chat_val, lc_eos_id)

    print(
        f"[06] 算术 exact-match: 训练集 {base_train:.0%} -> {train_acc:.0%}, "
        f"留出集 {base_val:.0%} -> {val_acc:.0%}"
    )
    print(f"[06] 聊天问答命中率（回归检查）: {ref_chat_acc:.0%} -> {chat_acc:.0%}")

    lc.save_checkpoint(
        model,
        cfg,
        lc.CKPT_DIR / "rl.pt",
        stage="rl_grpo",
        init=str(lc.Path(args.init).relative_to(lc.ROOT)),
        step=args.steps,
        tokenizer=str(lc.TOK_PATH.relative_to(lc.ROOT)),
        seed=args.seed,
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
        fig.savefig(lc.ROOT / "results" / "rl_reward.png")
        plt.close(fig)
        plot_ok = True
    except Exception:
        pass

    metrics = {
        "stage": "rl_grpo",
        "config": vars(args),
        "init": str(lc.Path(args.init).relative_to(lc.ROOT)),
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
        "wall_seconds": round(wall, 1),
        "reward_curve": "results/rl_reward.png" if plot_ok else None,
    }
    (lc.ROOT / "results" / "rl_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[06] 用时 {wall:.1f}s，产物见 results/rl_metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
