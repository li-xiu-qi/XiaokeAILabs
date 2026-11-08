#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""01_train_tokenizer.py — 语料生成 + BPE 分词器训练/导出/校验。

链路第 1 步：
  1. 确定性生成微型样本语料（data/corpus.txt + data/sft.jsonl + data/rl.jsonl）
  2. 在其上训练 Byte-Level BPE，预分词采用 GPT-4 风格正则（对齐 nanochat 的
     tiktoken 路线：先按词/数字/标点切分，再逐块做 byte-level BPE）
  3. 导出 results/tokenizer/tokenizer.json，做压缩率与 encode→decode roundtrip 校验

语料除首都/水果/职业/句法/身份问答外，新增：
  - 英文数词算术事实（"Two plus three is five."）
  - 阿拉伯数字算式语句（"3 + 4 = 7."，共 70 组，供 RL 阶段做可验证奖励）
  - 对话格式记录（User:/Assistant:，供 midtrain 阶段做格式适配）
data/rl.jsonl 为 100 组个位数加减法可验证任务（70 训练 / 30 留出，按算式组划分）。

用法：
    python scripts/01_train_tokenizer.py [--vocab-size 2048] [--seed 1234] [--force]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import llm_common as lc

# GPT-4 / tiktoken 风格预分词正则（nanochat tokenizer 同款思路）
GPT4_SPLIT_PATTERN = (
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    "|[^\r\n\\p{L}\\p{N}]?\\p{L}+"
    "|\\p{N}{1,3}"
    "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*"
    "|\\s*[\\r\\n]+"
    "|\\s+(?!\\S)"
    "|\\s+"
)

# ---------------------------------------------------------------- 样本语料 ----

CAPITALS = [
    ("France", "Paris"),
    ("Japan", "Tokyo"),
    ("China", "Beijing"),
    ("Egypt", "Cairo"),
    ("Brazil", "Brasilia"),
    ("Canada", "Ottawa"),
    ("Kenya", "Nairobi"),
    ("Peru", "Lima"),
    ("India", "New Delhi"),
    ("Norway", "Oslo"),
]
FRUITS = [
    ("apple", "red"),
    ("banana", "yellow"),
    ("grape", "purple"),
    ("orange", "orange"),
    ("lemon", "yellow"),
    ("berry", "blue"),
]
PEOPLE = [
    ("Alice", "doctor"),
    ("Bob", "teacher"),
    ("Carol", "engineer"),
    ("David", "artist"),
    ("Erin", "nurse"),
    ("Frank", "chef"),
    ("Grace", "pilot"),
    ("Henry", "farmer"),
]
ADJS = ["small", "big", "red", "blue", "quiet", "loud"]
NOUNS = ["boy", "girl", "cat", "dog", "bird", "robot", "student", "farmer"]
OBJS = ["bread", "water", "music", "books", "letters", "songs", "pictures", "stories"]
VERBS_BASE = ["like", "eat", "see", "find", "make", "take", "give", "keep", "read", "draw"]
VERBS_3RD = ["likes", "eats", "sees", "finds", "makes", "takes", "gives", "keeps", "reads", "draws"]
NUMBER_WORDS = [
    "Two plus three is five.",
    "Seven minus two is five.",
    "Three times four is twelve.",
    "Nine divided by three is three.",
    "The number after seven is eight.",
    "The number before ten is nine.",
    "Ten plus ten is twenty.",
    "Six times five is thirty.",
    "One plus one is two.",
    "Eight minus four is four.",
]
IDENTITY = [
    ("Who are you?", "I am XiaoLM, a small language model."),
    ("What is your name?", "My name is XiaoLM."),
    ("What can you do?", "I can answer simple questions about facts."),
    ("Are you a big model?", "No, I am a tiny model trained on a small corpus."),
]
CHAT = [
    ("Hello", "Hi there!"),
    ("How are you?", "I am fine, thank you."),
    ("Thank you", "You are welcome."),
    ("Goodbye", "Goodbye, see you later."),
]

# 个位数加减法：70 组进预训练语料（RL 训练集），30 组留出（RL 验证集）
ARITH_TRAIN_FRAC = 0.7


def build_arith_pairs() -> list[tuple[int, int, str, int]]:
    """返回 (a, b, op, result) 列表：加法 a+b<=18，减法 a>=b。"""
    pairs = []
    for a in range(0, 10):
        for b in range(0, 10):
            if a + b <= 18:
                pairs.append((a, b, "plus", a + b))
            if a >= b:
                pairs.append((a, b, "minus", a - b))
    return pairs


def build_corpus(seed: int = 1234) -> tuple[str, list[dict], list[dict]]:
    """生成样本语料。返回 (语料全文, SFT 指令对, RL 可验证任务)。"""
    rng = random.Random(seed)
    qa_pairs: list[tuple[str, str]] = []
    blocks: list[str] = []

    # 1) 首都事实 + 对应问答
    for country, capital in CAPITALS:
        blocks.append(
            f"The capital of {country} is {capital}.\n"
            f"{capital} is the capital of {country}.\n"
            f"Fact: the capital of {country} is {capital}.{lc.EOS}"
        )
        qa_pairs += [
            (f"What is the capital of {country}?", capital),
            (f"Which city is the capital of {country}?", capital),
            (
                f"Is {capital} the capital of {country}?",
                f"Yes, {capital} is the capital of {country}.",
            ),
        ]

    # 2) 水果颜色
    for fruit, color in FRUITS:
        blocks.append(f"An {fruit} is {color}.\nThe {fruit} is {color}.{lc.EOS}")
        qa_pairs.append((f"What color is an {fruit}?", f"The {fruit} is {color}."))

    # 3) 人物职业
    for name, job in PEOPLE:
        blocks.append(f"{name} is a {job}.\n{name} works as a {job}.{lc.EOS}")
        qa_pairs += [
            (f"What does {name} do?", f"{name} is a {job}."),
            (f"What is {name}'s job?", f"{name} is a {job}."),
            (f"Is {name} a {job}?", f"Yes, {name} is a {job}."),
        ]

    # 4) 句法模板组合（控制规模，固定子集）
    grammar = [
        f"The {adj} {noun} {v3} the {obj}."
        for adj in ADJS
        for noun in NOUNS
        for v3 in VERBS_3RD
        for obj in OBJS
    ]
    rng.shuffle(grammar)
    for i in range(0, 600, 8):
        blocks.append("\n".join(grammar[i : i + 8]) + lc.EOS)
    grammar2 = []
    for vb in VERBS_BASE:
        for obj in OBJS:
            grammar2 += [
                f"I {vb} {obj} every day.",
                f"She does not {vb} {obj}.",
                f"Do you {vb} {obj}?",
                f"They {vb} {obj} in the morning.",
                f"The student does not {vb} {obj}.",
            ]
    rng.shuffle(grammar2)
    for i in range(0, 300, 10):
        blocks.append("\n".join(grammar2[i : i + 10]) + lc.EOS)

    # 5) 数词算术事实（自然语言，无数字形式）
    for i in range(0, len(NUMBER_WORDS), 4):
        blocks.append("\n".join(NUMBER_WORDS[i : i + 4]) + lc.EOS)

    # 6) 阿拉伯数字算式语句（70 组）：预训练记住「数字→结果」映射，供 RL 放大
    arith = build_arith_pairs()
    rng.shuffle(arith)
    n_arith_train = int(len(arith) * ARITH_TRAIN_FRAC)
    arith_train, arith_val = arith[:n_arith_train], arith[n_arith_train:]
    for i in range(0, len(arith_train), 7):
        blocks.append(
            "\n".join(f"{a} {op} {b} = {r}." for a, b, op, r in arith_train[i : i + 7]) + lc.EOS
        )

    # 7) 身份/闲聊对话 + 全部事实问答对话（midtrain 学格式用）
    for q, a in IDENTITY + CHAT:
        blocks.append(f"User: {q}\nAssistant: {a}{lc.EOS}")
        qa_pairs.append((q, a))
    rng.shuffle(qa_pairs)
    for i in range(0, len(qa_pairs), 5):
        blocks.append(
            "\n".join(f"User: {q}\nAssistant: {a}" for q, a in qa_pairs[i : i + 5]) + lc.EOS
        )

    rng.shuffle(blocks)
    corpus_text = "\n".join(blocks) + "\n"

    sft = [{"prompt": f"User: {q}\nAssistant:", "response": f" {a}"} for q, a in qa_pairs]
    # 混入少量算术指令对：让 SFT 模型具备算术问答的起点能力（RL 从这里继续提升）。
    # nanochat 的 SFT 数据（SmolTalk）同样含多样内容，RL 阶段在其上做强化。
    arith_sft = rng.sample(arith_train, k=15)
    sft += [
        {"prompt": f"User: What is {a} {op} {b}?\nAssistant:", "response": f" {r}"}
        for a, b, op, r in arith_sft
    ]
    rl_tasks = [
        {
            "prompt": f"User: What is {a} {op} {b}?\nAssistant:",
            "answer": str(r),
            "in_corpus": (a, b, op) in {(x, y, o) for x, y, o, _ in arith_train},
        }
        for a, b, op, r in arith
    ]
    return corpus_text, sft, rl_tasks


# ------------------------------------------------------------------- 主流程 ----


def main() -> int:
    ap = argparse.ArgumentParser(description="训练样本语料 BPE 分词器")
    ap.add_argument("--vocab-size", type=int, default=2048)
    ap.add_argument("--min-frequency", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--force", action="store_true", help="忽略已有语料文件，重新生成")
    args = ap.parse_args()

    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    lc.DATA_DIR.mkdir(parents=True, exist_ok=True)
    lc.RESULT_DIR.mkdir(parents=True, exist_ok=True)

    corpus_path = lc.DATA_DIR / "corpus.txt"
    sft_path = lc.DATA_DIR / "sft.jsonl"
    rl_path = lc.DATA_DIR / "rl.jsonl"

    if args.force or not all(p.exists() for p in (corpus_path, sft_path, rl_path)):
        corpus_text, sft, rl_tasks = build_corpus(args.seed)
        corpus_path.write_text(corpus_text, encoding="utf-8")
        with sft_path.open("w", encoding="utf-8") as f:
            for row in sft:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        with rl_path.open("w", encoding="utf-8") as f:
            for row in rl_tasks:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(
            f"[01] 生成样本语料: {corpus_path} ({corpus_path.stat().st_size:,} bytes), "
            f"{len(sft)} 条 SFT 指令对 -> {sft_path}, "
            f"{len(rl_tasks)} 条 RL 可验证任务 -> {rl_path}"
        )
    else:
        print(f"[01] 复用已有语料: {corpus_path}, {sft_path}, {rl_path}")

    tok = Tokenizer(models.BPE())
    # GPT-4 风格正则预分词 + Byte-Level（对齐 nanochat 的 tiktoken 路线）
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(pattern=GPT4_SPLIT_PATTERN, behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False),
        ]
    )
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=[lc.EOS, lc.PAD],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    t0 = time.time()
    tok.train([str(corpus_path)], trainer)
    train_secs = time.time() - t0

    out_path = lc.TOK_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_path))

    # ---- 校验：roundtrip 与压缩率 ----
    text = corpus_path.read_text(encoding="utf-8")
    probe_lines = [ln for ln in text.splitlines() if ln.strip()][:50]
    bad = [
        ln for ln in probe_lines if tok.decode(tok.encode(ln).ids, skip_special_tokens=False) != ln
    ]
    enc = tok.encode(text)
    n_tokens = len(enc.ids)
    n_bytes = len(text.encode("utf-8"))
    eos = tok.token_to_id(lc.EOS)
    eos_in_corpus = enc.ids.count(eos)

    info = {
        "tokenizer": "ByteLevel BPE (tokenizers 库, GPT-4 风格正则预分词)",
        "pre_tokenizer": "Regex(GPT-4 pattern, isolated) + ByteLevel",
        "corpus_file": str(corpus_path.relative_to(lc.ROOT)),
        "corpus_bytes": n_bytes,
        "vocab_size": tok.get_vocab_size(),
        "requested_vocab_size": args.vocab_size,
        "special_tokens": {lc.EOS: tok.token_to_id(lc.EOS), lc.PAD: tok.token_to_id(lc.PAD)},
        "corpus_tokens": n_tokens,
        "eos_occurrences": eos_in_corpus,
        "bytes_per_token": round(n_bytes / n_tokens, 3),
        "roundtrip_checked": len(probe_lines),
        "roundtrip_failures": len(bad),
        "train_seconds": round(train_secs, 2),
        "seed": args.seed,
    }
    (out_path.parent / "tokenizer_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        f"[01] BPE 训练完成: vocab={tok.get_vocab_size()} (目标 {args.vocab_size}), "
        f"eos_id={tok.token_to_id(lc.EOS)}, pad_id={tok.token_to_id(lc.PAD)}"
    )
    print(
        f"[01] 语料 {n_bytes:,} bytes -> {n_tokens:,} tokens "
        f"({info['bytes_per_token']} bytes/token), roundtrip 失败 {len(bad)}/{len(probe_lines)}"
    )
    print(f"[01] 已导出 {out_path}")
    if bad:
        print("[01] roundtrip 失败样例:", bad[:2], file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
