#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""generate_toy_dataset.py — 离线单元测试与快速冒烟语料生成器。

【重要说明】：
本文件仅用于无网络、无外部数据集时的离线单元测试与 CPU 快速冒烟验证。
生产训练与正式实验应采用标准公开数据集（如 FineWeb-Edu 采样切片与 SmolTalk 对话流）。
请勿将本文件作为正式大模型训练的数据基准。

输出落点（由 XL_DATA_DIR 指定，未设置时默认写入 .work/data；可用 --out-dir 覆盖）：
  - corpus.txt: 离线基础语言事实与语法语料
  - sft.jsonl: 离线指令问答与身份对齐样本
  - rl.jsonl: 离线可验证符号加减法题目
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import llm_common as lc  # noqa: E402

EOS = "<|endoftext|>"

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

ARITH_TRAIN_FRAC = 0.7

def build_arith_pairs() -> list[tuple[int, int, str, int]]:
    pairs = []
    for a in range(0, 10):
        for b in range(0, 10):
            if a + b <= 18:
                pairs.append((a, b, "plus", a + b))
            if a >= b:
                pairs.append((a, b, "minus", a - b))
    return pairs

def generate_toy_data(seed: int = 1234, out_dir: Path | None = None):
    out = out_dir or lc.data_dir()
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    qa_pairs: list[tuple[str, str]] = []
    blocks: list[str] = []

    # 1) 首都事实
    for country, capital in CAPITALS:
        blocks.append(
            f"The capital of {country} is {capital}.\n"
            f"{capital} is the capital city of {country}."
            + EOS
        )
        qa_pairs += [
            (f"What is the capital of {country}?", capital),
            (f"Which city is the capital of {country}?", capital),
        ]

    # 2) 水果颜色
    for fruit, color in FRUITS:
        blocks.append(
            f"The {fruit} is {color}.\n"
            f"A {fruit} has a {color} color."
            + EOS
        )
        qa_pairs.append((f"What color is a {fruit}?", f"The {fruit} is {color}."))

    # 3) 职业常识
    for person, job in PEOPLE:
        blocks.append(
            f"{person} is a {job}.\n"
            f"{person} works as a {job}."
            + EOS
        )
        qa_pairs.append((f"What does {person} do?", f"{person} is a {job}."))

    # 4) 语法句子
    grammar = [
        f"The {adj} {noun} {v3} the {obj}."
        for adj in ADJS
        for noun in NOUNS
        for v3 in VERBS_3RD
        for obj in OBJS
    ]
    rng.shuffle(grammar)
    for i in range(0, 600, 8):
        blocks.append("\n".join(grammar[i : i + 8]) + EOS)

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
        blocks.append("\n".join(grammar2[i : i + 10]) + EOS)

    # 5) 数词算术事实
    for i in range(0, len(NUMBER_WORDS), 4):
        blocks.append("\n".join(NUMBER_WORDS[i : i + 4]) + EOS)

    # 6) 算式语句
    arith = build_arith_pairs()
    rng.shuffle(arith)
    n_train = int(len(arith) * ARITH_TRAIN_FRAC)
    arith_train, _ = arith[:n_train], arith[n_train:]
    for i in range(0, len(arith_train), 7):
        blocks.append(
            "\n".join(f"{a} {op} {b} = {r}." for a, b, op, r in arith_train[i : i + 7]) + EOS
        )

    # 7) 身份问答与闲聊
    for q, a in IDENTITY + CHAT:
        blocks.append(f"User: {q}\nAssistant: {a}{EOS}")
        qa_pairs.append((q, a))
    rng.shuffle(qa_pairs)
    for i in range(0, len(qa_pairs), 5):
        blocks.append(
            "\n".join(f"User: {q}\nAssistant: {a}" for q, a in qa_pairs[i : i + 5]) + EOS
        )

    rng.shuffle(blocks)
    corpus_text = "\n".join(blocks) + "\n"

    # 写出预训练语料
    corpus_file = out / "corpus.txt"
    corpus_file.write_text(corpus_text, encoding="utf-8")
    print(f"已生成冒烟预训练语料: {corpus_file} ({len(corpus_text):,} 字符)")

    # 生成 SFT 指令数据
    sft = [{"prompt": f"User: {q}\nAssistant:", "response": f" {a}"} for q, a in qa_pairs]
    arith_sft = rng.sample(arith_train, k=15)
    sft += [
        {"prompt": f"User: What is {a} {op} {b}?\nAssistant:", "response": f" {r}"}
        for a, b, op, r in arith_sft
    ]
    sft_file = out / "sft.jsonl"
    with open(sft_file, "w", encoding="utf-8") as f:
        for item in sft:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"已生成冒烟 SFT 数据: {sft_file} ({len(sft)} 条记录)")

    # 生成 RL 可验证任务数据
    rl_tasks = [
        {
            "prompt": f"User: What is {a} {op} {b}?\nAssistant:",
            "answer": str(r),
            "in_corpus": (a, b, op) in {(x, y, o) for x, y, o, _ in arith_train},
        }
        for a, b, op, r in arith
    ]
    rl_file = out / "rl.jsonl"
    with open(rl_file, "w", encoding="utf-8") as f:
        for item in rl_tasks:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"已生成冒烟 RL 数据: {rl_file} ({len(rl_tasks)} 条任务)")

def main():
    parser = argparse.ArgumentParser(description="生成离线冒烟测试语料")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out-dir", type=str, default=None,
                        help="输出目录（缺省 $XL_DATA_DIR）")
    args = parser.parse_args()
    generate_toy_data(seed=args.seed, out_dir=Path(args.out_dir) if args.out_dir else None)

if __name__ == "__main__":
    main()
