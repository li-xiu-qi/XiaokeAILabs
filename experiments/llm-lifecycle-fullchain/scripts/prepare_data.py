#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""prepare_data.py — 全生命周期实验数据集自动化下载与准备脚本。

提供两种真实可执行的数据就绪方式：
1. 真实公开数据集接入（--source real，默认）：
   - 预训练语料：Wikitext-2 高质量开源纯文本语料 -> $XL_DATA_DIR/corpus.txt
   - SFT 指令微调：Stanford Alpaca 开源高质量指令问答集 -> $XL_DATA_DIR/sft.jsonl
   - RL 强化学习：OpenAI GSM8K 经典小学数学可验证推理集 -> $XL_DATA_DIR/rl.jsonl
   支持国内镜像自动回退（raw.githubusercontent.com / jsdelivr / ghproxy），免 Token、开箱即用。

2. 离线极速自检验证（--source toy）：
   - 零网络依赖，5 秒内生成离线自测小语料（基础语言事实与算术），用于快速验证单机环境与全流程代码链路。

用法：
    # 默认：下载并构建真实公开数据集
    python scripts/prepare_data.py

    # 离线环境：生成本地自测数据
    python scripts/prepare_data.py --source toy
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llm_common as lc

# ==========================================
# 离线自测数据（Toy Dataset）合成逻辑
# ==========================================

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
    print(f"已生成自测预训练语料: {corpus_file} ({len(corpus_text):,} 字符)")

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
    print(f"已生成自测 SFT 数据: {sft_file} ({len(sft)} 条记录)")

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
    print(f"已生成自测 RL 数据: {rl_file} ({len(rl_tasks)} 条任务)")


# ==========================================
# 真实公开数据集下载与标准化逻辑
# ==========================================

def get_data_dir(custom_dir: str | None = None) -> Path:
    if custom_dir:
        d = Path(custom_dir)
    elif os.environ.get("XL_DATA_DIR"):
        d = Path(os.environ["XL_DATA_DIR"])
    else:
        d = Path(__file__).resolve().parents[1] / ".work" / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def download_with_fallback(urls: list[str], timeout: int = 20) -> bytes:
    """按候选 URL 顺序尝试下载，避免单一源被墙或超时。"""
    last_err = None
    for u in urls:
        try:
            req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0 (XiaokeAILabs/1.0)"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
                if len(data) > 0:
                    return data
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"所有下载镜像均失败，最后报错: {last_err}")


def prepare_real_pretrain(out_dir: Path, max_lines: int = 50000) -> Path:
    corpus_file = out_dir / "corpus.txt"
    print(f"[*] 正在下载真实预训练语料 (Wikitext-2)...")
    urls = [
        "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/train.txt",
        "https://cdn.jsdelivr.net/gh/pytorch/examples@master/word_language_model/data/wikitext-2/train.txt",
        "https://ghproxy.net/https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/train.txt",
    ]
    raw_data = download_with_fallback(urls)
    lines = raw_data.decode("utf-8", errors="replace").splitlines()
    
    clean_lines = [l.strip() for l in lines if l.strip() and not l.strip().startswith("=")]
    if max_lines and len(clean_lines) > max_lines:
        clean_lines = clean_lines[:max_lines]
    
    with open(corpus_file, "w", encoding="utf-8") as f:
        f.write("\n".join(clean_lines) + "\n")
    print(f"[✓] 预训练语料就绪: {corpus_file} ({len(clean_lines)} 行文本, {corpus_file.stat().st_size / 1024 / 1024:.2f} MB)")
    return corpus_file


def prepare_real_sft(out_dir: Path, max_samples: int = 2000) -> Path:
    sft_file = out_dir / "sft.jsonl"
    print(f"[*] 正在下载真实 SFT 指令数据集 (Stanford Alpaca)...")
    urls = [
        "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json",
        "https://cdn.jsdelivr.net/gh/tatsu-lab/stanford_alpaca@main/alpaca_data.json",
        "https://ghproxy.net/https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json",
    ]
    raw_data = download_with_fallback(urls)
    alpaca_items = json.loads(raw_data.decode("utf-8", errors="replace"))
    
    records = []
    for item in alpaca_items[:max_samples]:
        inst = item.get("instruction", "").strip()
        inp = item.get("input", "").strip()
        out = item.get("output", "").strip()
        if not inst or not out:
            continue
        prompt = f"User: {inst}\n{inp}\nAssistant:" if inp else f"User: {inst}\nAssistant:"
        records.append({"prompt": prompt, "response": f" {out}"})
        
    with open(sft_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[✓] SFT 指令数据就绪: {sft_file} ({len(records)} 条高质量对话)")
    return sft_file


def prepare_real_rl(out_dir: Path, max_samples: int = 1500) -> Path:
    rl_file = out_dir / "rl.jsonl"
    print(f"[*] 正在下载真实强化学习推理数据集 (OpenAI GSM8K)...")
    urls = [
        "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl",
        "https://cdn.jsdelivr.net/gh/openai/grade-school-math@master/grade_school_math/data/train.jsonl",
        "https://ghproxy.net/https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl",
    ]
    raw_data = download_with_fallback(urls)
    lines = raw_data.decode("utf-8", errors="replace").splitlines()
    
    records = []
    for line in lines:
        if not line.strip():
            continue
        item = json.loads(line)
        q = item.get("question", "").strip()
        a_full = item.get("answer", "").strip()
        if not q or "####" not in a_full:
            continue
        ans = a_full.split("####")[-1].strip().replace(",", "")
        prompt = f"User: {q}\nAssistant:"
        records.append({
            "prompt": prompt,
            "answer": ans,
            "response": f" {a_full}"
        })
        if max_samples and len(records) >= max_samples:
            break
            
    with open(rl_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[✓] RL 强化学习数据就绪: {rl_file} ({len(records)} 条带客观标准答案的推导题)")
    return rl_file


def main():
    parser = argparse.ArgumentParser(description="XiaokeAILabs 全生命周期实验数据准备工具")
    parser.add_argument("--source", choices=["real", "toy"], default="real",
                        help="数据源类型：real 为真实开源数据集（默认），toy 为本地快速自测数据")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="数据落盘目录，未指定时默认取 $XL_DATA_DIR 或 ./.work/data")
    parser.add_argument("--seed", type=int, default=1234,
                        help="随机种子（仅在 --source toy 时生效）")
    parser.add_argument("--max-pretrain-lines", type=int, default=50000,
                        help="真实预训练文本最大截断行数（默认 50000）")
    parser.add_argument("--max-sft", type=int, default=2000,
                        help="SFT 样本条数（默认 2000）")
    parser.add_argument("--max-rl", type=int, default=1500,
                        help="RL 任务样本条数（默认 1500）")
    args = parser.parse_args()

    out_dir = get_data_dir(args.out_dir)
    print(f"=== XiaokeAILabs 数据准备阶段 ===")
    print(f"数据落盘路径: {out_dir.resolve()}")
    print(f"选择模式: {args.source.upper()}")

    if args.source == "toy":
        print("[*] 正在生成本地快速自测数据...")
        generate_toy_data(seed=args.seed, out_dir=out_dir)
        print("[✓] 离线自测数据生成完成！")
    else:
        prepare_real_pretrain(out_dir, max_lines=args.max_pretrain_lines)
        prepare_real_sft(out_dir, max_samples=args.max_sft)
        prepare_real_rl(out_dir, max_samples=args.max_rl)
        print("=== 真实数据集全部就绪，可直接执行后续 01~08 完整训练脚本 ===")


if __name__ == "__main__":
    main()
