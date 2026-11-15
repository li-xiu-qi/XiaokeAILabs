#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""01_train_tokenizer.py — BPE 分词器训练、评估度量导出与自检。

架构设计严格对标 karpathy/nanochat (scripts/tok_train.py 与 nanochat/tokenizer.py):
1. 核心训练脚本零硬编码语料，训练数据完全通过 dataset 模块流式提供；
2. 预分词采用 GPT-4 风格正则，逐块进行 Byte-Level BPE 聚合；
3. 支持标准超参数配置：--vocab-size、--max-chars、--doc-cap、--min-frequency；
4. 导出 tokenizer.json，并缓存 token_bytes 映射（用于与词表大小无关的 bits-per-byte 评估）；
5. 执行 roundtrip 编解码一致性严密校验。

用法示例：
    # 针对标准预训练语料（数据目录由 XL_DATA_DIR 提供）：
    python scripts/01_train_tokenizer.py --corpus-path "$XL_DATA_DIR/corpus.txt" --vocab-size 2048

    # 完整流水线请走唯一入口（输出目录由 XL_OUT_DIR 提供）：
    bash run.sh --config configs/smoke.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

import dataset
import llm_common as lc

# GPT-4 / tiktoken 风格预分词正则（对标 nanochat/tokenizer.py 中的 SPLIT_PATTERN 思想）
GPT4_SPLIT_PATTERN = (
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    "|[^\r\n\\p{L}\\p{N}]?\\p{L}+"
    "|\\p{N}{1,3}"
    "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*"
    "|\\s*[\\r\\n]+"
    "|\\s+(?!\\S)"
    "|\\s+"
)


def parse_args():
    parser = argparse.ArgumentParser(description="训练 BPE 分词器 (对标 nanochat tok_train)")
    parser.add_argument(
        "--corpus-path",
        type=str,
        default=str(lc.data_dir() / "corpus.txt"),
        help="语料文件或 Parquet 目录路径 (默认: $XL_DATA_DIR/corpus.txt)",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=2048,
        help="词表目标大小 (默认: 2048，标准全量可设为 32768 或更大)",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=2_000_000_000,
        help="训练分词器所用的最大字符上限 (默认: 2B 字符)",
    )
    parser.add_argument(
        "--doc-cap",
        type=int,
        default=10_000,
        help="单篇文档截断字符数 (默认: 10,000 字符)",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=1,
        help="最小词频阈值 (默认: 1)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(lc.out_dir() / "tokenizer"),
        help="分词器输出保存目录 (默认: $XL_OUT_DIR/tokenizer)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    corpus_path = Path(args.corpus_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[01] 训练配置:")
    print(f"  语料输入路径: {corpus_path}")
    print(f"  目标词表大小: {args.vocab_size:,}")
    print(f"  文档字符截断: {args.doc_cap:,}")
    print(f"  最大训练字符: {args.max_chars:,}")

    # 1. 初始化 BPE 模型与预分词流水线
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(pattern=GPT4_SPLIT_PATTERN, behavior="isolated", invert=False),
        pre_tokenizers.ByteLevel(add_prefix_space=False),
    ])
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=[lc.EOS, lc.PAD],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )

    # 2. 构建文本迭代流（对标 nanochat/scripts/tok_train.py 的 text_iterator）
    try:
        text_stream = dataset.iter_texts(
            data_source=corpus_path,
            split="train",
            max_chars=args.max_chars,
            doc_cap=args.doc_cap,
        )
    except FileNotFoundError as err:
        print(f"[01] 错误: {err}", file=sys.stderr)
        return 1

    # 3. 训练分词器
    t0 = time.time()
    tok.train_from_iterator(text_stream, trainer)
    train_secs = time.time() - t0
    print(f"[01] BPE 分词器训练完成，耗时: {train_secs:.2f}s")

    # 4. 导出分词器到指定目录
    tokenizer_file = output_dir / "tokenizer.json"
    tok.save(str(tokenizer_file))
    print(f"[01] 分词器已导出至: {tokenizer_file}")

    # 5. 缓存 token_bytes（对标 nanochat/scripts/tok_train.py:72~91 行）
    # 用于按原始字节计算 bits per byte
    vocab_size = tok.get_vocab_size()
    token_bytes_tensor = lc.token_bytes_from_tokenizer(tok)
    token_bytes_file = output_dir / "token_bytes.pt"
    torch.save(token_bytes_tensor, str(token_bytes_file))
    print(f"[01] 已缓存 token_bytes 至: {token_bytes_file}")

    # 6. 一致性校验与压缩率验证（roundtrip check）
    sample_text = (
        "Hello world! This is a verification sentence.\n"
        "Numbers: 123, 4567, 89. Contractions: I'm, you're, it's.\n"
        "Special chars: @#$%^&*(). Unicode: 深度学习与大语言模型。"
    )
    if corpus_path.is_file():
        # 从语料文件开头读取一部分用于验证
        with open(corpus_path, "r", encoding="utf-8", errors="replace") as f:
            corpus_head = "".join([f.readline() for _ in range(50)])
        if corpus_head.strip():
            sample_text = corpus_head

    probe_lines = [ln for ln in sample_text.splitlines() if ln.strip()][:50]
    bad = [
        ln for ln in probe_lines
        if tok.decode(tok.encode(ln).ids, skip_special_tokens=False) != ln
    ]

    enc = tok.encode(sample_text)
    n_tokens = max(len(enc.ids), 1)
    n_bytes = len(sample_text.encode("utf-8"))
    bytes_per_token = round(n_bytes / n_tokens, 3)

    info = {
        "tokenizer": "ByteLevel BPE (tokenizers, GPT-4 regex pre-split, nanochat aligned)",
        "corpus_path": str(corpus_path),
        "vocab_size": vocab_size,
        "requested_vocab_size": args.vocab_size,
        "special_tokens": {
            lc.EOS: tok.token_to_id(lc.EOS),
            lc.PAD: tok.token_to_id(lc.PAD),
        },
        "sample_tokens": n_tokens,
        "sample_bytes": n_bytes,
        "bytes_per_token": bytes_per_token,
        "roundtrip_checked_lines": len(probe_lines),
        "roundtrip_failures": len(bad),
        "train_seconds": round(train_secs, 2),
    }

    info_file = output_dir / "tokenizer_info.json"
    info_file.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[01] 校验结果: 采样 {n_bytes:,} 字节 -> {n_tokens:,} tokens ({bytes_per_token} bytes/token)")
    print(f"[01] roundtrip 失败数: {len(bad)}/{len(probe_lines)}")

    # 阶段指标（证据白名单可拉回）：配置 + 结果 + 运行标识
    metrics = dict(info)
    metrics["run_id"] = lc.run_id()
    metrics["config"] = {
        "corpus_path": str(corpus_path),
        "vocab_size": args.vocab_size,
        "max_chars": args.max_chars,
        "doc_cap": args.doc_cap,
        "min_frequency": args.min_frequency,
        "output_dir": str(output_dir),
    }
    metrics_file = output_dir.parent / "tokenizer_metrics.json"
    metrics_file.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[01] 阶段指标已写入: {metrics_file}")

    if bad:
        print("[01] roundtrip 校验出现失败样例:", bad[:2], file=sys.stderr)
        return 1

    print("[01] 分词器构建全部通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
