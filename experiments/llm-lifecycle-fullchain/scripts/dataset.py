#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dataset.py — 统一语料加载与流式数据读取模块。

设计严格对标 karpathy/nanochat (nanochat/dataset.py):
1. 支持流式读取标准 Parquet 预训练数据集分片（如 ClimbMix / FineWeb-Edu）
2. 支持流式读取单个文本文件或大语料文本（如 $XL_DATA_DIR/corpus.txt）
3. 支持统一文档截断（doc_cap）与字符上限限制（max_chars）
4. 提供标准 SFT 与 RL 数据集加载器，与核心训练算法完全解耦
5. 核心模块不内联任何静态玩具数据，离线快速自测数据通过 scripts/prepare_data.py --source toy 生成
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llm_common as lc  # noqa: E402


# 统一获取项目根目录；数据目录走运行契约的 lc.data_dir()（$XL_DATA_DIR 或 .work/data）
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def default_data_dir() -> Path:
    """数据目录：只取 XL_DATA_DIR，不回退到仓内 data/。"""
    return lc.data_dir()


def list_parquet_files(data_dir: str | Path | None = None) -> list[str]:
    """遍历指定目录下的所有 parquet 语料分片（严格对标 nanochat.dataset.list_parquet_files）。"""
    target_dir = Path(data_dir) if data_dir else default_data_dir()
    if not target_dir.exists():
        return []
    files = sorted([
        str(target_dir / f) for f in os.listdir(target_dir)
        if f.endswith(".parquet") and not f.endswith(".tmp")
    ])
    return files


def parquet_file_iter_batched(filepath: str | Path) -> Iterator[list[str]]:
    """流式读取单个 parquet 文件的 row_group（对标 nanochat 的 parquets_iter_batched）。

    用于外部大语料（如 karpathy/climbmix-400b-shuffle 的 shard_XXXXX.parquet）：
    逐 row_group 读取 text 列，不把整个文件载入内存。
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError("读取 Parquet 语料需要 pyarrow 库，请先通过 pip install pyarrow 安装。")
    pf = pq.ParquetFile(str(filepath))
    for rg_idx in range(pf.num_row_groups):
        rg = pf.read_row_group(rg_idx)
        texts = rg.column("text").to_pylist()
        yield texts


def parquets_iter_batched(split: str = "train", data_dir: str | Path | None = None) -> Iterator[list[str]]:
    """以批次形式流式读取 Parquet 文件的 row_group（严格对标 nanochat.dataset.parquets_iter_batched）。"""
    assert split in ["train", "val"], "split 必须为 'train' 或 'val'"
    parquet_paths = list_parquet_files(data_dir)
    if not parquet_paths:
        return
    selected_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
    for filepath in selected_paths:
        yield from parquet_file_iter_batched(filepath)


def text_file_iter_batched(filepath: str | Path, chunk_size: int = 1000) -> Iterator[list[str]]:
    """逐行/分批流式读取纯文本语料文件，避免大文件一次性撑爆内存。"""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"未检测到语料文件: {path}\n"
            f"如需使用标准公开数据集，请指定 --corpus-path 参数或运行 python scripts/prepare_data.py；\n"
            f"如需进行快速自测验证，请运行: python scripts/prepare_data.py --source toy"
        )

    batch = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                batch.append(line_str)
                if len(batch) >= chunk_size:
                    yield batch
                    batch = []
    if batch:
        yield batch


def iter_texts(
    data_source: str | Path | None = None,
    split: str = "train",
    max_chars: int | None = None,
    doc_cap: int = 10000,
) -> Iterator[str]:
    """通用的统一文本迭代生成器（严格对标 nanochat/scripts/tok_train.py 的 text_iterator）。

    支持逻辑：
    1. 若 data_source 为目录且含 parquet，走 parquets_iter_batched
    2. 若 data_source 为单个 .parquet 文件，走 parquet_file_iter_batched（外部大语料分片）
    3. 若 data_source 为文本文件，走 text_file_iter_batched
    4. 支持按 doc_cap 单文档字符截断
    5. 支持按 max_chars 累计字符上限自动退出
    """
    source_path = Path(data_source) if data_source else (default_data_dir() / "corpus.txt")
    total_chars = 0

    if source_path.is_dir():
        batch_iter = parquets_iter_batched(split=split, data_dir=source_path)
    elif source_path.suffix.lower() == ".parquet":
        batch_iter = parquet_file_iter_batched(filepath=source_path)
    else:
        batch_iter = text_file_iter_batched(filepath=source_path)

    for batch in batch_iter:
        for doc in batch:
            doc_text = doc
            if len(doc_text) > doc_cap:
                doc_text = doc_text[:doc_cap]
            total_chars += len(doc_text)
            yield doc_text
            if max_chars is not None and total_chars >= max_chars:
                return


def load_sft_dataset(filepath: str | Path | None = None) -> list[dict]:
    """加载 SFT 指令微调数据集（JSONL 格式）。"""
    path = Path(filepath) if filepath else (default_data_dir() / "sft.jsonl")
    if not path.exists():
        raise FileNotFoundError(
            f"未检测到 SFT 数据文件: {path}\n"
            f"请指定 --sft-path 或先运行 python scripts/prepare_data.py 生成数据。"
        )
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                records.append(json.loads(line_str))
    return records


def load_rl_dataset(filepath: str | Path | None = None) -> list[dict]:
    """加载强化学习 RL (如 GRPO) 可验证任务数据集（JSONL 格式）。"""
    path = Path(filepath) if filepath else (default_data_dir() / "rl.jsonl")
    if not path.exists():
        raise FileNotFoundError(
            f"未检测到 RL 任务数据文件: {path}\n"
            f"请指定 --rl-path 或先运行 python scripts/prepare_data.py 生成数据。"
        )
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                records.append(json.loads(line_str))
    return records
