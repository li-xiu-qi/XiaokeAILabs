# -*- coding: utf-8 -*-
"""模型池与实验条件定义。

模型池按训练谱系选，覆盖：
  - 中文专用（large / small 两档）
  - 多语言 XLM-R 系（bge-m3，大规模弱监督对训练）
  - 多语言新一代（jina-v3，task instruction + RoPE）
  - 英文专用蒸馏小模型（MiniLM）
全部走 DGX Spark 本机 HF 缓存，离线加载。
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

MODELS = [
    {
        "name": "bge-m3",
        "hf_id": "BAAI/bge-m3",
        "family": "多语言 XLM-R 系（弱监督对 + 精调，默认 CLS 后归一化）",
        "lang": "multilingual",
        "dim": 1024,
    },
    {
        "name": "jina-v3",
        "hf_id": "jinaai/jina-embeddings-v3",
        "family": "多语言新一代（task instruction + RoPE + LoRA）",
        "lang": "multilingual",
        "dim": 1024,
    },
    {
        "name": "bge-large-zh",
        "hf_id": "BAAI/bge-large-zh-v1.5",
        "family": "中文专用（对比学习 + 难负样本，全参 large 档）",
        "lang": "zh",
        "dim": 1024,
    },
    {
        "name": "bge-small-zh",
        "hf_id": "BAAI/bge-small-zh-v1.5",
        "family": "中文专用（small 档蒸馏链）",
        "lang": "zh",
        "dim": 512,
    },
    {
        "name": "minilm-l6",
        "hf_id": "sentence-transformers/all-MiniLM-L6-v2",
        "family": "英文专用（6 层蒸馏，句子对监督）",
        "lang": "en",
        "dim": 384,
    },
]

# 落到数据的条件。单侧加长出 a/b 两个变体（_one_a 加 a 侧、_one_b 加 b 侧），
# 消除「永远只加同一侧」带来的 a/b 不对称偏差。
CONDITIONS = ["base", "rel_one_a", "rel_one_b", "irr_one_a", "irr_one_b", "rel_both", "irr_both"]

# 分析时折叠后的条件：rel_one / irr_one 为对应 a/b 两变体的平均。
ANALYSIS_CONDITIONS = ["base", "rel_one", "irr_one", "rel_both", "irr_both"]

# 语言对：zh-zh 同语言中文；en-en 同语言英文；zh-en 跨语言（a 中文 b 英文）
LANG_PAIRS = ["zh-zh", "en-en", "zh-en"]

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(DATA_DIR, "..", "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
