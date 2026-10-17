# -*- coding: utf-8 -*-
"""
模型封装：分词器与句向量编码器

统一由这里提供：
- tok(text) -> token 数（决定所有断点的坐标空间）
- encode(texts) -> L2 归一化向量（语义算法的输入）

默认模型 BAAI/bge-small-zh-v1.5（中文优化，512 维）。
多语言对照用 sentence-transformers/all-MiniLM-L6-v2（384 维）。

远程运行约定（本文件不写死路径）：
    HF_ENDPOINT=https://hf-mirror.com PYTHONPATH=~/text-seg/scripts \
      ~/xinfer-env/bin/python scripts/xxx.py
"""
import os
import threading

import numpy as np

MODEL_ZH = os.environ.get("TS_MODEL_ZH", "BAAI/bge-small-zh-v1.5")
MODEL_MULTI = os.environ.get("TS_MODEL_MULTI", "sentence-transformers/all-MiniLM-L6-v2")

_lock = threading.Lock()


class TextEncoder:
    """句向量编码器，懒加载，进程内单例。"""

    _instance = None

    def __init__(self, model_name=MODEL_ZH, device="cuda", batch_size=64):
        import torch
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=self.device)
        self.dim = self.model.get_sentence_embedding_dimension()

    @classmethod
    def get(cls, model_name=MODEL_ZH):
        with _lock:
            if cls._instance is None or cls._instance.model_name != model_name:
                cls._instance = cls(model_name)
            return cls._instance

    def encode(self, texts, show_progress_bar=False):
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

    @property
    def tok(self):
        return self.model.tokenizer

    def n_tokens(self, text):
        """token 数，不含 special tokens。"""
        return len(self.tok(text, add_special_tokens=False)["input_ids"])


def sentences_of(text):
    """把文本切成句子列表。中英混排，用规则切分。

    分隔符：。！？；!?; 以及换行。连续空白并入后一句。
    """
    import re

    parts = re.split(r"(?<=[。！？!?；;])\s*|\n+", text)
    return [p.strip() for p in parts if p and p.strip()]
