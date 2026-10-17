# -*- coding: utf-8 -*-
"""
算法 7：BM25 段落边界打分

思路：词法主题分割的另一条路。把相邻句子对当作「查询-文档」，
用 BM25 打分衡量两句话在词项分布上的独立性——
如果相邻两句共享的关键词很少、且各自的稀有词都很突出，
它们很可能属于不同主题。

具体做法：
1. 以整篇文档为语料库，算每句的 BM25 词权重（IDF 来自全篇）
2. 相邻句 i, i+1 的「边界分」= 1 - 两旬 BM25 向量的余弦
3. 超过阈值的边界即主题转移

与 TextTiling 的差别：TextTiling 用原始词频余弦，BM25 用带饱和与长度归一化
的权重，对「高频虚词偶然共现」更不敏感。
"""
import math
import re
from collections import Counter

import numpy as np

from splitter_base import BaseSplitter, Boundary, Segmentation
from model_hub import TextEncoder
from token_map import sentence_token_spans

_TOKEN_RE = re.compile(r"[a-z0-9]+|[一-鿿]", re.IGNORECASE)


def _tokens(text):
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class BM25Splitter(BaseSplitter):
    name = "bm25_boundary"
    kind = "lexical"

    def __init__(self, k1=1.5, b=0.75, threshold=None, percentile=15,
                 min_tokens=64):
        self.k1 = k1
        self.b = b
        self.threshold = threshold
        self.percentile = percentile
        self.min_tokens = min_tokens

    def split(self, text):
        enc = TextEncoder.get()
        spans = sentence_token_spans(text, enc.tok)
        if len(spans) < 3:
            return Segmentation(algo=self.name, boundaries=[], n_tokens=spans[-1][2])
        total = spans[-1][2]

        sents = [_tokens(s) for s, _, _ in spans]
        n_docs = len(sents)
        avgdl = sum(len(s) for s in sents) / n_docs

        # 文档频率
        df = Counter()
        for s in sents:
            for t in set(s):
                df[t] += 1
        idf = {t: math.log(1 + (n_docs - c + 0.5) / (c + 0.5)) for t, c in df.items()}

        # BM25 稀疏向量
        vecs = []
        vocab = {}
        for s in sents:
            tf = Counter(s)
            v = {}
            for t, c in tf.items():
                w = idf[t] * (c * (self.k1 + 1)) / (
                    c + self.k1 * (1 - self.b + self.b * len(s) / avgdl))
                v[t] = w
            for t in v:
                vocab.setdefault(t, len(vocab))
            vecs.append(v)
        M = np.zeros((len(vecs), len(vocab)), dtype=np.float32)
        for i, v in enumerate(vecs):
            for t, w in v.items():
                M[i, vocab[t]] = w
        M /= (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)

        # 相邻句余弦 → 边界分 = 1 - cos
        sims = np.array([float(M[i] @ M[i + 1]) for i in range(len(M) - 1)])
        bscore = 1.0 - sims

        if self.threshold is not None:
            thr = self.threshold
            basis = f"固定阈值 {thr:.3f}"
        else:
            thr = float(np.percentile(bscore, 100 - self.percentile))
            basis = f"百分位 P{self.percentile} (实际阈值 {thr:.3f})"

        bounds = []
        cur_tokens = 0
        for i in range(len(bscore)):
            pos = spans[i + 1][2]
            if bscore[i] >= thr and pos - cur_tokens >= self.min_tokens:
                bounds.append(Boundary(pos=pos, score=float(bscore[i]),
                                       reason=f"BM25 边界分 {bscore[i]:.3f}"))
                cur_tokens = pos
        return Segmentation(algo=self.name, boundaries=bounds, n_tokens=total)
