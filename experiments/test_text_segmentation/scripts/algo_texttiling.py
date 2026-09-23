# -*- coding: utf-8 -*-
"""
算法 5：TextTiling

经典词法主题分割算法（Hearst, 1997），不依赖神经网络：
1. 文本按 w 个 token 分成 token 序列，聚合为「伪句」
2. 相邻伪句计算词袋余弦相似度，得到相似度曲线
3. 曲线平滑（窗口平均），降低局部噪声
4. 在曲线的「谷底」（局部极小值）处断句，谷越深越可能是主题边界
5. 用深度分数排序，只保留最显著的 k 个边界（k 由目标块数或深度阈值决定）

这一步的价值：它是零模型依赖的基线。
语义算法都假设句向量能反映主题，TextTiling 直接看词汇重合，
两者结论不一致时说明「词汇换了但主题没变」或反过来。
"""
import math
import re

import numpy as np

from splitter_base import BaseSplitter, Boundary, Segmentation
from model_hub import TextEncoder

_TOKEN_RE = re.compile(r"[a-z0-9]+|[一-鿿]", re.IGNORECASE)


def _tokens(text):
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class TextTilingSplitter(BaseSplitter):
    name = "texttiling"
    kind = "lexical"

    def __init__(self, w=40, smooth=2, depth_threshold=None, max_boundaries=None,
                 min_tokens=64):
        self.w = w                # 伪句长度（token 数）
        self.smooth = smooth      # 平滑窗半径
        self.depth_threshold = depth_threshold
        self.max_boundaries = max_boundaries
        self.min_tokens = min_tokens

    def _pseudo_sentences(self, toks):
        return [toks[i:i + self.w] for i in range(0, len(toks), self.w)
                if len(toks[i:i + self.w]) >= max(3, self.w // 2)]

    def _sims(self, ps):
        vecs = []
        vocab = {}
        for s in ps:
            v = {}
            for t in s:
                v[t] = v.get(t, 0) + 1
            for t in v:
                vocab.setdefault(t, len(vocab))
            vecs.append(v)
        M = np.zeros((len(vecs), len(vocab)), dtype=np.float32)
        for i, v in enumerate(vecs):
            for t, c in v.items():
                M[i, vocab[t]] = c
        M /= (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
        sims = np.array([float(M[i] @ M[i + 1]) for i in range(len(M) - 1)])
        return sims

    def split(self, text):
        total = TextEncoder.get().n_tokens(text)
        toks = _tokens(text)
        ps = self._pseudo_sentences(toks)
        if len(ps) < 3:
            return Segmentation(algo=self.name, boundaries=[], n_tokens=total)

        sims = self._sims(ps)

        # 平滑
        if self.smooth > 0:
            k = np.ones(2 * self.smooth + 1) / (2 * self.smooth + 1)
            sims = np.convolve(sims, k, mode="same")

        # 深度分数：局部极小值到左右峰顶的落差均值
        depths = np.zeros(len(sims))
        for i in range(1, len(sims) - 1):
            if sims[i] < sims[i - 1] and sims[i] < sims[i + 1]:
                l = max((j for j in range(i - 1, -1, -1) if sims[j] > sims[i]),
                        key=lambda j: sims[j], default=i)
                r = min((j for j in range(i + 1, len(sims)) if sims[j] > sims[i]),
                        key=lambda j: sims[j], default=i)
                depths[i] = (sims[l] - sims[i] + sims[r] - sims[i]) / 2

        cand = [(i, depths[i]) for i in range(len(depths)) if depths[i] > 0]
        if self.depth_threshold is not None:
            cand = [c for c in cand if c[1] >= self.depth_threshold]
        if self.max_boundaries is not None:
            cand = sorted(cand, key=lambda x: -x[1])[:self.max_boundaries]

        # 相似度下标 i 对应伪句 i 与 i+1 之间，换算到 token 位置
        bounds = []
        for i, d in sorted(cand):
            pos = min((i + 1) * self.w, total - 1)
            if pos > self.min_tokens and all(b.pos != pos for b in bounds):
                bounds.append(Boundary(pos=pos, score=float(d),
                                       reason=f"TextTiling 谷深 {d:.3f}"))
        return Segmentation(algo=self.name, boundaries=bounds, n_tokens=total)
