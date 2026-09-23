# -*- coding: utf-8 -*-
"""
算法 6：C99 主题分割（Choi, 2000）

比 TextTiling 更严密的词法方法，用「相似度矩阵的局部密度」而非曲线谷底：
1. 句子两两余弦相似度，得 n×n 矩阵
2. 排序后的矩阵按列重排，得到条带化的相似度矩阵
   同一主题的句子在矩阵中聚成方块，方块边界即主题边界
3. 用 divisive 聚类：递归把矩阵中「最不相似」的一列切出去
4. 用「内部密度 / 外部密度」的比值决定是否继续切

实现上取其实用版本：对相似度矩阵做 rank transform，
用贪心 divisive 聚类切到目标块数，密度比作为停止判据。
"""
import re

import numpy as np

from splitter_base import BaseSplitter, Boundary, Segmentation
from model_hub import TextEncoder
from token_map import sentence_token_spans

_TOKEN_RE = re.compile(r"[a-z0-9]+|[一-鿿]", re.IGNORECASE)


def _tokens(text):
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class C99Splitter(BaseSplitter):
    name = "c99"
    kind = "lexical"

    def __init__(self, target_chunks=None, std_coeff=1.2, min_tokens=64,
                 max_sentences=200):
        self.target_chunks = target_chunks
        self.std_coeff = std_coeff
        self.min_tokens = min_tokens
        self.max_sentences = max_sentences

    def _sim_matrix(self, sents):
        vocab = {}
        rows = []
        for s in sents:
            v = {}
            for t in _tokens(s):
                v[t] = v.get(t, 0) + 1
            for t in v:
                vocab.setdefault(t, len(vocab))
            rows.append(v)
        M = np.zeros((len(rows), len(vocab)), dtype=np.float32)
        for i, v in enumerate(rows):
            for t, c in v.items():
                M[i, vocab[t]] = c
        M /= (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
        return M @ M.T

    @staticmethod
    def _rank_transform(S):
        """按列做 rank transform：把相似度值替换为它在列中的排名。"""
        n = S.shape[0]
        R = np.zeros_like(S)
        for j in range(n):
            order = np.argsort(np.argsort(S[:, j]))
            R[:, j] = order
        return R

    @staticmethod
    def _density(S, lo, hi):
        """块 [lo, hi) 的内部密度：块内相似度均值。"""
        if hi - lo <= 1:
            return 0.0
        blk = S[lo:hi, lo:hi]
        return float(blk.sum() / max((hi - lo) * (hi - lo), 1))

    def split(self, text):
        enc = TextEncoder.get()
        spans = sentence_token_spans(text, enc.tok)
        if len(spans) < 3:
            return Segmentation(algo=self.name, boundaries=[], n_tokens=spans[-1][2])

        sents = [s for s, _, _ in spans]
        if len(sents) > self.max_sentences:
            # 超长文只取前 max_sentences 句，避免 O(n^2) 爆内存
            sents = sents[:self.max_sentences]
            spans = spans[:self.max_sentences]
        total = spans[-1][2]

        S = self._sim_matrix(sents)
        R = self._rank_transform(S)

        # 贪心 divisive：每次把使「切分后密度提升最大」的位置切开
        bounds = set()
        segments = [(0, len(sents))]
        target = self.target_chunks or max(2, len(sents) // 8)
        while len(segments) < target:
            best = None
            for lo, hi in segments:
                if hi - lo < 2:
                    continue
                d_in = self._density(R, lo, hi)
                for k in range(lo + 1, hi):
                    d_l = self._density(R, lo, k)
                    d_r = self._density(R, k, hi)
                    gain = (d_l + d_r) / 2 - d_in
                    if best is None or gain > best[0]:
                        best = (gain, k, lo, hi)
            if best is None or best[0] <= 0:
                break
            _, k, lo, hi = best
            bounds.add(k)
            segments.remove((lo, hi))
            segments.extend([(lo, k), (k, hi)])

        out = []
        for k in sorted(bounds):
            pos = spans[k][2]
            if pos > self.min_tokens and all(b.pos != pos for b in out):
                out.append(Boundary(pos=pos, reason=f"C99 聚类断点 @ 句{k}"))
        return Segmentation(algo=self.name, boundaries=out, n_tokens=total)
