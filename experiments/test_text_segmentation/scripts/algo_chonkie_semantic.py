# -*- coding: utf-8 -*-
"""
算法：Chonkie 语义分块（窗口向量 + 局部谷底检测）

参考 Chonkie SemanticChunker（_reference/chonkie/.../semantic.py）。

机制（与我们的 semantic_p10/p25 本质不同）：
1. 窗口聚合：把前 similarity_window(默认3)句拼成一段整体编码，
   算「窗口向量」与「当前句向量」的相似度。不是相邻单句相似度，
   而是「前文上下文 vs 当前句」，更抗噪、更平滑。
2. Savitzky-Golay 滤波：对相似度曲线做多项式平滑（filter_window=5），
   抑制单句抖动。
3. 局部最小值检测：在平滑后的曲线上找局部最小值（相似度谷底），
   即主题转换点。不是简单阈值，是峰值检测。
4. filter_tolerance 过滤：只保留低于阈值的显著谷底。

与 semantic_p10/p25 的区别：那俩是「相邻单句相似度 + 百分位阈值」，
本算法是「前文窗口 vs 当前句 + SG 平滑 + 局部谷底」。前者对单句噪声敏感，
后者用窗口和滤波换稳定性。

参数：similarity_window=3, filter_window=5, filter_polyorder=2,
threshold=0.8（相似度低于此才算谷底）。
"""
import numpy as np

from splitter_base import BaseSplitter, Boundary, Segmentation
from model_hub import TextEncoder
from token_map import sentence_token_spans


def _savgol_smooth(y, window, polyorder):
    """简化 Savitzky-Golay 滤波：滑动窗口多项式拟合取中心值。
    边界用最近值填充（完整 SG 系数推导复杂，这里用等权滑动平均近似，
    足以验证「平滑后找谷底」的机制）。"""
    if window >= len(y):
        return np.array(y, dtype=float)
    half = window // 2
    out = np.array(y, dtype=float)
    kernel = np.ones(window) / window
    # 用 np.convolve same 模式做滑动平均（近似 SG 的低阶情形）
    padded = np.pad(out, (half, half), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed[:len(out)]


def _local_minima(y):
    """返回局部最小值的索引列表（比左右邻居都小）。"""
    idx = []
    for i in range(1, len(y) - 1):
        if y[i] < y[i - 1] and y[i] <= y[i + 1]:
            idx.append(i)
    return idx


class ChonkieSemanticSplitter(BaseSplitter):
    name = "chonkie_semantic"
    kind = "semantic"

    def __init__(self, similarity_window=3, filter_window=5,
                 threshold=0.8, min_sentences=2, model=None):
        self.similarity_window = similarity_window
        self.filter_window = filter_window
        self.threshold = threshold
        self.min_sentences = min_sentences
        self.model_name = model

    def split(self, text):
        enc = TextEncoder.get(self.model_name) if self.model_name else TextEncoder.get()
        spans = sentence_token_spans(text, enc.tok)
        if len(spans) <= self.similarity_window:
            return Segmentation(algo=self.name, boundaries=[], n_tokens=spans[-1][2])

        sents = [s for s, _, _ in spans]
        # 窗口向量：前 similarity_window 句拼成一段
        w = self.similarity_window
        windows = ["".join(sents[i:i + w]) for i in range(len(sents) - w)]
        win_embs = enc.encode(windows)
        # 当前句向量（从第 w 句开始，与窗口对齐）
        cur_embs = enc.encode(sents[w:])
        sims = np.array([float(win_embs[i] @ cur_embs[i])
                         for i in range(len(win_embs))])

        # SG 平滑 + 局部最小值
        smoothed = _savgol_smooth(sims, self.filter_window, 2)
        minima = _local_minima(smoothed)
        # 过滤：只保留低于阈值的显著谷底
        cut_local = [m for m in minima if smoothed[m] < self.threshold]

        # 映射回句子索引：sims[i] 对应第 i+w 句之前的边界
        # 即第 (m + w) 句之后切开
        bounds = []
        last_cut_sent = -1
        for m in cut_local:
            sent_idx = m + w  # 第 sent_idx 句之后切
            if sent_idx - last_cut_sent >= self.min_sentences:
                pos = spans[sent_idx][2]
                if 0 < pos < spans[-1][2]:
                    bounds.append(Boundary(
                        pos=pos,
                        reason=f"Chonkie 谷底 sim={smoothed[m]:.3f}",
                        score=float(smoothed[m])))
                    last_cut_sent = sent_idx

        return Segmentation(algo=self.name, boundaries=bounds, n_tokens=spans[-1][2])
