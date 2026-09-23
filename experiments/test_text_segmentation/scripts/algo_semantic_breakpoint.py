# -*- coding: utf-8 -*-
"""
算法 2：句向量语义断句

流程：
1. 切句（model_hub.sentences_of，中英规则切分）
2. 逐句编码成句向量（一次 batch，比逐对算快一个量级）
3. 算相邻句余弦相似度序列
4. 断点判据（两种可选）：
   - threshold：相似度 < t 即断
   - percentile：相似度低于全序列第 p 百分位即断（自适应，无阈值调参）
5. 尺寸约束：块 token 数超过 max_tokens 时强制在最大相似度处切开，
   低于 min_tokens 时不切（防止碎块）

这一步是 archive_predecessors/test_semantic_splitter 里 v2/v3 的核心逻辑，
原实现散在 improved_semantic_splitter_v2/v3 两个文件且依赖硅基流动 API，
这里改成本地 sentence-transformers，并补上 percentile 自适应判据和尺寸硬约束。
"""
import numpy as np

from splitter_base import BaseSplitter, Boundary, Segmentation
from model_hub import TextEncoder
from token_map import sentence_token_spans


class SemanticSplitter(BaseSplitter):
    name = "semantic_breakpoint"
    kind = "semantic"

    def __init__(self, threshold=None, percentile=10, min_tokens=64,
                 max_tokens=512, model=None):
        self.threshold = threshold
        self.percentile = percentile
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.model_name = model

    def split(self, text):
        enc = TextEncoder.get(self.model_name) if self.model_name else TextEncoder.get()
        spans = sentence_token_spans(text, enc.tok)
        if len(spans) < 2:
            return Segmentation(algo=self.name, boundaries=[], n_tokens=spans[-1][2])

        sents = [s for s, _, _ in spans]
        embs = enc.encode(sents)

        # 相邻句相似度，长度 n-1
        sims = np.array([float(embs[i] @ embs[i + 1]) for i in range(len(embs) - 1)])

        if self.threshold is not None:
            cut_sims = sims < self.threshold
            basis = f"阈值 {self.threshold}"
        else:
            thr = float(np.percentile(sims, self.percentile))
            cut_sims = sims < thr
            basis = f"百分位 P{self.percentile} (实际阈值 {thr:.3f})"

        bounds = []
        cur_start = 0
        for i in range(len(sims)):
            cur_end_tok = spans[i + 1][2]
            cur_tokens = cur_end_tok - spans[cur_start][1]
            forced = cur_tokens >= self.max_tokens
            too_short = cur_tokens < self.min_tokens

            if forced:
                # 强制切：在当前块内找相似度最低的位置
                lo, hi = cur_start, i + 1
                seg = sims[lo:hi]
                j = lo + int(np.argmin(seg)) if len(seg) else i
                pos = spans[j + 1][2]
                bounds.append(Boundary(pos=pos, reason=f"超长强制切 @ 句{j+1}"))
                cur_start = j + 1
            elif cut_sims[i] and not too_short:
                pos = spans[i + 1][2]
                bounds.append(Boundary(pos=pos, score=float(sims[i]),
                                       reason=f"语义断点 @ 句{i+1}"))
                cur_start = i + 1

        return Segmentation(algo=self.name, boundaries=bounds,
                            n_tokens=spans[-1][2])
