# -*- coding: utf-8 -*-
"""
算法 1：固定 token 窗口（基线）

不做任何语义判断，每 window 个 token 切一刀，最后一块不满也单独成块。
可选 sentence_boundary=True 时把切口回退到最近的句子边界，避免切断句子。

这是所有 RAG 系统的默认基线（LangChain RecursiveCharacterTextSplitter 的退化形态）。
它的作用是给质量-成本坐标系定原点：任何算法如果质量不如它，就不值得那笔开销。
"""
from splitter_base import BaseSplitter, Boundary, Segmentation
from model_hub import TextEncoder
from token_map import sentence_token_spans


class FixedWindowSplitter(BaseSplitter):
    name = "fixed_window"
    kind = "rule"

    def __init__(self, window=256, overlap=0, sentence_boundary=False):
        self.window = window
        self.overlap = overlap
        self.sentence_boundary = sentence_boundary

    def split(self, text):
        total = TextEncoder.get().n_tokens(text)
        bounds = []

        if self.sentence_boundary:
            spans = sentence_token_spans(text, TextEncoder.get().tok)
            sent_ends = [e for _, _, e in spans]
            pos = self.window
            while pos < total:
                # 回退到不超过 pos 的最近句子边界
                cand = [e for e in sent_ends if e <= pos]
                cut = cand[-1] if cand else pos
                if cut > 0 and cut < total and all(b.pos != cut for b in bounds):
                    bounds.append(Boundary(pos=cut, reason="窗口回退到句界"))
                pos += self.window
                if self.overlap:
                    pos -= self.overlap
            return Segmentation(algo=self.name, boundaries=bounds, n_tokens=total)

        pos = self.window
        while pos < total:
            bounds.append(Boundary(pos=pos, reason="窗口填满"))
            pos += self.window
            if self.overlap:
                pos -= self.overlap
        return Segmentation(algo=self.name, boundaries=bounds, n_tokens=total)
