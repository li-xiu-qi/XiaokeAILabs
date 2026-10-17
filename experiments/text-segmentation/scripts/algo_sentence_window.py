# -*- coding: utf-8 -*-
"""
算法：句子窗口（SentenceWindowSplitter / SentenceWindowNodeParser 精简复现）

参考 LangChain SentenceWindowSplitter 与 LlamaIndex SentenceWindowNodeParser。

机制（严格说是检索策略，但框架把它当分块方案）：
- 文档先切成句子
- 每个 chunk = 一个中心句 + 前后各 window_size 句的上下文窗口
- 检索时用中心句匹配，命中后把完整窗口喂给生成模型

边界质量语境下的等价实现：
P_k 关心「断点是否落在真边界附近」。句子窗口的断点是窗口外边界，
即每隔 (2*window_size+1) 句切一刀。这里按该周期产生断点，
window_size 越大，块越大、断点越稀。

与 fixed 的区别：块边界对齐句子，且块大小是「奇数句窗口」而非定长 token。
与 semantic 的区别：不看语义，纯按句索引周期切。

参数 window_size=1 → 每 3 句一块（前1+中心+后1）。
"""
from splitter_base import BaseSplitter, Boundary, Segmentation
from model_hub import TextEncoder
from token_map import sentence_token_spans


class SentenceWindowSplitter(BaseSplitter):
    name = "sentence_window"
    kind = "rule"

    def __init__(self, window_size=1):
        # 每块包含句数 = 2*window_size + 1
        self.window_size = window_size

    def split(self, text):
        enc = TextEncoder.get()
        spans = sentence_token_spans(text, enc.tok)
        if len(spans) < 2:
            return Segmentation(algo=self.name, boundaries=[], n_tokens=spans[-1][2])

        total = spans[-1][2]
        block = 2 * self.window_size + 1
        bounds = []
        # 每隔 block 句，在该句末尾切一刀
        for i in range(block - 1, len(spans) - 1, block):
            pos = spans[i][2]  # 第 i 句的结束 token 位置
            if 0 < pos < total:
                bounds.append(Boundary(pos=pos,
                                       reason=f"句窗周期 {block}"))
        return Segmentation(algo=self.name, boundaries=bounds, n_tokens=total)
