# -*- coding: utf-8 -*-
"""
分割器统一接口

所有分割算法实现同一个签名：
    split(text: str) -> list[Boundary]

其中 Boundary 是一个「断点位置」，用 token 偏移量表示（0-based，指第几个 token 之后切开）。
统一用 token 偏移而不是字符偏移，是为了让不同算法在同一把尺子上比较——
字符数受中文/英文/数字混排影响极大，token 数才是模型实际看到的信息量。

断点语义：boundaries[i] = k 表示「第 k 个 token 之后切开」，即前 k 个 token 是第一块。
首尾不加断点（0 和 n_tokens 是文档固有边界，不是算法决策）。
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Boundary:
    """一个断点。pos = 第几个 token 之后切开。"""
    pos: int
    reason: str = ""          # 断点成因，便于诊断（句子结束/主题转移/结构标记/窗口填满）
    score: float = 0.0        # 该断点的置信度（语义算法的相邻句差异度等）


@dataclass
class Segmentation:
    """一次分割结果。"""
    algo: str
    boundaries: list = field(default_factory=list)   # list[Boundary]
    n_tokens: int = 0

    def chunk_token_spans(self):
        """返回 [(start, end), ...]，覆盖 [0, n_tokens)。"""
        cuts = sorted({0, self.n_tokens} | {b.pos for b in self.boundaries
                                            if 0 < b.pos < self.n_tokens})
        return list(zip(cuts[:-1], cuts[1:]))


class BaseSplitter(ABC):
    """分割器基类。"""

    name: str = "base"
    kind: str = "unknown"     # rule / lexical / semantic / structural / model

    @abstractmethod
    def split(self, text: str) -> Segmentation:
        """把文本切成块，返回断点列表。"""
        raise NotImplementedError
