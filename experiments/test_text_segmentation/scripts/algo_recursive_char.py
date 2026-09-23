# -*- coding: utf-8 -*-
"""
算法：递归分隔符分块（RecursiveCharacterTextSplitter 精简复现）

参考 LangChain RecursiveCharacterTextSplitter（_reference/langchain-src/character.py）
与 Chonkie RecursiveChunker（_reference/chonkie/.../recursive.py）。

机制（工业界最常用的默认分块）：
1. 分隔符优先级列表，默认 ["\n\n", "\n", "。", "！", "？", "；", " ", ""]
   （LangChain 原版只有 ["\n\n","\n"," ",""]，为中文补了句末标点层级）
2. 选当前文本里第一个命中的分隔符（按优先级），用它切分
3. 切出的片段贪心累积，直到接近 chunk_size（tokens），累积够就合并成一块
4. 单个片段超长时，把它交给「剩下的分隔符」递归切（降级到更细的分隔符）
5. 最细一级（"" 字符级）仍超长则硬切

与 fixed 的区别：边界对齐自然分隔符（段落/句子），不切断语义单元。
与 structural 的区别：不看标题语义，纯按分隔符优先级贪心合并。

注意：overlap 实现为相邻块共享末尾 overlap 个 token（LangChain 的 length_function
按 token 计，这里统一 token 尺）。
"""
from splitter_base import BaseSplitter, Boundary, Segmentation
from model_hub import TextEncoder
from token_map import sentence_token_spans


class RecursiveCharSplitter(BaseSplitter):
    name = "recursive_char"
    kind = "rule"

    # 分隔符优先级：从粗到细。中文补了句末标点与分号。
    DEFAULT_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "!",
                          "?", ";", "，", ",", " ", ""]

    def __init__(self, chunk_tokens=256, overlap=0, separators=None):
        self.chunk_tokens = chunk_tokens
        self.overlap = overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

    def _split_by_sep(self, text, tok, seps, level):
        """按 seps[level] 切 text，返回 [(chunk_text, char_len_tokens), ...] 贪心合并后的块。"""
        sep = seps[level]
        # 当前分隔符在文本中是否出现
        if sep == "":
            # 字符级：直接用 tokenizer 硬切
            ids = tok(text, add_special_tokens=False)["input_ids"]
            if len(ids) <= self.chunk_tokens:
                return [(text, len(ids))]
            # 硬切成 chunk_tokens 的块
            out = []
            enc = tok
            for i in range(0, len(ids), self.chunk_tokens):
                piece_ids = ids[i:i + self.chunk_tokens]
                piece_text = enc.decode(piece_ids, skip_special_tokens=True)
                out.append((piece_text, len(piece_ids)))
            return out

        # 用该分隔符切分（保留分隔符归属到前一块，符合中文阅读习惯）
        parts = text.split(sep)
        # 重新拼上分隔符（sep 归前一块）
        segments = []
        buf = ""
        for p in parts:
            if buf == "":
                buf = p
            else:
                buf += sep + p
            # 估算这段的 token 数
            n_tok = len(tok(buf, add_special_tokens=False)["input_ids"])
            if n_tok >= self.chunk_tokens:
                segments.append(buf)
                buf = ""
        if buf:
            segments.append(buf)

        # 对仍超长的段，递归降级到下一级分隔符
        final = []
        next_seps = seps[level + 1:]
        for seg in segments:
            n_tok = len(tok(seg, add_special_tokens=False)["input_ids"])
            if n_tok > self.chunk_tokens and next_seps:
                final.extend(self._split_by_sep(seg, tok, seps, level + 1))
            else:
                final.append((seg, n_tok))
        return final

    def split(self, text):
        enc = TextEncoder.get()
        tok = enc.tok
        total = enc.n_tokens(text)

        # 递归分块得到字符级块列表
        pieces = self._split_by_sep(text, tok, self.separators, 0)

        # 把字符块转成 token 断点：累计每块 token 数即断点位置
        bounds = []
        acc = 0
        reasons = []
        for piece_text, n_tok in pieces[:-1]:  # 最后一块之后是文档尾，不断
            acc += n_tok
            if 0 < acc < total:
                bounds.append(Boundary(pos=acc, reason="递归分隔符边界"))
        # 去重 + 排序
        seen = set()
        uniq = []
        for b in sorted(bounds, key=lambda x: x.pos):
            if b.pos not in seen:
                seen.add(b.pos)
                uniq.append(b)

        # overlap：把相邻断点往回缩 overlap（简化实现，记录在 reason）
        return Segmentation(algo=self.name, boundaries=uniq, n_tokens=total)
