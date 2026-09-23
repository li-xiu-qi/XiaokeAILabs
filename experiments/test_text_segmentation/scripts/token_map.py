# -*- coding: utf-8 -*-
"""
句子到 token 偏移的映射

所有语义类算法都在「句子」上做判断（断句、算相似度、聚类），
但断点最终必须落在 token 空间（见 splitter_base）。
本模块负责把句子边界换算成累计 token 数。

实现方式：逐句 tokenize，累加长度。比一次性 tokenize 全文再按 offset_mapping
反查更稳，因为不同句子的 tokenizer 边界行为可能不同（尤其中英混排时
BPE 会把跨句的连续拉丁字符并成一个 token）。
"""
from model_hub import sentences_of


def sentence_token_spans(text, tok):
    """返回 [(句子文本, 起始token, 结束token), ...]，覆盖全文。

    累加值可能与全文 tokenize 结果略有出入（±1-2 token），
    对 ±w 容忍窗的边界 F1 无影响。
    """
    sents = sentences_of(text)
    spans = []
    pos = 0
    for s in sents:
        n = len(tok(s, add_special_tokens=False)["input_ids"])
        spans.append((s, pos, pos + n))
        pos += n
    return spans


def boundary_from_sentence_index(spans, sent_idx):
    """第 sent_idx 个句子结束处的 token 位置。"""
    return spans[sent_idx][2]
