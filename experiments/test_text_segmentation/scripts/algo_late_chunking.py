# -*- coding: utf-8 -*-
"""
算法 4：Late Chunking（迟分）

原理：先把整篇文档过一遍 encoder，拿到每个 token 的上下文相关隐状态；
再在 token 序列上按窗口切块，块向量 = 窗口内隐状态的均值池化。
相比「先切块再分别编码」，每个 token 的表示都看过全文上下文，
代词消解、跨句指代、省略主语这些在独立编码时丢失的信息得以保留。

实现要点（与 test_late_chunking 的实现差异）：
1. 原实现用 AutoModel + 手动 mask 池化，这里直接用 HF 的
   mean pooling + 手动取窗口切片，等价且更短
2. 原实现只支持 BGE-M3 且路径硬编码，这里走 model_hub 统一入口
3. 窗口边界对齐到句子边界，避免切断句子
4. 块尺寸硬约束：超 max_tokens 必切

代价：必须整篇进模型，长度受 max_length 限制（bge-small 是 512，
长文需滑窗，本实现对超长文做分段迟分）。
"""
import numpy as np
import torch

from splitter_base import BaseSplitter, Boundary, Segmentation
from model_hub import TextEncoder
from token_map import sentence_token_spans


class LateChunkingSplitter(BaseSplitter):
    name = "late_chunking"
    kind = "semantic"

    def __init__(self, window=256, max_tokens=512, max_length=8192,
                 sentence_align=True, model=None):
        self.window = window
        self.max_tokens = max_tokens
        self.max_length = max_length
        self.sentence_align = sentence_align
        self.model_name = model

    @torch.no_grad()
    def split(self, text):
        enc = TextEncoder.get(self.model_name) if self.model_name else TextEncoder.get()
        tok = enc.tok

        # 整篇编码，取 last_hidden_state
        # 注意：enc.model 是 SentenceTransformer 包装对象，直接 mdl(**inputs) 会命中
        # 其内部 Transformer 的 forward（签名是 forward(input)，不是 kwargs）。
        # 必须取 enc.model[0].auto_model（底层 HF 模型）才能用 kwargs 调用。
        mdl = enc.model[0].auto_model
        inputs = tok(text, return_tensors="pt", truncation=True,
                     max_length=self.max_length)
        inputs = {k: v.to(enc.device) for k, v in inputs.items()}
        out = mdl(**inputs)
        hs = out.last_hidden_state[0]            # [seq, dim]
        mask = inputs["attention_mask"][0].float()
        total = int(mask.sum().item()) - 2        # 去掉 CLS/SEP

        # 窗口边界
        if self.sentence_align:
            spans = sentence_token_spans(text, tok)
            sent_ends = [e for _, _, e in spans]
            cuts = []
            pos = self.window
            while pos < total:
                cand = [e for e in sent_ends if e <= pos]
                cuts.append(cand[-1] if cand else pos)
                pos += self.window
        else:
            cuts = list(range(self.window, total, self.window))

        # 超长强制切
        forced = []
        pos = 0
        for c in cuts:
            if c - pos > self.max_tokens:
                extra = list(range(pos + self.max_tokens, c, self.max_tokens))
                forced.extend(extra)
            pos = c
        cuts = sorted(set(cuts) | set(forced))

        # 每个窗口的均值池化（用 hidden states，池化后归一化）
        hs_norm = hs / (hs.norm(dim=-1, keepdim=True) + 1e-9)
        chunk_vecs = []
        prev = 0
        for c in cuts + [total]:
            seg = hs_norm[prev:c]
            if len(seg) == 0:
                prev = c
                continue
            v = seg.mean(dim=0)
            v = v / (v.norm() + 1e-9)
            chunk_vecs.append(v.cpu().numpy())
            prev = c

        self.chunk_embeddings = np.array(chunk_vecs) if chunk_vecs else None

        bounds = [Boundary(pos=c, reason="迟分窗口") for c in cuts if 0 < c < total]
        return Segmentation(algo=self.name, boundaries=bounds, n_tokens=total)
