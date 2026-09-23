# -*- coding: utf-8 -*-
"""
检索可用性评测：分割质量的终极判据

边界 F1 衡量「切得准不准」，但它不回答「切出来的块好不好用」。
这一层直接测检索：用段落真值边界把全文切成 K 个块，
把块编码成向量后做检索，看答案块排第几。

## 查询怎么构造（关键，前两个版本在这里测错了）

**错误做法一**：查询 = 目标块的前 60 字符。块与自己的子串必然最相似，
R@1 恒为 1.000，测的是自匹配而非检索能力。

**错误做法二（留一法）**：查询 = 目标块首句，编码时把该句从块里删掉。
这个设计对迟分不利——传统编码的块向量主要由字面词项决定，删一句影响小；
迟分的块向量强依赖全文上下文，删一句后该块失去定位锚点。实测迟分 R@1=0.615
低于传统 1.000，与迟分原始论文结论方向相反，是评测设计造成的假象。

**现行做法**：查询 = 同篇**另一段**的首句（非目标块），
两种编码面对完全相同的查询与相同的块边界，只在「块向量怎么算」上有差异。
这是控制变量的正确形态。

## 长度分桶

bge-m3 上限 8192 token，长文整篇过模型会越界。
按文档长度分桶：短文档走整篇迟分，长文档滑窗迟分（窗口重叠一句），
避免切片越界把整篇静默跳过。
"""
import argparse
import json
import os
import re
import sys
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from corpus_loader import load_corpus
from model_hub import TextEncoder
import torch

SENT_SPLIT = re.compile(r"(?<=[。！？!?；;])\s*|\n+")


def split_sentences(text):
    return [p.strip() for p in SENT_SPLIT.split(text) if p.strip()]


def gold_chunks(doc, tok):
    """真值段落边界 → 块文本列表。"""
    paras = [p for p in doc.text.split("\n\n") if p.strip()]
    return paras


def chunk_token_offsets(doc, tok):
    """每段的 (起始token, 结束token)，含 CLS 偏移。

    关键：chunk_token_offsets 输出的是「不含 special tokens」的累加位置（0 起），
    而 late_chunking_encode 切的是 hs（含 CLS+SEP，hs[0] 是 CLS）。
    两者差 1 位，直接切会导致所有块左偏 1 token 且首块丢首 token。
    所以这里统一 +1 对齐到 hs 的坐标系。
    """
    paras = [p for p in doc.text.split("\n\n") if p.strip()]
    pos, out = 1, []          # 跳过 CLS
    for p in paras:
        n = len(tok(p, add_special_tokens=False)["input_ids"])
        out.append((pos, pos + n))
        pos += n
    return out


def splitter_chunk_texts(doc, tok, splitter):
    """用指定分块器切文本，返回块文本列表。

    与 gold_chunks 的区别：边界来自算法而非真值段落。
    块文本用句子级 token span 反查切出（精确），不用字符比例近似。
    """
    from token_map import sentence_token_spans
    seg = splitter.split(doc.text)
    spans = sentence_token_spans(doc.text, tok)
    cuts = sorted({0, seg.n_tokens} | {b.pos for b in seg.boundaries
                                      if 0 < b.pos < seg.n_tokens})
    out = []
    for i in range(len(cuts) - 1):
        a, b = cuts[i], cuts[i + 1]
        sents = [s for s, sa, sb in spans if sb > a and sa < b]
        t = "".join(sents).strip()
        if t:
            out.append(t)
    return out


def late_chunking_encode_spans(enc, text, spans, max_length=8192, window=4096):
    """迟分编码：整篇过 encoder，按给定 (start,end) token 区间切隐状态均值池化。

    spans 已是含 CLS 偏移的坐标系，直接用于切 hs。
    超长文档滑窗，窗口间重叠一段以保证跨窗边界不断裂。
    """
    total = enc.n_tokens(text)

    if total <= max_length:
        hs_n = _doc_hidden(enc, text, max_length)
        segs = {0: hs_n}
        ranges = [(0, total)]
    else:
        # 滑窗：每窗 max_length，重叠 window 的一半
        step = window // 2
        segs, ranges = {}, []
        start = 0
        while start < total:
            end = min(start + window, total)
            hs_n = _doc_hidden(enc, text, end)
            segs[start] = hs_n[start:end] if len(hs_n) >= end else hs_n[start:]
            ranges.append((start, end))
            if end >= total:
                break
            start += step

    out = []
    for a, b in spans:
        vecs = []
        for w_start, w_end in ranges:
            lo, hi = max(a, w_start), min(b, w_end)
            if hi > lo:
                seg = segs[w_start][lo - w_start:hi - w_start]
                if len(seg) > 0:
                    vecs.append(seg)
        if not vecs:
            out.append(np.zeros(enc.dim, dtype=np.float32))
            continue
        v = torch.cat(vecs, dim=0).mean(dim=0)
        out.append((v / (v.norm() + 1e-9)).cpu().numpy())
    return np.array(out, dtype=np.float32)


def splitter_spans_in_token(doc, tok, splitter):
    """分块器边界 → 含 CLS 偏移的 token 区间列表。"""
    from token_map import sentence_token_spans
    seg = splitter.split(doc.text)
    sspans = sentence_token_spans(doc.text, tok)
    cuts = sorted({0, seg.n_tokens} | {b.pos for b in seg.boundaries
                                      if 0 < b.pos < seg.n_tokens})
    out = []
    for i in range(len(cuts) - 1):
        a, b = cuts[i], cuts[i + 1]
        sents = [s for s, sa, sb in sspans if sb > a and sa < b]
        if sents:
            out.append((a + 1, b + 1))     # +1 对齐 hs 坐标系
    return out


def traditional_encode(enc, texts):
    """传统编码：每块独立过 encoder。"""
    return enc.encode(texts)


@torch.no_grad()
def _doc_hidden(enc, text, max_length=8192):
    """整篇过 encoder，返回 [seq, dim] 的 last_hidden_state（L2 归一化）。

    注意：enc.model 是 SentenceTransformer 包装对象，直接 mdl(**inputs) 会命中
    其内部 Transformer 的 forward（签名是 forward(input)，不是 kwargs）。
    必须取 enc.model[0]（Transformer 模块）才能用 kwargs 调用。
    """
    tok = enc.tok
    mdl = enc.model[0].auto_model      # 底层 HF 模型
    inputs = tok(text, return_tensors="pt", truncation=True,
                 max_length=max_length)
    inputs = {k: v.to(enc.device) for k, v in inputs.items()}
    out = mdl(**inputs)
    hs = out.last_hidden_state[0]
    return hs / (hs.norm(dim=-1, keepdim=True) + 1e-9)


def late_chunking_encode(enc, text, spans, max_length=8192, window=4096):
    """迟分编码：整篇过 encoder，按段边界切隐状态做均值池化。

    超长文档滑窗，窗口间重叠一段以保证跨窗边界不断裂。
    """
    total = enc.n_tokens(text)

    if total <= max_length:
        hs_n = _doc_hidden(enc, text, max_length)
        segs = {0: hs_n}
        ranges = [(0, total)]
    else:
        # 滑窗：每窗 max_length，重叠 window 的一半
        step = window // 2
        segs, ranges = {}, []
        start = 0
        while start < total:
            end = min(start + window, total)
            hs_n = _doc_hidden(enc, text, end)
            segs[start] = hs_n[start:end] if len(hs_n) >= end else hs_n[start:]
            ranges.append((start, end))
            if end >= total:
                break
            start += step

    out = []
    for a, b in spans:
        vecs = []
        for w_start, w_end in ranges:
            lo, hi = max(a, w_start), min(b, w_end)
            if hi > lo:
                seg = segs[w_start][lo - w_start:hi - w_start]
                if len(seg) > 0:
                    vecs.append(seg)
        if not vecs:
            out.append(np.zeros(enc.dim, dtype=np.float32))
            continue
        v = torch.cat(vecs, dim=0).mean(dim=0)
        out.append((v / (v.norm() + 1e-9)).cpu().numpy())
    return np.array(out, dtype=np.float32)


def evaluate(name, mode, docs, enc, seed=42, max_queries_per_doc=2,
             splitter=None, splitter_name=""):
    """检索评测。

    splitter=None 时用真值段落边界（原行为）。
    传入 splitter 时用该分块器的边界，用于回答「这个分块器切出来的块
    检索效果如何」——边界 F1 测的是切得准不准，这一层测的是好不好用。
    迟分编码需要 token 级区间，走 splitter_spans_in_token；传统编码只需块文本。
    """
    rng = np.random.default_rng(seed)
    r1, r5, mrr = [], [], []
    n_q = 0
    skipped = 0
    n_chunks_all = []
    for d in docs:
        if splitter is None:
            chunks = gold_chunks(d, enc.tok)
            spans = chunk_token_offsets(d, enc.tok)
        else:
            chunks = splitter_chunk_texts(d, enc.tok, splitter)
            spans = splitter_spans_in_token(d, enc.tok, splitter)
        if len(chunks) < 4:
            skipped += 1
            continue
        n_chunks_all.append(len(chunks))

        # 块向量：两种编码在同一组块边界上算
        if mode == "traditional":
            base_vecs = traditional_encode(enc, chunks)
        else:
            try:
                base_vecs = late_chunking_encode_spans(enc, d.text, spans)
            except Exception:
                skipped += 1
                continue

        norms = np.linalg.norm(base_vecs, axis=1, keepdims=True) + 1e-9
        base_unit = base_vecs / norms

        for _ in range(min(max_queries_per_doc, len(chunks) - 1)):
            # 目标块与查询块必须不同，避免自匹配
            qi = int(rng.integers(len(chunks)))
            q_sents = split_sentences(chunks[qi])
            if not q_sents:
                continue
            qv_block = enc.encode([q_sents[0]])[0]
            qv = qv_block / (np.linalg.norm(qv_block) + 1e-9)

            sims = base_unit @ qv
            order = np.argsort(-sims)
            rank = int(np.where(order == qi)[0][0]) + 1
            r1.append(1.0 if rank == 1 else 0.0)
            r5.append(1.0 if rank <= 5 else 0.0)
            mrr.append(1.0 / rank)
            n_q += 1

    return {"algo": name, "n_queries": n_q, "skipped_docs": skipped,
            "recall@1": float(np.mean(r1)) if r1 else 0.0,
            "recall@5": float(np.mean(r5)) if r5 else 0.0,
            "mrr": float(np.mean(mrr)) if mrr else 0.0,
            "splitter": splitter_name,
            "mean_chunks": float(np.mean(n_chunks_all)) if n_chunks_all else 0.0}


def build_splitter(name):
    """按名字造分块器。"""
    from algo_semantic_breakpoint import SemanticSplitter
    from algo_fixed_window import FixedWindowSplitter
    from algo_llm_proposition import LLMPropositionSplitter
    from algo_llm_topic import LLMTopicSplitter
    table = {
        "semantic_p10": lambda: SemanticSplitter(percentile=10, max_tokens=512),
        "semantic_t060": lambda: SemanticSplitter(threshold=0.60, max_tokens=512),
        "fixed_256": lambda: FixedWindowSplitter(window=256),
        "fixed_256_sb": lambda: FixedWindowSplitter(window=256, sentence_boundary=True),
        "llm_proposition": lambda: LLMPropositionSplitter(max_tokens=512,
                                                          chunk_word_cap=500,
                                                          concurrency=16),
        "llm_pairwise": lambda: LLMTopicSplitter(mode="pairwise", max_tokens=512),
    }
    if name not in table:
        raise SystemExit(f"未知 splitter: {name}，可选: {', '.join(table)}")
    return table[name]()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-docs", type=int, default=50)
    ap.add_argument("--model", default="BAAI/bge-m3")
    ap.add_argument("--splitter", default=None,
                    help="用指定分块器的边界而非真值段落边界，可选: "
                         "semantic_p10, semantic_t060, fixed_256, fixed_256_sb, "
                         "llm_proposition, llm_pairwise")
    ap.add_argument("--only-traditional", action="store_true",
                    help="只跑传统编码（跳过迟分，省时间）")
    args = ap.parse_args()

    enc = TextEncoder.get(args.model)
    docs = load_corpus("corpus", n_docs=args.n_docs,
                       cache_path="corpus/wiki_eval_300.jsonl")
    print(f"model={enc.model_name} dim={enc.dim} docs={len(docs)}")

    if args.splitter:
        sp = build_splitter(args.splitter)
        modes = [("traditional", "traditional")]
        if not args.only_traditional:
            modes.append(("late", "late"))
        out = []
        for mode, tag in modes:
            nm = f"{args.splitter}+{tag}"
            out.append(evaluate(nm, mode, docs, enc, splitter=sp,
                                splitter_name=args.splitter))
    else:
        out = [evaluate("traditional_encode", "traditional", docs, enc),
               evaluate("late_chunking_encode", "late", docs, enc)]

    os.makedirs("results", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = f"results/retrieval_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"timestamp": ts, "model": enc.model_name,
                   "n_docs": len(docs),
                   "design": "查询=同篇另一块首句，非目标块自身（避免自匹配）",
                   "results": out}, f, ensure_ascii=False, indent=2)
    for r in out:
        print(f"  {r['algo']:34s} R@1={r['recall@1']:.3f} "
              f"R@5={r['recall@5']:.3f} MRR={r['mrr']:.3f} "
              f"chunks={r['mean_chunks']:.1f} "
              f"n={r['n_queries']} skip={r['skipped_docs']}")
    print(f"-> {path}")


if __name__ == "__main__":
    main()
