# -*- coding: utf-8 -*-
"""
DTC + 框架分块方案复现：检索 MRR 评测

在 Pride and Prejudice 10 章上，用新复现的框架分块方案（recursive_char /
sentence_window / chonkie_semantic）现场切块，traditional 独立编码，
与固定 25 条查询对算 MRR，对照 fixed_256 基线。

评测口径：文档级召回。每条查询标注 ans_doc（答案所在章），
该章内任一 chunk 进入 top-k 即算命中。与 fixed_256 基线口径一致。

输出 results/dtcs_retrieval_<ts>.json
"""
import sys, os, types, json, time
sys.path.insert(0, "scripts")

# --- 补桩（复用现有补丁）---
try:
    import transformers.onnx
except ModuleNotFoundError:
    _fake = types.ModuleType("transformers.onnx")
    class OnnxConfig: pass
    _fake.OnnxConfig = OnnxConfig
    sys.modules["transformers.onnx"] = _fake

import transformers.pytorch_utils as _putils
if not hasattr(_putils, "find_pruneable_heads_and_indices"):
    def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
        return (set(heads) - set(already_pruned_heads), list(range(n_heads * head_size)))
    _putils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices

import transformers.configuration_utils as _cu
def _patched_getattr(self, key):
    if key == "add_cross_attention": return False
    if key == "chunk_size_feed_forward": return 0
    if key == "is_decoder": return False
    if key == "cross_attention_hidden_size": return None
    raise AttributeError(key)
_cu.PretrainedConfig.__getattr__ = _patched_getattr

import torch
import numpy as np
from datetime import datetime

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from model_hub import TextEncoder
from algo_recursive_char import RecursiveCharSplitter
from algo_sentence_window import SentenceWindowSplitter
from algo_chonkie_semantic import ChonkieSemanticSplitter
from algo_dtc import DTCSplitter
from algo_fixed_window import FixedWindowSplitter

MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
QA_FILE = "corpus/pp_qa_25_final.jsonl"
PP_DIR = "corpus/pride_prejudice.jsonl"


def chunk_doc_with_splitter(splitter, text, enc):
    """用 splitter 切块，返回 chunk 文本列表。"""
    seg = splitter.split(text)
    spans = seg.chunk_token_spans()
    # 从 token span 取文本：用 offset mapping
    tok = enc.tok
    full = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    om = full["offset_mapping"]
    chunks = []
    for a, b in spans:
        if a >= len(om):
            continue
        char_start = om[a][0] if a < len(om) else len(text)
        char_end = om[b - 1][1] if b - 1 < len(om) else len(text)
        chunks.append(text[char_start:char_end])
    return chunks


def eval_algo(name, splitter, docs, enc, qa_file):
    """对新算法切块做检索评测。"""
    all_texts, all_doc_ids = [], []
    for doc_id, doc in enumerate(docs):
        chunks = chunk_doc_with_splitter(splitter, doc.text, enc)
        for c in chunks:
            all_texts.append(c)
            all_doc_ids.append(doc_id)
    if not all_texts:
        return None
    chunk_vecs = enc.encode(all_texts)
    chunk_vecs = chunk_vecs / (np.linalg.norm(chunk_vecs, axis=1, keepdims=True) + 1e-9)
    doc_arr = np.array(all_doc_ids)

    # 查询
    queries = [json.loads(l) for l in open(qa_file, encoding="utf-8") if l.strip()]
    q_texts = [q["text"] for q in queries]
    q_vecs = enc.encode(q_texts)
    q_vecs = q_vecs / (np.linalg.norm(q_vecs, axis=1, keepdims=True) + 1e-9)

    mrr, r1, r5 = [], [], []
    for qi, q in enumerate(queries):
        ans_doc = q.get("answer_doc_id")
        sims = chunk_vecs @ q_vecs[qi]
        order = np.argsort(-sims)
        ranked_docs = [int(doc_arr[i]) for i in order]
        # 找答案章第一次出现的排名（文档级召回）
        rank = None
        for r, d in enumerate(ranked_docs):
            if d == ans_doc:
                rank = r + 1
                break
        if rank is None:
            mrr.append(0.0); r1.append(0.0); r5.append(0.0)
        else:
            mrr.append(1.0 / rank)
            r1.append(1.0 if rank == 1 else 0.0)
            r5.append(1.0 if rank <= 5 else 0.0)
    return {
        "algo": name, "kind": splitter.kind,
        "n_chunks": len(all_texts),
        "avg_chunks_per_doc": len(all_texts) / len(docs),
        "R@1": float(np.mean(r1)), "R@5": float(np.mean(r5)),
        "MRR": float(np.mean(mrr)),
    }


class Doc:
    """极简文档对象，与 load_corpus 返回的兼容（有 .text / .doc_id）。"""
    def __init__(self, doc_id, text, title=""):
        self.doc_id = doc_id
        self.text = text
        self.title = title


def load_pp(path, n_docs=0):
    """读 jsonl 语料。n_docs=0 读全部，>0 只取前 N 个 doc。

    P&P 原始 jsonl 有 59 段，查询对 answer_doc_id 只在 0-9，需 --n-docs 10 截断；
    红楼梦（20 章）/JE（10 章）answer_doc_id 覆盖全部，用默认 0 读全部。
    """
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            docs.append(Doc(d.get("doc_id", len(docs)), d["text"], d.get("title", "")))
            if n_docs > 0 and len(docs) >= n_docs:
                break
    return docs


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=PP_DIR, help="语料 jsonl 路径")
    ap.add_argument("--qa", default=QA_FILE, help="查询对 jsonl 路径")
    ap.add_argument("--n-docs", type=int, default=0, help="只取前 N 个 doc（0=全部）")
    ap.add_argument("--only", default="", help="只跑指定算法，逗号分隔")
    args = ap.parse_args()

    enc = TextEncoder(MODEL_NAME, device=DEVICE)
    docs = load_pp(args.corpus, n_docs=args.n_docs)
    print(f"model={enc.model_name} docs={len(docs)} corpus={args.corpus}")

    algos = [
        ("fixed_256_baseline", FixedWindowSplitter(window=256)),
        ("recursive_char_256", RecursiveCharSplitter(chunk_tokens=256)),
        ("recursive_char_512", RecursiveCharSplitter(chunk_tokens=512)),
        ("sentence_window_w1", SentenceWindowSplitter(window_size=1)),
        ("sentence_window_w2", SentenceWindowSplitter(window_size=2)),
        ("chonkie_semantic", ChonkieSemanticSplitter(similarity_window=3, filter_window=5, threshold=0.8)),
        ("dtc_50_200", DTCSplitter(k_min=50, k_max=200, density_window=3)),
    ]

    os.makedirs("results", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    results = []
    if args.only:
        keep = set(args.only.split(","))
        algos = [a for a in algos if a[0] in keep]

    for name, sp in algos:
        t0 = time.time()
        r = eval_algo(name, sp, docs, enc, args.qa)
        dt = time.time() - t0
        if r:
            r["seconds"] = round(dt, 1)
            results.append(r)
            print(f"  {name:22s} MRR={r['MRR']:.4f} R@1={r['R@1']:.3f} "
                  f"chunks={r['n_chunks']} ({r['avg_chunks_per_doc']:.1f}/doc) {dt:.1f}s", flush=True)

    import os as _os
    corpus_name = _os.path.splitext(_os.path.basename(args.corpus))[0]
    n_queries = sum(1 for l in open(args.qa, encoding="utf-8") if l.strip())
    out = {"timestamp": ts, "model": enc.model_name,
           "dataset": corpus_name,
           "corpus": args.corpus, "qa": args.qa,
           "n_docs": len(docs), "n_queries": n_queries, "results": results}
    path = f"results/dtcs_retrieval_{corpus_name}_{ts}.json"
    json.dump(out, open(path, "w"), ensure_ascii=False, indent=2)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
