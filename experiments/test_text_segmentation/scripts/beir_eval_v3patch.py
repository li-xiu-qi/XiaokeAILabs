# -*- coding: utf-8 -*-
"""
BEIR scifact 迟分检索评测

用标准 BEIR 格式（corpus + queries + qrels）评测 late chunking。
文档级评分：每个文档取其最高分 chunk 的相似度。
指标：nDCG@10、Recall@10。

设计要点（区别于 verify_retrieval.py）：
- 查询 = BEIR 真实查询（不是自动构造的子串），消除自匹配偏差
- 答案 = qrels 标注的相关文档，不是同段另一块
- 这是论文的评测方式，能给出可对照论文数字的基准
"""
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 复用 jina/transformers 5.x 桩
try:
    import transformers.onnx  # noqa: F401
except ModuleNotFoundError:
    import types as _types
    _fake = _types.ModuleType("transformers.onnx")
    class OnnxConfig:  # noqa
        pass
    _fake.OnnxConfig = OnnxConfig
    sys.modules["transformers.onnx"] = _fake

import transformers.configuration_utils as _cu
def _patched_getattr(self, key):
    if key == "add_cross_attention": return False
    if key == "chunk_size_feed_forward": return 0
    if key == "is_decoder": return False
    if key == "cross_attention_hidden_size": return None
    raise AttributeError(key)
_cu.PretrainedConfig.__getattr__ = _patched_getattr

import transformers.modeling_rope_utils as _rope
if "default" not in _rope.ROPE_INIT_FUNCTIONS:
    def _compute_default_rope_parameters(config=None, device=None, seq_len=None, **kwargs):
        import torch
        base = getattr(config, "rope_theta", 10000.0)
        partial = getattr(config, "partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        dim = int(head_dim * partial)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
        return inv_freq, 1.0
    _rope.ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters

import transformers.modeling_utils as _mu
_orig_get_tied = _mu.PreTrainedModel.get_expanded_tied_weights_keys
def _patched_get_tied(self, all_submodels=False):
    for cls in type(self).__mro__:
        if hasattr(cls, "_tied_weights_keys") and isinstance(cls._tied_weights_keys, list):
            cls._tied_weights_keys = {k: k for k in cls._tied_weights_keys}
            break
    return _orig_get_tied(self, all_submodels)
_mu.PreTrainedModel.get_expanded_tied_weights_keys = _patched_get_tied

_orig_missing = _mu.PreTrainedModel._move_missing_keys_from_meta_to_device
def _patched_missing(self, *args, **kwargs):
    if not hasattr(self, "all_tied_weights_keys"):
        tw = getattr(type(self), "_tied_weights_keys", [])
        if isinstance(tw, list):
            tw = {k: k for k in tw}
        self.all_tied_weights_keys = tw if isinstance(tw, dict) else {}
    return _orig_missing(self, *args, **kwargs)
_mu.PreTrainedModel._move_missing_keys_from_meta_to_device = _patched_missing


import transformers.pytorch_utils as _putils
if not hasattr(_putils, "find_pruneable_heads_and_indices"):
    def find_pruneable_heads_and_indices(heads, n_heads, head_size,
                                         already_pruned_heads):
        return (set(heads) - set(already_pruned_heads),
                list(range(n_heads * head_size)))
    _putils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices

from model_hub import TextEncoder

DATA_DIR = "beir_data/scifact"
CHUNK_TOKENS = 256


def load_beir(data_dir):
    """加载 BEIR 格式数据。"""
    # corpus
    corpus = {}
    with open(os.path.join(data_dir, "corpus.jsonl"), encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            text = (d.get("title", "") + " " + d.get("text", "")).strip()
            corpus[d["_id"]] = text
    # queries
    queries = {}
    with open(os.path.join(data_dir, "queries.jsonl"), encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            queries[d["_id"]] = d["text"]
    # qrels (test split)
    qrels = defaultdict(dict)
    qrels_path = os.path.join(data_dir, "qrels", "test.tsv")
    with open(qrels_path, encoding="utf-8") as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                qid, did, score = parts[0], parts[1], int(parts[2])
                if score > 0:
                    qrels[qid][did] = score
    return corpus, queries, qrels


def chunk_document(enc, text, chunk_tokens=CHUNK_TOKENS):
    """把文档切成固定 token 数的 chunks，返回 (chunk_texts, token_spans)。

    token_spans 是 [(start, end), ...]，在 hs 坐标系下（含 CLS 偏移 +1）。
    """
    tok = enc.tok
    full = tok(text, return_offsets_mapping=True, add_special_tokens=False,
               truncation=True, max_length=8192)
    ids = full["input_ids"]
    om = full["offset_mapping"]
    n = len(ids)

    chunk_texts = []
    spans = []
    for start in range(0, n, chunk_tokens):
        end = min(start + chunk_tokens, n)
        # 用 offset_mapping 找到对应的字符区间
        char_start = om[start][0]
        char_end = om[end - 1][1]
        chunk_texts.append(text[char_start:char_end])
        spans.append((start + 1, end + 1))  # +1 for CLS
    return chunk_texts, spans


@torch.no_grad()
def _doc_hidden(enc, text, max_length=8192):
    tok, mdl = enc.tok, enc.model[0].auto_model
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {k: v.to(enc.device) for k, v in inputs.items()}
    hs = mdl(**inputs).last_hidden_state[0].float()
    return hs / (hs.norm(dim=-1, keepdim=True) + 1e-9)


def late_encode_chunks(enc, text, spans, max_length=8192):
    """Late chunking: 整个文档过 model，按 spans 切 mean pool。"""
    hs = _doc_hidden(enc, text, max_length)
    out = []
    for a, b in spans:
        seg = hs[a:b]
        if len(seg) > 0:
            v = seg.mean(dim=0)
            out.append((v / (v.norm() + 1e-9)).cpu().numpy())
        else:
            out.append(np.zeros(enc.dim, dtype=np.float32))
    return np.array(out, dtype=np.float32)


def traditional_encode_chunks(enc, chunk_texts):
    """Traditional: 每个 chunk 独立 encode。"""
    return enc.encode(chunk_texts)


def ndcg_at_k(ranked_doc_ids, qrels_for_query, k=10):
    """计算 nDCG@k。ranked_doc_ids 是按分数从高到低排列的文档 ID 列表。"""
    dcg = 0.0
    for i, did in enumerate(ranked_doc_ids[:k]):
        rel = qrels_for_query.get(did, 0)
        dcg += rel / math.log2(i + 2)  # i+2 because i is 0-based

    # IDCG: 理想排序
    ideal_rels = sorted(qrels_for_query.values(), reverse=True)[:k]
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_rels))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked_doc_ids, qrels_for_query, k=10):
    """计算 Recall@k。"""
    relevant = set(did for did, s in qrels_for_query.items() if s > 0)
    if not relevant:
        return 0.0
    retrieved = set(ranked_doc_ids[:k])
    return len(retrieved & relevant) / len(relevant)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nomic-ai/nomic-embed-text-v1.5")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--chunk-tokens", type=int, default=CHUNK_TOKENS)
    ap.add_argument("--only-traditional", action="store_true")
    args = ap.parse_args()

    print(f"加载数据: {args.data_dir}")
    corpus, queries, qrels = load_beir(args.data_dir)
    print(f"  corpus={len(corpus)} queries={len(queries)} test_qrels={len(qrels)}")

    # 只保留 test split 里有标注的查询
    test_qids = [qid for qid in qrels if qid in queries]
    print(f"  test queries with qrels: {len(test_qids)}")

    # 需要的文档：所有在 qrels 中出现的文档
    needed_docs = set()
    for qid in test_qids:
        needed_docs.update(qrels[qid].keys())
    needed_docs = sorted(needed_docs)
    print(f"  needed docs (in qrels): {len(needed_docs)}")

    enc = TextEncoder.get(args.model)
    print(f"model={enc.model_name} dim={enc.dim}")

    # 对所有需要的文档切 chunks + encode
    print(f"切 chunks (chunk_tokens={args.chunk_tokens})...")
    doc_chunk_texts = {}   # doc_id -> [chunk_texts]
    doc_spans = {}         # doc_id -> [token_spans]
    doc_chunk_to_idx = {}  # doc_id -> 该文档第一个 chunk 在全局矩阵中的起始位置

    all_chunk_texts = []   # 全局 chunk 文本列表（traditional 用）
    all_chunk_doc_ids = []  # 每个 chunk 属于哪个文档
    for did in needed_docs:
        text = corpus[did]
        ctexts, spans = chunk_document(enc, text, args.chunk_tokens)
        doc_chunk_texts[did] = ctexts
        doc_spans[did] = spans
        doc_chunk_to_idx[did] = len(all_chunk_texts)
        all_chunk_texts.extend(ctexts)
        all_chunk_doc_ids.extend([did] * len(ctexts))

    total_chunks = len(all_chunk_texts)
    print(f"  total chunks: {total_chunks}, avg chunks/doc: {total_chunks/len(needed_docs):.1f}")

    # Traditional encoding
    print("Traditional encoding...")
    trad_vecs = traditional_encode_chunks(enc, all_chunk_texts)
    trad_vecs = trad_vecs / (np.linalg.norm(trad_vecs, axis=1, keepdims=True) + 1e-9)
    print(f"  trad_vecs shape: {trad_vecs.shape}")

    # Late encoding
    if not args.only_traditional:
        print("Late chunking encoding...")
        late_vecs_list = []
        for i, did in enumerate(needed_docs):
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(needed_docs)}")
            vecs = late_encode_chunks(enc, corpus[did], doc_spans[did])
            late_vecs_list.append(vecs)
        late_vecs = np.vstack(late_vecs_list)
        late_vecs = late_vecs / (np.linalg.norm(late_vecs, axis=1, keepdims=True) + 1e-9)
        print(f"  late_vecs shape: {late_vecs.shape}")

    # 对每个 test query，检索并计算指标
    print("检索 + 评测...")
    query_vecs = enc.encode([queries[qid] for qid in test_qids])
    query_vecs = query_vecs / (np.linalg.norm(query_vecs, axis=1, keepdims=True) + 1e-9)

    # 构建 chunk → doc 的映射
    chunk_doc_arr = np.array(all_chunk_doc_ids)
    needed_docs_arr = np.array(needed_docs)
    doc_id_to_pos = {did: i for i, did in enumerate(needed_docs)}

    mode_scores = {}
    for mode_name, chunk_vecs in [("traditional", trad_vecs)] + (
        [("late", late_vecs)] if not args.only_traditional else []):
        # 计算 query × chunk 相似度矩阵
        sims = query_vecs @ chunk_vecs.T  # (n_queries, n_chunks)

        ndcgs, recalls = [], []
        for qi, qid in enumerate(test_qids):
            qrels_q = qrels[qid]
            # 对每个文档，取其最高 chunk 分数
            doc_scores = {}
            for ci in range(total_chunks):
                did = chunk_doc_arr[ci]
                s = float(sims[qi, ci])
                if did not in doc_scores or s > doc_scores[did]:
                    doc_scores[did] = s

            # 排序
            ranked = sorted(doc_scores.keys(), key=lambda d: -doc_scores[d])
            ndcgs.append(ndcg_at_k(ranked, qrels_q, k=10))
            recalls.append(recall_at_k(ranked, qrels_q, k=10))

        mode_scores[mode_name] = {
            "nDCG@10": float(np.mean(ndcgs)),
            "Recall@10": float(np.mean(recalls)),
        }
        print(f"  {mode_name:12s} nDCG@10={np.mean(ndcgs):.4f}  "
              f"Recall@10={np.mean(recalls):.4f}  (n={len(test_qids)})")

    # 保存结果
    os.makedirs("results", exist_ok=True)
    from datetime import datetime
    result = {
        "timestamp": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "model": enc.model_name,
        "dataset": "beir/" + args.data_dir.split("/")[-1],
        "chunk_tokens": args.chunk_tokens,
        "n_docs": len(needed_docs),
        "n_chunks": total_chunks,
        "n_queries": len(test_qids),
        "scores": mode_scores,
    }
    ds_prefix = "beir_" + args.data_dir.split("/")[-1]
    path = f"results/{ds_prefix}_{result['timestamp']}.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
