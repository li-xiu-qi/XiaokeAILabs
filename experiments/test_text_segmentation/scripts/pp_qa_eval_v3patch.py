# -*- coding: utf-8 -*-
"""
Pride and Prejudice 长文档迟分检索评测 v2

改进:
1. 25 条干净查询（查询不含答案关键词）
2. 分桶统计: late赢/traditional赢/打平
3. 逐查询排名输出
4. 新增: 查询含答案关键词的对照查询（验证"词匹配"效应）
"""
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import transformers.onnx  # noqa: F401
except ModuleNotFoundError:
    import types as _types
    _fake = _types.ModuleType("transformers.onnx")
    class OnnxConfig:
        pass
    _fake.OnnxConfig = OnnxConfig
    sys.modules["transformers.onnx"] = _fake

import transformers.pytorch_utils as _putils
if not hasattr(_putils, "find_pruneable_heads_and_indices"):
    def find_pruneable_heads_and_indices(heads, n_heads, head_size,
                                         already_pruned_heads):
        return (set(heads) - set(already_pruned_heads),
                list(range(n_heads * head_size)))
    _putils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices

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


from model_hub import TextEncoder


@torch.no_grad()
def _doc_hidden(enc, text, max_length=8192):
    tok, mdl = enc.tok, enc.model[0].auto_model
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {k: v.to(enc.device) for k, v in inputs.items()}
    hs = mdl(**inputs).last_hidden_state[0]
    return (hs.float()) / (hs.norm(dim=-1, keepdim=True) + 1e-9)


def late_encode_chunks(enc, text, spans, max_length=8192):
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


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nomic-ai/nomic-embed-text-v1.5")
    ap.add_argument("--qa-file", default="corpus/pp_qa_25_final.jsonl")
    args = ap.parse_args()

    # 加载 chunks
    with open("corpus/pp_chunks_10.json", encoding="utf-8") as f:
        docs = json.load(f)

    all_chunk_texts = []
    all_chunk_meta = []
    doc_spans = {}
    for d in docs:
        doc_id = d["doc_id"]
        texts = [c["text"] for c in d["chunks"]]
        spans = [c["token_span"] for c in d["chunks"]]
        all_chunk_texts.extend(texts)
        for c in d["chunks"]:
            all_chunk_meta.append((doc_id, c["chunk_id"]))
        doc_spans[doc_id] = spans

    total_chunks = len(all_chunk_texts)
    print(f"total chunks: {total_chunks}")

    # 加载问答对
    with open(args.qa_file, encoding="utf-8") as f:
        queries = [json.loads(l) for l in f]
    print(f"queries: {len(queries)}")

    # 找到每个查询的答案 chunk 全局索引
    query_texts = []
    answer_indices = []
    valid_queries = []
    for q in queries:
        target = (q["answer_doc_id"], q["answer_chunk_id"])
        found = None
        for ci, meta in enumerate(all_chunk_meta):
            if meta == target:
                found = ci
                break
        if found is not None:
            query_texts.append(q["text"])
            answer_indices.append(found)
            valid_queries.append(q)
        else:
            print(f"  WARNING: answer chunk not found for qid={q['qid']} target={target}")

    print(f"valid queries: {len(valid_queries)}")

    enc = TextEncoder.get(args.model)
    print(f"model={enc.model_name} dim={enc.dim}")

    # 加载原始文档全文
    with open("corpus/pride_prejudice.jsonl", encoding="utf-8") as f:
        raw_docs = [json.loads(l) for l in f]
    raw_texts = {}
    for d in raw_docs[:10]:
        raw_texts[d["doc_id"]] = d["text"]

    # Traditional encoding
    print("Traditional encoding...")
    trad_vecs = enc.encode(all_chunk_texts)
    trad_vecs = trad_vecs / (np.linalg.norm(trad_vecs, axis=1, keepdims=True) + 1e-9)

    # Late encoding
    print("Late chunking encoding...")
    late_vecs_list = []
    for d in docs:
        doc_id = d["doc_id"]
        text = raw_texts[doc_id]
        spans = doc_spans[doc_id]
        vecs = late_encode_chunks(enc, text, spans)
        late_vecs_list.append(vecs)
    late_vecs = np.vstack(late_vecs_list)
    late_vecs = late_vecs / (np.linalg.norm(late_vecs, axis=1, keepdims=True) + 1e-9)
    print(f"  late_vecs shape: {late_vecs.shape}")

    # 编码查询
    print("Encoding queries...")
    query_vecs = enc.encode(query_texts)
    query_vecs = query_vecs / (np.linalg.norm(query_vecs, axis=1, keepdims=True) + 1e-9)

    # 评测
    results = {"traditional": {"r1": [], "r5": [], "mrr": []},
               "late": {"r1": [], "r5": [], "mrr": []}}
    per_query = []

    for qi, qv in enumerate(query_vecs):
        qid = valid_queries[qi]["qid"]
        qtext = valid_queries[qi]["text"]
        ans_idx = answer_indices[qi]
        ans_doc = valid_queries[qi]["answer_doc_id"]
        kw = valid_queries[qi]["answer_keyword"]

        trad_sims = trad_vecs @ qv
        late_sims = late_vecs @ qv
        trad_order = np.argsort(-trad_sims)
        late_order = np.argsort(-late_sims)

        trad_rank = int(np.where(trad_order == ans_idx)[0][0]) + 1
        late_rank = int(np.where(late_order == ans_idx)[0][0]) + 1

        results["traditional"]["r1"].append(1.0 if trad_rank == 1 else 0.0)
        results["traditional"]["r5"].append(1.0 if trad_rank <= 5 else 0.0)
        results["traditional"]["mrr"].append(1.0 / trad_rank)
        results["late"]["r1"].append(1.0 if late_rank == 1 else 0.0)
        results["late"]["r5"].append(1.0 if late_rank <= 5 else 0.0)
        results["late"]["mrr"].append(1.0 / late_rank)

        per_query.append({
            "qid": qid, "text": qtext[:70], "ans_doc": ans_doc,
            "kw": kw, "trad_rank": trad_rank, "late_rank": late_rank,
            "delta": trad_rank - late_rank,
        })

    # 打印逐查询结果
    print("\n=== Per-query ranking ===")
    print(f"{'qid':>4} {'trad':>6} {'late':>6} {'delta':>6}  kw / text")
    for pq in per_query:
        marker = "W" if pq["delta"] > 0 else ("L" if pq["delta"] < 0 else "=")
        print(f"{pq['qid']:4d} {pq['trad_rank']:6d} {pq['late_rank']:6d} "
              f"{pq['delta']:+6d}  {marker} kw={pq['kw']} | {pq['text']}")

    # 分桶
    late_win = [pq for pq in per_query if pq["delta"] > 0]
    trad_win = [pq for pq in per_query if pq["delta"] < 0]
    tie = [pq for pq in per_query if pq["delta"] == 0]

    print(f"\n=== Buckets ===")
    print(f"  Late wins: {len(late_win)}")
    print(f"  Traditional wins: {len(trad_win)}")
    print(f"  Tie: {len(tie)}")
    if late_win:
        print(f"\n  Late wins details:")
        for pq in late_win:
            print(f"    q{pq['qid']}: T={pq['trad_rank']} L={pq['late_rank']} kw={pq['kw']}")
    if trad_win:
        print(f"\n  Traditional wins details:")
        for pq in trad_win:
            print(f"    q{pq['qid']}: T={pq['trad_rank']} L={pq['late_rank']} kw={pq['kw']}")

    # 总指标
    print(f"\n=== Overall ===")
    for mode in ["traditional", "late"]:
        r = results[mode]
        print(f"  {mode:12s} R@1={np.mean(r['r1']):.3f} R@5={np.mean(r['r5']):.3f} "
              f"MRR={np.mean(r['mrr']):.3f} (n={len(valid_queries)})")

    # 保存
    os.makedirs("results", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    result = {
        "timestamp": ts,
        "model": enc.model_name,
        "dataset": "Pride and Prejudice (10 chapters)",
        "total_chunks": total_chunks,
        "n_queries": len(valid_queries),
        "qa_file": args.qa_file,
        "buckets": {"late_win": len(late_win), "trad_win": len(trad_win), "tie": len(tie)},
        "traditional": {"R@1": float(np.mean(results["traditional"]["r1"])),
                        "R@5": float(np.mean(results["traditional"]["r5"])),
                        "MRR": float(np.mean(results["traditional"]["mrr"]))},
        "late": {"R@1": float(np.mean(results["late"]["r1"])),
                 "R@5": float(np.mean(results["late"]["r5"])),
                 "MRR": float(np.mean(results["late"]["mrr"]))},
        "per_query": per_query,
    }
    path = f"results/pp_qa_v2_{ts}.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
