# -*- coding: utf-8 -*-
"""
HiChunk Auto-Merge 检索侧复现（Hierarchical Retrieval）

HiChunk 本体是 LLM 分块器（vLLM + 分层断点输出），按已封版决策不复现。
本脚本只复现其检索侧 Auto-Merge（检索算法仓库 pipeline/retrieval/retrieval_algo.py），
层级用固定窗口构造（非 LLM 层级）：leaf = fixed_256，parent = 4 个 leaf 合并（~1024 tok）。

Auto-Merge 逻辑（忠实复现源码）：
1. leaf 向量检索 top-k
2. 逐个加入候选 passage，若 passage 已被已选 passage 覆盖则跳过
3. 若 passage 是其 parent 的唯一子块 → 直接用 parent
4. 沿祖先链尝试 merge，满足三条件则用 parent 替换子块：
   - condition_1: 已选子块字符数 >= threshold，threshold =
     (context_tokens/token_num)*parent_chars/3 + parent_chars/3
   - condition_2: 同 parent 的已选 passage >= 2
   - condition_3: parent 未覆盖部分 token 数 <= 剩余预算
5. 超 token 预算停止

评测口径（与 test_text_segmentation 既有框架分块表一致）：
- 章级召回 MRR（答案章任一 passage 首现排名）
- 证据覆盖率：最终 context 覆盖答案章 leaf 的比例
- context token 数、auto-merge 次数

输出 results/auto_merge_<ts>.json
"""
import sys, os, types, json, time
sys.path.insert(0, "scripts")

try:
    import transformers.onnx
except ModuleNotFoundError:
    _fake = types.ModuleType("transformers.onnx")
    class OnnxConfig: pass
    _fake.OnnxConfig = OnnxConfig
    sys.modules["transformers.onnx"] = _fake

import torch
import numpy as np
from datetime import datetime

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from model_hub import TextEncoder
from algo_fixed_window import FixedWindowSplitter

MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
QA_FILE = "corpus/pp_qa_25_final.jsonl"
PP_DIR = "corpus/pride_prejudice.jsonl"

# 评测参数
TOP_K = 8            # leaf 检索深度
PARENT_LEAVES = 4    # 每个 parent 合并的 leaf 数
TOKEN_NUM = 2048     # context token 预算


class Doc:
    def __init__(self, doc_id, text, title=""):
        self.doc_id = doc_id
        self.text = text
        self.title = title


def load_pp(path, n_docs=0):
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


def build_hierarchy(doc_text, enc, splitter):
    """leaf=splitter 切块；parent 每 PARENT_LEAVES 个 leaf 合并。"""
    seg = splitter.split(doc_text)
    spans = seg.chunk_token_spans()
    tok = enc.tok
    full = tok(doc_text, return_offsets_mapping=True, add_special_tokens=False)
    om = full["offset_mapping"]
    leaves = []
    for a, b in spans:
        if a >= len(om):
            continue
        cs = om[a][0] if a < len(om) else len(doc_text)
        ce = om[b - 1][1] if b - 1 < len(om) else len(doc_text)
        leaves.append({"text": doc_text[cs:ce],
                       "tokens": len(tok(doc_text[cs:ce])["input_ids"]),
                       "chars": len(doc_text[cs:ce])})
    return leaves


def make_passages(leaves):
    """构造层级：leaf passage（含 parent 指针）与 parent passage。"""
    passages = []
    # 先建 parent（按 leaf 分组）
    parents = []
    for i in range(0, len(leaves), PARENT_LEAVES):
        group = list(range(i, min(i + PARENT_LEAVES, len(leaves))))
        p = {"level": "parent",
             "left_index_idx": group[0], "right_index_idx": group[-1] + 1,
             "children": [], "parent": None}
        parents.append(p)
    # 再建 leaf，连 parent
    for i, leaf in enumerate(leaves):
        p_idx = i // PARENT_LEAVES
        p = parents[p_idx]
        leaf_p = {"level": "leaf", "leaf_idx": i, "text": leaf["text"],
                  "tokens": leaf["tokens"], "chars": leaf["chars"],
                  "left_index_idx": i, "right_index_idx": i + 1,
                  "parent": p, "children": []}
        p["children"].append(leaf_p)
        passages.append(leaf_p)
    for p in parents:
        passages.append(p)
    return passages


def eval_auto_merge(enc, docs, queries, leaf_texts, leaf_doc_ids,
                    leaf_vecs, leaves_p, doc_leaf_start, top_k=TOP_K, token_num=TOKEN_NUM):
    """对全部 leaf 建索引后逐查询跑 auto-merge，返回聚合指标。"""
    mrr, evidence_cov, ctx_tokens, merge_counts, hit_ranks = [], [], [], [], []
    leaf_chars = [len(t) for t in leaf_texts]
    for q in queries:
        ans_doc = q.get("answer_doc_id")
        qv = enc.encode([q["text"]])[0]
        qv = qv / (np.linalg.norm(qv) + 1e-9)
        sims = leaf_vecs @ qv
        order = np.argsort(-sims)

        # 取 top-k leaf（按文档分组建 passage 集合）
        top_leaves = [int(i) for i in order[:top_k]]

        # --- Auto-Merge（忠实复现源码逻辑）---
        retrieved = []  # passage 列表（含 doc_id）
        context_chars = 0

        def passage_chars(p):
            if p["level"] == "leaf":
                return leaf_chars[p["leaf_idx"]]
            return sum(leaf_chars[i] for i in range(p["left_index_idx"], p["right_index_idx"]))

        # 简化：用字符数近似 token 预算（nomic tok 与字符线性）
        merged = 0
        for li in top_leaves:
            doc_id = leaf_doc_ids[li]
            # 跳过已被覆盖
            if any(rp["left_index_idx"] <= li < rp["right_index_idx"]
                   for rp in retrieved if rp["doc_id"] == doc_id):
                continue
            local_li = li - doc_leaf_start[doc_id]
            passage = leaves_p[doc_id][local_li]
            # 唯一子块 → 直接用 parent
            if len(passage["parent"]["children"]) == 1:
                passage = passage["parent"]
            retrieved.append({"doc_id": doc_id, **passage})
            # 沿祖先链尝试 merge
            parent = passage["parent"]
            while parent is not None:
                same_parent = [rp for rp in retrieved
                               if rp.get("parent") is parent]
                child_chars = sum(passage_chars(rp) for rp in same_parent)
                parent_chars = sum(leaf_chars[i]
                                   for i in range(parent["left_index_idx"],
                                                  parent["right_index_idx"]))
                threshold = (context_chars / (token_num * 4)) * parent_chars / 3 + parent_chars / 3
                cond1 = child_chars >= threshold
                cond2 = len(same_parent) >= 2
                remain_budget = token_num * 4 - context_chars
                covered_chars = sum(passage_chars(rp) for rp in same_parent)
                cond3 = (parent_chars - covered_chars) <= remain_budget
                if cond1 and cond2 and cond3:
                    retrieved = [rp for rp in retrieved if rp not in same_parent]
                    retrieved.append({"doc_id": doc_id, **parent})
                    merged += 1
                    parent = parent["parent"]
                else:
                    break
            context_chars = sum(passage_chars(rp) for rp in retrieved)
            if context_chars >= token_num * 4:
                break

        # --- 章级召回 MRR（答案章首现 passage 的排名）---
        rank = None
        for r, rp in enumerate(retrieved):
            if rp["doc_id"] == ans_doc:
                rank = r + 1
                break
        if rank is None:
            mrr.append(0.0)
            hit_ranks.append(0)
        else:
            mrr.append(1.0 / rank)
            hit_ranks.append(rank)

        # --- 证据覆盖率：最终 context 覆盖答案章 leaf 的比例 ---
        ans_leaf_total = sum(1 for i in range(len(leaves_p[ans_doc]))
                             if leaf_doc_ids[i] == ans_doc) if ans_doc is not None else 0
        covered = set()
        for rp in retrieved:
            for i in range(rp["left_index_idx"], rp["right_index_idx"]):
                if leaf_doc_ids[i] == ans_doc:
                    covered.add(i)
        evidence_cov.append(len(covered) / max(ans_leaf_total, 1))
        ctx_tokens.append(context_chars // 4)  # 粗估 token
        merge_counts.append(merged)
    return {"MRR": float(np.mean(mrr)),
            "evidence_coverage": float(np.mean(evidence_cov)),
            "avg_context_tokens": float(np.mean(ctx_tokens)),
            "avg_merges": float(np.mean(merge_counts)),
            "hit_ranks": hit_ranks}


def main():
    enc = TextEncoder(MODEL_NAME, device=DEVICE)
    docs = load_pp(PP_DIR, n_docs=10)
    queries = [json.loads(l) for l in open(QA_FILE, encoding="utf-8") if l.strip()]
    print(f"model={enc.model_name} docs={len(docs)} queries={len(queries)}")

    splitter = FixedWindowSplitter(window=256)
    # 全局 leaf 索引
    leaf_texts, leaf_doc_ids = [], []
    leaves_p = {}
    doc_leaf_start = {}
    for doc in docs:
        doc_leaf_start[doc.doc_id] = len(leaf_texts)
        leaves = build_hierarchy(doc.text, enc, splitter)
        passages = make_passages(leaves)
        leaves_p[doc.doc_id] = [p for p in passages if p["level"] == "leaf"]
        for p in leaves_p[doc.doc_id]:
            leaf_texts.append(p["text"])
            leaf_doc_ids.append(doc.doc_id)
    print(f"leaf chunks={len(leaf_texts)}")

    t0 = time.time()
    leaf_vecs = enc.encode(leaf_texts)
    leaf_vecs = leaf_vecs / (np.linalg.norm(leaf_vecs, axis=1, keepdims=True) + 1e-9)
    print(f"encoding done {time.time()-t0:.1f}s")

    # baseline：无 merge 的 top-k leaf 检索
    mrr_b, r1_b = [], []
    for q in queries:
        qv = enc.encode([q["text"]])[0]
        qv = qv / (np.linalg.norm(qv) + 1e-9)
        sims = leaf_vecs @ qv
        order = np.argsort(-sims)
        rank = None
        for r, i in enumerate(order[:TOP_K]):
            if leaf_doc_ids[int(i)] == q.get("answer_doc_id"):
                rank = r + 1
                break
        if rank is None:
            mrr_b.append(0.0); r1_b.append(0.0)
        else:
            mrr_b.append(1.0 / rank)
            r1_b.append(1.0 if rank == 1 else 0.0)

    result = eval_auto_merge(enc, docs, queries, leaf_texts, leaf_doc_ids, leaf_vecs, leaves_p, doc_leaf_start)

    out = {
        "timestamp": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "model": MODEL_NAME,
        "algo": "auto_merge_f256x4_top8",
        "params": {"top_k": TOP_K, "parent_leaves": PARENT_LEAVES,
                   "token_budget": TOKEN_NUM},
        "baseline_topk": {"MRR": float(np.mean(mrr_b)), "R@1": float(np.mean(r1_b))},
        "auto_merge": result,
        "n_queries": len(queries),
        "n_leaf_chunks": len(leaf_texts),
    }
    os.makedirs("results", exist_ok=True)
    path = f"results/auto_merge_{out['timestamp']}.json"
    json.dump(out, open(path, "w"), ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "algo"}, indent=2, ensure_ascii=False)[:1500])
    print("->", path)


if __name__ == "__main__":
    main()
