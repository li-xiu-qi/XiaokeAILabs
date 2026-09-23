# -*- coding: utf-8 -*-
"""
全量评测驱动：八种算法 × 300 篇 Wikipedia

输出：
  results/seg_<algo>_<model>_<timestamp>.json   逐算法原始数据
  results/summary_<timestamp>.json              汇总表

评测轴：边界 F1（容忍窗 ±32/±64/±128 token）、Pk、WindowDiff、
       块尺寸分布、单篇耗时。检索轴（Recall@k）由 verify_retrieval.py 单独跑。
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from corpus_loader import load_corpus
from metrics import boundary_prf, pk_score, windowdiff, size_stats

from algo_fixed_window import FixedWindowSplitter
from algo_semantic_breakpoint import SemanticSplitter
from algo_structural import StructuralSplitter
from algo_late_chunking import LateChunkingSplitter
from algo_texttiling import TextTilingSplitter
from algo_c99 import C99Splitter
from algo_bm25_boundary import BM25Splitter
from algo_llm_topic import LLMTopicSplitter
from algo_llm_proposition import LLMPropositionSplitter
from algo_recursive_char import RecursiveCharSplitter
from algo_sentence_window import SentenceWindowSplitter
from algo_chonkie_semantic import ChonkieSemanticSplitter
from algo_dtc import DTCSplitter

TOLS = [32, 64, 128]
PK_K = 32
MODEL_LONG = "BAAI/bge-m3"      # 迟分需长上下文，bge-small-zh 的 512 上限会截断长文


def gold_boundaries(doc, tok):
    """把 ground truth 段落边界映射到 token 空间。

    doc.text 的段落用 \\n\\n 分隔，逐个段落 tokenize 累加，
    去掉最后一段的尾部位置。
    """
    paras = [p for p in doc.text.split("\n\n") if p.strip()]
    pos = 0
    out = []
    for p in paras[:-1]:
        pos += len(tok(p, add_special_tokens=False)["input_ids"])
        out.append(pos)
    return out


def build_algos(model):
    return [
        ("fixed_128", FixedWindowSplitter(window=128)),
        ("fixed_256", FixedWindowSplitter(window=256)),
        ("fixed_512", FixedWindowSplitter(window=512)),
        ("fixed_256_sb", FixedWindowSplitter(window=256, sentence_boundary=True)),
        ("semantic_t050", SemanticSplitter(threshold=0.50, max_tokens=512)),
        ("semantic_t060", SemanticSplitter(threshold=0.60, max_tokens=512)),
        ("semantic_p10", SemanticSplitter(percentile=10, max_tokens=512)),
        ("semantic_p25", SemanticSplitter(percentile=25, max_tokens=512)),
        # structural 不在本 benchmark：纯文本无结构标记，算法退化为段落累积，
        # 且逐行正则扫描在长文上复杂度爆炸。它的正确定位是 Markdown/代码仓库。
        ("late_chunking_256", LateChunkingSplitter(window=256, max_length=8192,
                                                   model=MODEL_LONG)),
        ("texttiling_w40", TextTilingSplitter(w=40, smooth=2)),
        ("texttiling_w80", TextTilingSplitter(w=80, smooth=2, max_boundaries=12)),
        ("c99", C99Splitter(target_chunks=8)),
        ("bm25_p15", BM25Splitter(percentile=15)),
        ("bm25_p25", BM25Splitter(percentile=25)),
        # 框架分块三件套（LangChain/LlamaIndex/Chonkie 复现）
        ("recursive_char_256", RecursiveCharSplitter(chunk_tokens=256)),
        ("recursive_char_512", RecursiveCharSplitter(chunk_tokens=512)),
        ("sentence_window_w1", SentenceWindowSplitter(window_size=1)),
        ("sentence_window_w2", SentenceWindowSplitter(window_size=2)),
        ("chonkie_semantic", ChonkieSemanticSplitter(similarity_window=3,
                                                      filter_window=5, threshold=0.8)),
        # DFC（Dynamic Token Size Chunking），复现 2603.06976 Table 2。
        # 论文只给 [Kmin, Kmax] 区间未给选择机制，本实现按 CDAC 密度
        # 反比定目标块大小（解释性实现，口径见 algo_dtc.py docstring）。
        ("dtc_50_200", DTCSplitter(k_min=50, k_max=200, density_window=3)),
    ]


def eval_one(name, splitter, docs, tok):
    rec = {"algo": name, "kind": splitter.kind, "n_docs": len(docs)}
    per_doc = []
    t0 = time.perf_counter()
    for d in docs:
        seg = splitter.split(d.text)
        gold = gold_boundaries(d, tok)
        pred = [b.pos for b in seg.boundaries]
        spans = seg.chunk_token_spans()
        sizes = [e - s for s, e in spans]
        row = {"doc_id": d.doc_id, "n_tokens": seg.n_tokens,
               "n_gold": len(gold), "n_pred": len(pred),
               "sizes": sizes}
        for tol in TOLS:
            p, r, f = boundary_prf(pred, gold, tol)
            row[f"f1_{tol}"] = f
        row["pk"] = pk_score(pred, gold, seg.n_tokens, PK_K)
        row["wd"] = windowdiff(pred, gold, seg.n_tokens, PK_K)
        per_doc.append(row)
    rec["seconds"] = time.perf_counter() - t0

    for tol in TOLS:
        key = f"f1_{tol}"
        rec[f"f1_{tol}"] = float(np.mean([r[key] for r in per_doc]))
    rec["pk"] = float(np.mean([r["pk"] for r in per_doc]))
    rec["wd"] = float(np.mean([r["wd"] for r in per_doc]))
    all_sizes = np.concatenate([np.array(r["sizes"]) for r in per_doc if r["sizes"]])
    rec["size"] = {"mean": float(all_sizes.mean()), "std": float(all_sizes.std()),
                   "p50": float(np.percentile(all_sizes, 50)),
                   "p95": float(np.percentile(all_sizes, 95)),
                   "max": int(all_sizes.max()),
                   "n_chunks": int(len(all_sizes)),
                   "over_512": float((all_sizes > 512).mean())}
    rec["per_doc"] = per_doc
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-docs", type=int, default=300)
    ap.add_argument("--data-dir", default="corpus")
    ap.add_argument("--cache", default="corpus/wiki_eval_300.jsonl")
    ap.add_argument("--model", default=None, help="留空用 MODEL_ZH 默认值")
    ap.add_argument("--only", default=None, help="逗号分隔的算法名过滤")
    ap.add_argument("--skip-llm", action="store_true")
    args = ap.parse_args()

    from model_hub import TextEncoder, MODEL_ZH
    model = args.model or MODEL_ZH
    enc = TextEncoder.get(model)
    print(f"model={enc.model_name} dim={enc.dim} device={enc.device}")

    docs = load_corpus(args.data_dir, n_docs=args.n_docs, cache_path=args.cache)
    print(f"docs={len(docs)}")

    algos = build_algos(args.model)
    if args.only:
        keep = set(args.only.split(","))
        algos = [a for a in algos if a[0] in keep]
    if not args.skip_llm:
        algos.append(("llm_pairwise", LLMTopicSplitter(mode="pairwise", max_tokens=512)))
        algos.append(("llm_segment", LLMTopicSplitter(mode="segment", max_tokens=512)))
        algos.append(("llm_proposition",
                      LLMPropositionSplitter(max_tokens=512, chunk_word_cap=500)))

    os.makedirs("results", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    summary = []

    for name, sp in algos:
        print(f"\n--- {name} ({sp.kind}) ---", flush=True)
        try:
            rec = eval_one(name, sp, docs, enc.tok)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            summary.append({"algo": name, "error": str(e)})
            continue
        path = f"results/seg_{name}_{ts}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False)
        row = {k: v for k, v in rec.items() if k != "per_doc"}
        summary.append(row)
        print(f"  F1@32={rec['f1_32']:.3f} F1@64={rec['f1_64']:.3f} "
              f"F1@128={rec['f1_128']:.3f} Pk={rec['pk']:.3f} "
              f"chunks={rec['size']['n_chunks']} mean={rec['size']['mean']:.0f} "
              f"max={rec['size']['max']} {rec['seconds']:.1f}s", flush=True)

    with open(f"results/summary_{ts}.json", "w", encoding="utf-8") as f:
        json.dump({"timestamp": ts, "model": enc.model_name, "n_docs": len(docs),
                   "tols": TOLS, "results": summary}, f, ensure_ascii=False, indent=2)
    print(f"\nsummary -> results/summary_{ts}.json")


if __name__ == "__main__":
    main()
