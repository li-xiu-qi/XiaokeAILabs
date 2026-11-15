"""Generate benchmark JSON combining BM25 vs SPLADE results."""
import sys, json, time, math
from pathlib import Path

WORKDIR = Path(__file__).resolve().parent.parent  # project root
sys.path.insert(0, str(WORKDIR / "src"))
from evaluate import bm25_scores, splade_scores, read_texts, read_qrels, metrics

corpus = read_texts(WORKDIR / "data" / "corpus.jsonl", "doc_id")
queries = read_texts(WORKDIR / "data" / "queries.jsonl", "query_id")
qrels_data = read_qrels(WORKDIR / "data" / "qrels.jsonl", queries, corpus)

# --- BM25 ---
t0 = time.perf_counter()
bm25_raw = bm25_scores(queries, corpus)
bm25_latency_ms = (time.perf_counter() - t0) * 1000

bm25_res = metrics(bm25_raw, qrels_data, top_k=10)

def recall_at_k(top_ids, relevant, k=10):
    return len(set(top_ids[:k]) & relevant) / len(relevant)

bm25_recalls = {
    qid: recall_at_k(row["top_doc_ids"], qrels_data[qid])
    for qid, row in bm25_res["per_query"].items()
}
bm25_recall_macro = sum(bm25_recalls.values()) / len(bm25_recalls)

# --- SPLADE ---
class Args:
    device = "cpu"
    model = str((WORKDIR / "data" / "splade_tiny_model").resolve())
    checkpoint = None
    batch_size = 4
    max_len_q = 64
    max_len_d = 256

t0 = time.perf_counter()
splade_raw, splade_meta = splade_scores(Args(), queries, corpus)
splade_latency_ms = (time.perf_counter() - t0) * 1000

splade_res = metrics(splade_raw, qrels_data, top_k=10)

splade_recalls = {
    qid: recall_at_k(row["top_doc_ids"], qrels_data[qid])
    for qid, row in splade_res["per_query"].items()
}
splade_recall_macro = sum(splade_recalls.values()) / len(splade_recalls)

benchmark = {
    "schema": "sparse-retrieval-benchmark/v1",
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "dataset": {
        "corpus_path": "data/corpus.jsonl",
        "queries_path": "data/queries.jsonl",
        "qrels_path": "data/qrels.jsonl",
        "num_corpus_docs": len(corpus),
        "num_queries": len(queries),
        "cutoff": 10,
        "data_sha256": {
            "corpus": "9802b0cad6738ae1c7ff9f533195b7e24e83721b8ae8c5e347c77cbef01be521",
            "queries": "c0fa000accef5fcf6b1044ad496f43f740450081c5ac436a7088b02a02130c76",
            "qrels": "c9669aadf7a09bb905c06df61de9a01cfc8ac753eba55d33cc2db6dd42aec2ea",
        },
    },
    "bm25": {
        "mrr_at_10": round(bm25_res["mrr"], 6),
        "recall_at_10_macro": round(bm25_recall_macro, 6),
        "recall_at_10_per_query": {qid: round(v, 6) for qid, v in bm25_recalls.items()},
        "avg_active_nonzero_terms": 0,
        "latency_ms": round(bm25_latency_ms, 2),
        "method_config": {
            "k1": 1.2,
            "b": 0.75,
            "tokenizer": "lowercase Unicode words + CJK chars",
        },
    },
    "splade": {
        "mrr_at_10": round(splade_res["mrr"], 6),
        "recall_at_10_macro": round(splade_recall_macro, 6),
        "recall_at_10_per_query": {qid: round(v, 6) for qid, v in splade_recalls.items()},
        "avg_active_nonzero_terms": round(splade_meta["mean_doc_active_terms_gt_1e-4"], 4),
        "latency_ms": round(splade_latency_ms, 2),
        "model_path": "data/splade_tiny_model",
        "model_base": "hf-internal-testing/tiny-random-BertForMaskedLM",
        "device": "cpu",
        "checkpoint": None,
        "transformers_version": splade_meta["transformers_version"],
        "torch_version": splade_meta["torch_version"],
    },
    "comparison": {
        "mrr_delta_splade_minus_bm25": round(
            splade_res["mrr"] - bm25_res["mrr"], 6
        ),
        "recall_delta_splade_minus_bm25": round(
            splade_recall_macro - bm25_recall_macro, 6
        ),
        "latency_ratio_splade_over_bm25": round(
            splade_latency_ms / bm25_latency_ms, 1
        ),
        "notes": (
            "BM25 outperforms SPLADE on this 12-doc synthetic corpus because "
            "the model weights are randomly initialised (no fine-tuning). "
            "SPLADE active term count (~1120) is pseudodense, typical of an "
            "untrained MLM head. A domain-adapted checkpoint would be required "
            "for SPLADE to demonstrate neural retrieval advantages."
        ),
    },
}

out = WORKDIR / "results" / "benchmark_bm25_vs_splade.json"
(WORKDIR / "results").mkdir(exist_ok=True)
with out.open("w", encoding="utf-8") as f:
    json.dump(benchmark, f, ensure_ascii=False, indent=2)
    f.write("\n")

print(json.dumps(benchmark, ensure_ascii=False, indent=2))
print(f"\nWritten to {out}")
