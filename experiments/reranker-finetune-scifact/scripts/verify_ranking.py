"""Recalculate the released SciFact ranking results from candidate scores."""

import argparse
import json
import math
from pathlib import Path
from statistics import mean


DEFAULT_EVIDENCE = Path(__file__).resolve().parents[1] / "docs" / "evidence"


def ranking_metrics(ranking, relevant):
    discounts = [1 / math.log2(rank + 2) for rank in range(10)]
    hits = [rank for rank, doc_id in enumerate(ranking[:10]) if doc_id in relevant]
    return {
        "ndcg_at_10": sum(discounts[rank] for rank in hits) / sum(discounts[:len(relevant)]),
        "mrr_at_10": 1 / (hits[0] + 1) if hits else 0,
    }


def check(actual, expected, label):
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(f"{label}: recomputed {actual}, recorded {expected}")


def verify(evidence):
    summary = json.loads((evidence / "summary.json").read_text(encoding="utf-8"))
    splits = json.loads((evidence / "split-query-ids.json").read_text(encoding="utf-8"))
    split_sets = {}
    for name in ("train", "dev", "test"):
        ids = splits[name]
        if not isinstance(ids, list) or any(not isinstance(qid, str) or not qid for qid in ids):
            raise ValueError(f"Invalid {name} split query ids")
        split_sets[name] = set(ids)
        if len(ids) != len(split_sets[name]):
            raise ValueError(f"Duplicate query id in {name} split")
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        if split_sets[left] & split_sets[right]:
            raise ValueError(f"Query ids overlap between {left} and {right} splits")
    with (evidence / "training-pairs.jsonl").open(encoding="utf-8") as source:
        training_pairs = [json.loads(line) for line in source]
    training_queries = {row["query_id"] for row in training_pairs}
    positive_docs = {row["positive_doc_id"] for row in training_pairs}
    training_docs = positive_docs | {row["unjudged_doc_id"] for row in training_pairs}
    training = summary["training"]
    if (len(training_pairs) != training["triplets"]
            or len(training_queries) != training["distinct_queries_used"]
            or len(positive_docs) != training["distinct_positive_doc_ids"]
            or not training_queries <= set(splits["train"])):
        raise ValueError("Training pairs do not match the summary or training split")
    overall = summary["evaluation"]["all"]
    methods = {"bm25": None, "initial": "before", "finetuned": "after"}
    totals = {method: [] for method in methods}
    recalls = []
    query_ids = set()
    with (evidence / "per-query.jsonl").open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            qid = row["query_id"]
            candidates = row["candidates_in_bm25_order"]
            candidate_ids = {doc["doc_id"] for doc in candidates}
            relevant = set(row["relevant_doc_ids"])
            if (set(row["relevant_doc_ids_seen_as_training_positive"]) != relevant & positive_docs
                    or set(row["relevant_doc_ids_seen_in_training_pairs"]) != relevant & training_docs):
                raise ValueError(f"Training overlap differs for query {qid}")
            if qid in query_ids or not relevant:
                raise ValueError(f"Invalid query {qid}: duplicate ID or missing relevance labels")
            if len(candidate_ids) != len(candidates) or len(candidates) != summary["config"]["pool_size"]:
                raise ValueError(f"Invalid candidate pool for query {qid}")
            query_ids.add(qid)
            recalls.append(len(relevant & candidate_ids) / len(relevant))

            for method, score_key in methods.items():
                ordered = sorted(candidates, key=lambda doc: -doc[score_key]) if score_key else candidates
                metrics = ranking_metrics([doc["doc_id"] for doc in ordered], relevant)
                for name, value in metrics.items():
                    check(value, row["metrics"][method][name], f"{qid}/{method}/{name}")
                totals[method].append(metrics)

    if summary["evaluation_split"] == "beir_scifact_test" and query_ids != set(splits["test"]):
        raise ValueError("Evaluation queries do not match the recorded test split")
    if summary["evaluation_split"] == "dev_subset" and not query_ids <= set(splits["dev"]):
        raise ValueError("Evaluation queries are not in the recorded development split")
    if len(query_ids) != overall["queries"]:
        raise ValueError(f"Expected {overall['queries']} queries, found {len(query_ids)}")
    check(mean(recalls), overall["candidate_recall"], "Recall at candidate cutoff")
    check(mean(recall > 0 for recall in recalls), overall["candidate_hit_rate"], "Hit at candidate cutoff")
    print(f"Verified {len(query_ids)} queries")
    for method, rows in totals.items():
        metrics = {name: mean(row[name] for row in rows) for name in rows[0]}
        for name, value in metrics.items():
            check(value, overall["rankings"][method][name], f"{method}/{name}")
        print(f"{method:9s} nDCG@10={metrics['ndcg_at_10']:.4f} MRR@10={metrics['mrr_at_10']:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path, nargs="?", default=DEFAULT_EVIDENCE)
    verify(parser.parse_args().evidence)


if __name__ == "__main__":
    main()
