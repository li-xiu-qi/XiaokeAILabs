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
