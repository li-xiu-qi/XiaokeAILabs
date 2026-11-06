"""Fix BEIR SciFact candidate pools and audit the data before training."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import re
import unicodedata


def digest(path):
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def load_jsonl(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    by_id = {row["_id"]: row for row in rows}
    if len(rows) != len(by_id):
        raise ValueError(f"Duplicate ids in {path.name}")
    return by_id


def load_qrels(path):
    qrels = defaultdict(dict)
    with path.open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source, delimiter="\t"):
            qid, did, grade = row["query-id"], row["corpus-id"], int(row["score"])
            if grade != 1 or did in qrels[qid]:
                raise ValueError(f"Unexpected qrel: {qid}, {did}, {grade}")
            qrels[qid][did] = grade
    return qrels


def terms(text):
    return re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", text).casefold())


def write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def candidate_pools(docs, queries, config):
    postings = defaultdict(list)
    lengths = {}
    for did, row in docs.items():
        words = terms(f"{row['title']} {row['text']}")
        lengths[did] = len(words)
        for word, frequency in Counter(words).items():
            postings[word].append((did, frequency))
    average = sum(lengths.values()) / len(docs)
    k1, b = config["bm25_k1"], config["bm25_b"]
    norm = {did: k1 * (1 - b + b * n / average) for did, n in lengths.items()}
    weights = {}
    for word, pairs in postings.items():
        idf = math.log1p((len(docs) - len(pairs) + 0.5) / (len(pairs) + 0.5))
        weights[word] = [
            (did, idf * tf * (k1 + 1) / (tf + norm[did])) for did, tf in pairs
        ]
    pools = {}
    for qid, row in queries.items():
        scores = defaultdict(float)
        for word in sorted(set(terms(row["text"]))):
            for did, score in weights.get(word, ()):
                scores[did] += score
        ranked = sorted(scores, key=lambda did: (-scores[did], did))
        pools[qid] = [[did, scores[did]] for did in ranked[:config["pool_size"]]]
    return pools


def cross_duplicates(source, heldout, tokens, config):
    result = []
    for qid in source:
        left = tokens[qid]
        for oid in heldout:
            right = tokens[oid]
            similarity = len(left & right) / len(left | right)
            if similarity >= config["query_jaccard_cutoff"]:
                result.append([qid, oid, round(similarity, 5)])
    return result


def load_original_evidence(path, benchmark_qrels, queries, docs):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    evidence = {str(row["id"]): row for row in rows}
    if len(evidence) != len(rows) or evidence.keys() != benchmark_qrels.keys():
        raise ValueError("AllenAI training query ids do not match BEIR")
    evidence_docs = {}
    for qid, row in evidence.items():
        if row["claim"].strip() != queries[qid]["text"].strip():
            raise ValueError(f"Original claim differs from benchmark text: {qid}")
        evidence_docs[qid] = {
            did for did, rationales in row["evidence"].items()
            if any(r["label"] in {"SUPPORT", "CONTRADICT"} and r["sentences"] for r in rationales)
        }
        if evidence_docs[qid] - docs.keys():
            raise ValueError(f"Original evidence points outside the corpus: {qid}")
    return evidence_docs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--original-claims", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    docs = load_jsonl(args.data / "corpus.jsonl")
    queries = load_jsonl(args.data / "queries.jsonl")
    original_train = load_qrels(args.data / "qrels/train.tsv")
    test = load_qrels(args.data / "qrels/test.tsv")
    if original_train.keys() & test.keys() or original_train.keys() | test.keys() != queries.keys():
        raise ValueError("Query ids do not form a disjoint labeled partition")
    if len(docs) != 5183 or len(original_train) != 809 or len(test) != 300:
        raise ValueError("Unexpected benchmark size")
    if any(not row["text"].strip() for row in docs.values()):
        raise ValueError("Empty document")
    if any(did not in docs for qrels in (original_train, test) for rel in qrels.values() for did in rel):
        raise ValueError("Qrels point outside the corpus")

    evidence_docs = load_original_evidence(args.original_claims, original_train, queries, docs)
    tokens = {qid: set(terms(row["text"])) for qid, row in queries.items()}
    if any(not q for q in tokens.values()):
        raise ValueError("Empty query")
    initial_order = sorted(
        original_train, key=lambda qid: hashlib.sha256(f"{config['seed']}:{qid}".encode()).hexdigest()
    )
    dev_ids = initial_order[:config["dev_queries"]]
    train_ids = initial_order[config["dev_queries"]:]
    dev_test_near = cross_duplicates(dev_ids, test, tokens, config)
    dev_ids = [qid for qid in dev_ids if qid not in {p[0] for p in dev_test_near}]
    train_heldout_near = cross_duplicates(train_ids, [*dev_ids, *test], tokens, config)
    train_ids = [qid for qid in train_ids if qid not in {p[0] for p in train_heldout_near}]
    remaining = cross_duplicates(train_ids, [*dev_ids, *test], tokens, config)
    remaining += cross_duplicates(dev_ids, test, tokens, config)
    if remaining:
        raise ValueError(f"Found {len(remaining)} remaining cross-split near duplicates")
    pools = candidate_pools(docs, queries, config)
    if any(len(pool) != config["pool_size"] for pool in pools.values()):
        raise ValueError("A query has insufficient BM25 candidates")
    doc_tokens = {did: tuple(terms(row["text"])) for did, row in docs.items()}
    duplicate_abstracts = len(docs) - len(set(doc_tokens.values()))
    train = {}
    excluded_no_evidence = []
    excluded_pairs = 0
    for qid in train_ids:
        benchmark_relevant = original_train[qid].keys()
        relevant = benchmark_relevant & evidence_docs[qid]
        excluded_pairs += len(benchmark_relevant - relevant)
        if not relevant:
            excluded_no_evidence.append(qid)
            continue
        known_relevant = benchmark_relevant | evidence_docs[qid]
        positive_texts = {doc_tokens[did] for did in known_relevant}
        unjudged = [
            did for did, _ in pools[qid]
            if did not in known_relevant and doc_tokens[did] not in positive_texts
        ][:config["hard_negatives"]]
        if len(unjudged) != config["hard_negatives"]:
            raise ValueError(f"Insufficient negatives for {qid}")
        train[qid] = {"relevant": sorted(relevant), "unjudged": unjudged}
    partitions = {
        split: {qid: {"relevant": sorted(qrels[qid]), "pool": pools[qid]} for qid in ids}
        for split, ids, qrels in (("dev", dev_ids, original_train), ("test", sorted(test), test))
    }
    args.out.mkdir(parents=True, exist_ok=True)
    pools_file = args.out / "pools.json"
    write_json(pools_file, {"train": train, **partitions})

    train_positive = {did for row in train.values() for did in row["relevant"]}
    train_seen = train_positive | {did for row in train.values() for did in row["unjudged"]}
    heldout_positive = {
        split: {did for row in group.values() for did in row["relevant"]}
        for split, group in partitions.items()
    }
    audit = {
        "source_sha256": {
            path: digest(args.data / path)
            for path in ("corpus.jsonl", "queries.jsonl", "qrels/train.tsv", "qrels/test.tsv")
        },
        "original_claims_train_sha256": digest(args.original_claims),
        "config_sha256": digest(args.config),
        "pools_sha256": digest(pools_file),
        "corpus_documents": len(docs),
        "original_labeled_queries": {"train": len(original_train), "test": len(test)},
        "retained_queries": {"train": len(train), **{split: len(group) for split, group in partitions.items()}},
        "relevance_grades": [1],
        "duplicate_normalized_abstracts": duplicate_abstracts,
        "max_query_token_jaccard_exclusive": config["query_jaccard_cutoff"],
        "cross_split_query_near_matches_remaining": len(remaining),
        "excluded_query_ids": {
            "dev_matches_test": dev_test_near,
            "train_matches_dev_or_test": train_heldout_near,
        },
        "training_evidence_filter": {
            "matched_original_claim_ids_and_text": len(evidence_docs),
            "post_dedup_queries": len(train_ids),
            "excluded_queries_without_overlapping_evidence_ids": excluded_no_evidence,
            "removed_beir_positive_pairs_without_original_rationales": excluded_pairs,
            "retained_positive_pairs": sum(len(row["relevant"]) for row in train.values()),
        },
        "supervised_train_positive_document_ids": len(train_positive),
        "heldout_positive_doc_ids_seen_as_train_positive": {
            split: len(ids & train_positive) for split, ids in heldout_positive.items()
        },
        "heldout_positive_doc_ids_in_training_pair_pool": {
            split: len(ids & train_seen) for split, ids in heldout_positive.items()
        },
        "train_negative_candidates_per_query": config["hard_negatives"],
    }
    write_json(args.out / "data-audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
