"""Evaluate BM25 or a SPLADE checkpoint on the same labeled candidate corpus."""

import argparse
import hashlib
import json
import math
import platform
import re
from collections import Counter
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_texts(path, id_field):
    records = {}
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            key, value = obj.get(id_field), obj.get("text")
            if not isinstance(key, str) or not key.strip() or key in records:
                raise ValueError(f"{path}:{number}: missing or duplicate {id_field}")
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{path}:{number}: text must be a nonempty string")
            records[key] = value
    if not records:
        raise ValueError(f"{path}: no records")
    return records


def read_qrels(path, queries, documents):
    relevant = {key: set() for key in queries}
    seen = set()
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            qid, did, label = obj.get("query_id"), obj.get("doc_id"), obj.get("relevance")
            if qid not in queries or did not in documents:
                raise ValueError(f"{path}:{number}: unknown query_id or doc_id")
            if (qid, did) in seen or not isinstance(label, (int, float)) or isinstance(label, bool):
                raise ValueError(f"{path}:{number}: duplicate pair or nonnumeric relevance")
            if not math.isfinite(label) or label < 0:
                raise ValueError(f"{path}:{number}: relevance must be finite and nonnegative")
            seen.add((qid, did))
            if label > 0:
                relevant[qid].add(did)
    missing = [key for key, values in relevant.items() if not values]
    if missing:
        raise ValueError(f"each query requires at least one positive qrel; missing: {missing[:5]}")
    return relevant


def terms(text):
    # CJK ideographs are characters; other Unicode word sequences are lowercased words.
    return re.findall(r"[\u3400-\u9fff]|[^\W_\u3400-\u9fff]+", text.lower(), flags=re.UNICODE)


def bm25_scores(queries, documents, k1=1.2, b=0.75):
    counts = {did: Counter(terms(text)) for did, text in documents.items()}
    lengths = {did: sum(tokens.values()) for did, tokens in counts.items()}
    average_length = sum(lengths.values()) / len(lengths)
    if not average_length:
        raise ValueError("BM25 corpus has no tokens")
    frequencies = Counter(token for tokens in counts.values() for token in tokens)
    n = len(documents)
    result = {}
    for qid, text in queries.items():
        query_terms = Counter(terms(text))
        if not query_terms:
            raise ValueError(f"query {qid} has no BM25 tokens")
        scores = {}
        for did, tokens in counts.items():
            score = 0.0
            for token, query_frequency in query_terms.items():
                frequency = tokens[token]
                if frequency:
                    idf = math.log1p((n - frequencies[token] + 0.5) / (frequencies[token] + 0.5))
                    norm = k1 * (1 - b + b * lengths[did] / average_length)
                    score += query_frequency * idf * frequency * (k1 + 1) / (frequency + norm)
            scores[did] = score
        result[qid] = scores
    return result


def metrics(scores, relevant, top_k=10):
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    per_query = {}
    for qid, items in scores.items():
        if any(not math.isfinite(value) for value in items.values()):
            raise ValueError(f"nonfinite score for query {qid}")
        ranked = sorted(
            (did for did, score in items.items() if score > 0), key=lambda did: (-items[did], did)
        )[:top_k]
        rank = next((index for index, did in enumerate(ranked, 1) if did in relevant[qid]), None)
        per_query[qid] = {
            "first_relevant_rank": rank,
            "reciprocal_rank": 1 / rank if rank else 0.0,
            "top_doc_ids": ranked,
        }
    return {
        "mrr": sum(row["reciprocal_rank"] for row in per_query.values()) / len(per_query),
        "per_query": per_query,
    }


def splade_scores(args, queries, documents):
    import torch
    from transformers import AutoTokenizer, __version__ as transformers_version
    from model import SpladeModel

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = SpladeModel(args.model).to(device)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()

    def encode(texts, max_length):
        inputs = tokenizer(
            texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        return model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])

    query_ids, document_ids = list(queries), list(documents)
    result = {qid: {} for qid in query_ids}
    active_total = 0
    with torch.inference_mode():
        vectors = []
        for start in range(0, len(query_ids), args.batch_size):
            batch = query_ids[start : start + args.batch_size]
            vectors.append(encode([queries[qid] for qid in batch], args.max_len_q).cpu())
        query_vectors = torch.cat(vectors)
        for start in range(0, len(document_ids), args.batch_size):
            batch = document_ids[start : start + args.batch_size]
            doc_vectors = encode([documents[did] for did in batch], args.max_len_d).cpu()
            active_total += int((doc_vectors > 1e-4).sum().item())
            batch_scores = query_vectors @ doc_vectors.T
            for index, qid in enumerate(query_ids):
                result[qid].update(zip(batch, batch_scores[index].tolist()))
    model_root = Path(args.model)
    model_files = sorted(path for path in model_root.rglob("*") if path.is_file())
    metadata = {
        "model": args.model,
        "model_files_sha256": {
            str(path.relative_to(model_root)): sha256(path) for path in model_files
        },
        "checkpoint_sha256": sha256(args.checkpoint) if args.checkpoint else None,
        "torch_version": torch.__version__,
        "transformers_version": transformers_version,
        "device": str(device),
        "max_len_q": args.max_len_q,
        "max_len_d": args.max_len_d,
        "batch_size": args.batch_size,
        "mean_doc_active_terms_gt_1e-4": active_total / len(documents),
    }
    return result, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, help="JSONL records: doc_id, text")
    parser.add_argument("--queries", required=True, help="JSONL records: query_id, text")
    parser.add_argument("--qrels", required=True, help="JSONL records: query_id, doc_id, relevance")
    parser.add_argument("--method", choices=("bm25", "splade"), required=True)
    parser.add_argument(
        "--out", required=True, help="new output JSON; existing files are never overwritten"
    )
    parser.add_argument("--model", help="local base-model and tokenizer directory for SPLADE")
    parser.add_argument("--checkpoint", help="optional trained splade_weights.pt")
    parser.add_argument("--device", default="cpu", help="SPLADE device, e.g. cuda or cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-len-q", type=int, default=64)
    parser.add_argument("--max-len-d", type=int, default=256)
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_len_q < 1 or args.max_len_d < 1:
        parser.error("batch size and token limits must be positive")
    if args.method == "splade" and not args.model:
        parser.error("--model is required for SPLADE")
    if args.model and not Path(args.model).is_dir():
        parser.error("--model must be an existing local directory")
    if args.method == "bm25" and (args.model or args.checkpoint):
        parser.error("--model and --checkpoint are only used for SPLADE")
    if Path(args.out).exists():
        parser.error("output already exists; choose a new path")
    corpus = read_texts(args.corpus, "doc_id")
    queries = read_texts(args.queries, "query_id")
    qrels = read_qrels(args.qrels, queries, corpus)
    if args.method == "bm25":
        scores = bm25_scores(queries, corpus)
        method_info = {
            "k1": 1.2,
            "b": 0.75,
            "tokenizer": "lowercase Unicode words and CJK characters",
        }
    else:
        scores, method_info = splade_scores(args, queries, corpus)
    result = {
        "schema": "sparse-retrieval-evaluation/v1",
        "method": args.method,
        "candidate_documents": len(corpus),
        "queries": len(queries),
        "cutoff": 10,
        "data_sha256": {
            "corpus": sha256(args.corpus),
            "queries": sha256(args.queries),
            "qrels": sha256(args.qrels),
        },
        "ranking": "positive scores only; ties by doc_id ascending",
        "python_version": platform.python_version(),
        "method_config": method_info,
        **metrics(scores, qrels),
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"method": args.method, "MRR@10": result["mrr"], "output": str(output)}))


if __name__ == "__main__":
    main()
