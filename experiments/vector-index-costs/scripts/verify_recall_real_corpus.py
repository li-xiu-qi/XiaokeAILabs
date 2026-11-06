"""在真实文本 embedding 上验证召回率与查询延迟。

FLAT 精确搜索提供 top-k ground truth，随后对 HNSW、IVF、SQ8、PQ 和 IVF-PQ
做参数扫描。输入文件与维度、条数、模型和语料信息由准备脚本写出的 meta.json
共同确定，避免把不同语料和不同维度混在一起。

默认用法（corpus/ 下有 base_vectors.bin、query_vectors.bin、meta.json）：
    python scripts/verify_recall_real_corpus.py
"""
import argparse
import json
import os
import platform
import statistics
import time

import numpy as np

BASE_BIN = "corpus/base_vectors.bin"
QUERY_BIN = "corpus/query_vectors.bin"
OUT_JSON = "results/recall_real_corpus.json"
TOPK = 10


def load_metadata(base_path, query_path):
    for parent in (os.path.dirname(base_path), os.path.dirname(query_path)):
        candidate = os.path.join(parent or ".", "meta.json")
        if os.path.exists(candidate):
            with open(candidate, encoding="utf-8") as handle:
                return json.load(handle), candidate
    return {}, None


def resolve_positive(value, metadata, key):
    resolved = value if value is not None else metadata.get(key)
    if resolved is None:
        raise ValueError(f"缺少 {key}：请从 meta.json 提供，或显式传 --{key.replace('_', '-')}")
    resolved = int(resolved)
    if resolved <= 0:
        raise ValueError(f"{key} must be positive")
    return resolved


def load_vectors(path, n, dim):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} 不存在。先运行 prepare_corpus_from_20news.py，"
            "或用 --base/--query 指向已准备的 float32 文件"
        )
    expected = n * dim * 4
    actual = os.path.getsize(path)
    if actual != expected:
        raise ValueError(
            f"{path} 大小为 {actual} 字节，期望 {expected}（{n} x {dim} x 4）。"
            "请检查 meta.json 与 --n-base/--n-query/--dim 是否匹配"
        )
    return np.memmap(path, dtype=np.float32, mode="r", shape=(n, dim))


def search_batch(index, queries, k, repeats):
    index.search(queries[:min(8, len(queries))], k)
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        _, ids = index.search(queries, k)
        times.append((time.perf_counter() - started) * 1000.0)
    return ids, statistics.median(times), times


def recall_at_k(pred, truth):
    hits = sum(len(set(pred_row) & set(truth_row)) for pred_row, truth_row in zip(pred, truth))
    return hits / (len(pred) * TOPK)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-base", type=int)
    parser.add_argument("--n-query", type=int)
    parser.add_argument("--query-limit", type=int, help="Use the first N queries after loading the complete query file")
    parser.add_argument("--dim", type=int)
    parser.add_argument("--base", default=BASE_BIN)
    parser.add_argument("--query", default=QUERY_BIN)
    parser.add_argument("--out", default=OUT_JSON)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.threads < 1 or args.repeats < 1:
        raise ValueError("threads and repeats must be >= 1")

    metadata, metadata_path = load_metadata(args.base, args.query)
    n_base = resolve_positive(args.n_base, metadata, "n_base")
    n_query = resolve_positive(args.n_query, metadata, "n_query")
    dim = resolve_positive(args.dim, metadata, "dim")
    # This fixed scan trains 1024 IVF centroids and 32-subspace PQ.
    # Fail before expensive work rather than returning invalid neighbors or a late Faiss error.
    if n_base < 1024:
        parser.error("this scan needs at least 1024 base vectors for nlist=1024")
    if dim % 32:
        parser.error("this scan needs a vector dimension divisible by 32 for PQ m=32")

    import faiss

    faiss.omp_set_num_threads(args.threads)
    print(
        f"loading base={n_base} query={n_query} dim={dim} "
        f"metadata={metadata_path or 'none'}",
        flush=True,
    )
    started = time.time()
    base = load_vectors(args.base, n_base, dim)
    queries = load_vectors(args.query, n_query, dim)
    if args.query_limit is not None:
        if not 0 < args.query_limit <= n_query:
            raise ValueError("query-limit must be between 1 and the number of query vectors")
        queries = queries[:args.query_limit]
        n_query = args.query_limit
    base_array = np.ascontiguousarray(base)
    query_array = np.ascontiguousarray(queries)
    print(f"loaded in {time.time() - started:.1f}s", flush=True)

    print("building ground truth (FLAT) ...", flush=True)
    started = time.time()
    gt_index = faiss.IndexFlatIP(dim)
    gt_index.add(base_array)
    truth_ids, gt_ms, gt_times = search_batch(gt_index, query_array, TOPK, args.repeats)
    del gt_index
    print(
        f"ground truth done in {time.time() - started:.1f}s "
        f"({gt_ms:.1f} ms for {n_query} queries, {gt_ms / n_query:.2f} ms/query)",
        flush=True,
    )

    results = {
        "config": {
            "n_base": n_base,
            "n_query": n_query,
            "dim": dim,
            "topk": TOPK,
            "threads": args.threads,
            "repeats": args.repeats,
            "faiss": faiss.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "corpus": metadata.get("corpus", metadata.get("source", "unknown")),
            "source": metadata.get("source", "unknown"),
            "model": metadata.get("model", "unknown"),
            "replicate": metadata.get("replicate", 1),
            "normalized": metadata.get("normalized"),
        },
        "ground_truth": {
            "total_ms": round(gt_ms, 2),
            "ms_per_query": round(gt_ms / n_query, 4),
            "batch_times_ms": gt_times,
        },
        "indexes": [],
    }

    def run_case(name, build_fn, params_label):
        print(f"  {name} {params_label} ...", flush=True)
        started = time.time()
        index = build_fn()
        build_s = time.time() - started
        ids, query_ms, query_times = search_batch(index, query_array, TOPK, args.repeats)
        recall = recall_at_k(ids, truth_ids)
        record = {
            "index": name,
            "params": params_label,
            "build_s": round(build_s, 2),
            "total_ms": round(query_ms, 2),
            "ms_per_query": round(query_ms / n_query, 4),
            "batch_times_ms": query_times,
            "recall@10": round(recall, 4),
            "speedup_vs_flat": round(gt_ms / query_ms, 2) if query_ms > 0 else None,
        }
        results["indexes"].append(record)
        print(
            f"    recall={recall:.4f}  {query_ms / n_query:.3f} ms/q  build={build_s:.1f}s",
            flush=True,
        )
        del index

    print("HNSW parameter scan (M=16) ...", flush=True)
    for ef in (16, 32, 64, 128, 256):
        def build_hnsw(ef=ef):
            index = faiss.IndexHNSWFlat(dim, 16, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = 64
            index.add(base_array)
            index.hnsw.efSearch = ef
            return index

        run_case("HNSW", build_hnsw, f"M=16 efConstruction=64 efSearch={ef}")

    print("IVF parameter scan (nlist=1024) ...", flush=True)
    for nprobe in (1, 2, 4, 8, 16, 32, 64):
        def build_ivf(nprobe=nprobe):
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, 1024, faiss.METRIC_INNER_PRODUCT)
            index.train(base_array)
            index.add(base_array)
            index.nprobe = nprobe
            return index

        run_case("IVF", build_ivf, f"nlist=1024 nprobe={nprobe}")

    print("quantized indexes ...", flush=True)
    def build_sq8():
        index = faiss.IndexScalarQuantizer(dim, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT)
        index.train(base_array)
        index.add(base_array)
        return index
    run_case("SQ8", build_sq8, "SQ8 8bit metric=IP")

    def build_pq():
        index = faiss.IndexPQ(dim, 32, 8, faiss.METRIC_INNER_PRODUCT)
        index.train(base_array)
        index.add(base_array)
        return index
    run_case("PQ", build_pq, "m=32 nbits=8 metric=IP")

    def build_ivfpq():
        index = faiss.IndexIVFPQ(faiss.IndexFlatIP(dim), dim, 1024, 32, 8, faiss.METRIC_INNER_PRODUCT)
        index.train(base_array)
        index.add(base_array)
        index.nprobe = 8
        return index
    run_case("IVF-PQ", build_ivfpq, "nlist=1024 m=32 nbits=8 nprobe=8 metric=IP")

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)

    print("=" * 60, flush=True)
    print(f"{'index':<10} {'params':<34} {'recall@10':>10} {'ms/q':>9}", flush=True)
    for record in results["indexes"]:
        print(
            f"{record['index']:<10} {record['params']:<34} {record['recall@10']:>10.4f} "
            f"{record['ms_per_query']:>9.3f}",
            flush=True,
        )
    print(f"\nsaved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
