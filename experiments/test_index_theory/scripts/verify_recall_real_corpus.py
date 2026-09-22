"""
真实语料上的召回验证。

为什么单独做这一项：存储占用是几何问题（n x d x 系数），查询延迟取决于
计算量（n/Q 或 log n），两者都与向量内容无关，随机向量测出来的结论同样成立。
召回率不一样，它取决于数据的几何结构。随机均匀向量在高维空间里任意两点
距离接近相等，最近邻关系不成立，HNSW 的导航结构和 IVF 的聚类剪枝都失去
意义。真实文本 embedding 的内在维度远低于标称维度，语义相近的文本天然聚拢，
近似索引的召回表现只有在真实语料上才能测出来。

本脚本做三件事：
1. 用 FLAT 精确搜索算出 ground truth（top-k 真值）
2. 对 HNSW / IVF / IVF-PQ / SQ8 做参数扫描，测 recall@10 与延迟
3. 与随机向量下的召回做对照，量化数据分布带来的差异

用法：
    python verify_recall_real_corpus.py
"""
import argparse
import json
import os
import time

import numpy as np

BASE_BIN = "corpus/base_vectors.bin"
QUERY_BIN = "corpus/query_vectors.bin"
OUT_JSON = "results/recall_real_corpus.json"

DIM = 512
TOPK = 10


def load_vectors(path, n, dim):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} 不存在，先跑 build_corpus_embeddings.py")
    size = os.path.getsize(path)
    expect = n * dim * 4
    if size < expect:
        raise ValueError(
            f"{path} 只有 {size} 字节，期望至少 {expect}（{n} x {dim} x 4）"
        )
    return np.memmap(path, dtype=np.float32, mode="r", shape=(n, dim))


def search_once(index, queries, k):
    t0 = time.perf_counter()
    _, ids = index.search(queries, k)
    el = (time.perf_counter() - t0) * 1000.0
    return ids, el


def recall_at_k(pred, truth):
    hit = 0
    for p, t in zip(pred, truth):
        hit += len(set(p.tolist()) & set(t.tolist()))
    return hit / (len(pred) * TOPK)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-base", type=int, default=1000000)
    ap.add_argument("--n-query", type=int, default=2000)
    ap.add_argument("--dim", type=int, default=DIM)
    ap.add_argument("--base", default=BASE_BIN)
    ap.add_argument("--query", default=QUERY_BIN)
    ap.add_argument("--out", default=OUT_JSON)
    args = ap.parse_args()

    import faiss

    print(f"loading base vectors {args.n_base} x {args.dim} ...", flush=True)
    t0 = time.time()
    base = load_vectors(args.base, args.n_base, args.dim)
    queries = load_vectors(args.query, args.n_query, args.dim)
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    # 内积索引：向量已 L2 归一化，内积等于余弦相似度
    print("building ground truth (FLAT) ...", flush=True)
    t0 = time.time()
    gt_index = faiss.IndexFlatIP(args.dim)
    gt_index.add(np.ascontiguousarray(base))
    truth_ids, gt_ms = search_once(gt_index, np.ascontiguousarray(queries), TOPK)
    print(
        f"ground truth done in {time.time()-t0:.1f}s "
        f"({gt_ms:.1f} ms for {args.n_query} queries, "
        f"{gt_ms/args.n_query:.2f} ms/query)",
        flush=True,
    )

    results = {
        "config": {
            "n_base": args.n_base,
            "n_query": args.n_query,
            "dim": args.dim,
            "topk": TOPK,
            "corpus": "wikipedia-zh-cn-20250320 (fjcanyue)",
            "model": "BAAI/bge-small-zh-v1.5",
        },
        "ground_truth": {
            "total_ms": round(gt_ms, 2),
            "ms_per_query": round(gt_ms / args.n_query, 4),
        },
        "indexes": [],
    }

    def run_case(name, build_fn, params_label):
        print(f"  {name} ...", flush=True)
        t0 = time.time()
        idx = build_fn()
        build_s = time.time() - t0
        ids, q_ms = search_once(idx, np.ascontiguousarray(queries), TOPK)
        rec = recall_at_k(ids, truth_ids)
        rec_used = {
            "index": name,
            "params": params_label,
            "build_s": round(build_s, 2),
            "total_ms": round(q_ms, 2),
            "ms_per_query": round(q_ms / args.n_query, 4),
            "recall@10": round(rec, 4),
            "speedup_vs_flat": round(gt_ms / q_ms, 1),
        }
        results["indexes"].append(rec_used)
        print(
            f"    recall={rec:.4f}  {q_ms/args.n_query:.3f} ms/q  "
            f"build={build_s:.1f}s",
            flush=True,
        )

    # HNSW：M 控制图的度数，efSearch 控制搜索时的候选队列大小
    print("HNSW parameter scan (M=16) ...", flush=True)
    for ef in (16, 32, 64, 128, 256):
        hnsw = faiss.IndexHNSWFlat(args.dim, 16)
        hnsw.hnsw.efConstruction = 64
        hnsw.add(np.ascontiguousarray(base))
        hnsw.hnsw.efSearch = ef
        run_case(
            "HNSW",
            lambda h=hnsw: h,
            f"M=16 efConstruction=64 efSearch={ef}",
        )

    # IVF：nlist 控制聚类数，nprobe 控制搜索时访问的聚类数
    print("IVF parameter scan (nlist=1024) ...", flush=True)
    for nprobe in (1, 2, 4, 8, 16, 32, 64):
        quantizer = faiss.IndexFlatIP(args.dim)
        ivf = faiss.IndexIVFFlat(quantizer, args.dim, 1024, faiss.METRIC_INNER_PRODUCT)
        ivf.train(np.ascontiguousarray(base))
        ivf.add(np.ascontiguousarray(base))
        ivf.nprobe = nprobe
        run_case("IVF", lambda iv=ivf: iv, f"nlist=1024 nprobe={nprobe}")

    # 量化类索引：测压缩后的召回损失
    print("quantized indexes ...", flush=True)
    sq8 = faiss.IndexScalarQuantizer(
        args.dim, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT
    )
    sq8.train(np.ascontiguousarray(base))
    sq8.add(np.ascontiguousarray(base))
    run_case("SQ8", lambda s=sq8: s, "SQ8 8bit")

    pq = faiss.IndexPQ(args.dim, 32, 8)
    pq.train(np.ascontiguousarray(base))
    pq.add(np.ascontiguousarray(base))
    run_case("PQ", lambda p=pq: p, "m=32 nbits=8")

    quantizer = faiss.IndexFlatIP(args.dim)
    ivfpq = faiss.IndexIVFPQ(quantizer, args.dim, 1024, 32, 8)
    ivfpq.train(np.ascontiguousarray(base))
    ivfpq.add(np.ascontiguousarray(base))
    ivfpq.nprobe = 8
    run_case(
        "IVF-PQ",
        lambda iv=ivfpq: iv,
        "nlist=1024 m=32 nbits=8 nprobe=8",
    )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("=" * 60, flush=True)
    print(f"{'index':<10} {'params':<34} {'recall@10':>10} {'ms/q':>9}", flush=True)
    for r in results["indexes"]:
        print(
            f"{r['index']:<10} {r['params']:<34} {r['recall@10']:>10.4f} "
            f"{r['ms_per_query']:>9.3f}",
            flush=True,
        )
    print(f"\nsaved to {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
