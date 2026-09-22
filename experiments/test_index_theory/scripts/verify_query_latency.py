"""标定向量索引的查询延迟计算量，并验证延迟缩放律。

计算量（一次查询访问多少数据、做多少次距离计算）可理论推导，墙钟时间需在本机标定
一个算力常数后外推。本脚本：
  E1 FLAT：延迟随 n 线性，标定每秒可评向量数 Q（单核）。
  E2 HNSW：延迟随 log n 增长、随 efSearch 线性增长（召回-延迟权衡）。
  E3 IVF：延迟随 nprobe 线性增长（召回-延迟权衡），IVF-Flat vs IVF-PQ。
召回率以 FLAT 精确 top-k 为 ground truth。单线程标定，避免多核调度噪声。

用法（在 test_index_essence 下）：
    <venv>/Scripts/python.exe scripts/verify_query_latency.py
"""

import json
import os
import time

import faiss
import numpy as np

faiss.omp_set_num_threads(1)          # 单核标定，常数干净
D = 128
K = 10
NQ = 100                             # 一批 query，摊销 Python 开销
SEED = 42
NS = [100, 1000, 10_000, 50_000, 100_000, 200_000]


def make(n):
    rng = np.random.default_rng(SEED)
    xb = rng.standard_normal((n, D)).astype("float32")
    xq = rng.standard_normal((NQ, D)).astype("float32")
    return xb, xq


def gt_index(xb):
    idx = faiss.IndexFlatL2(D)
    idx.add(xb)
    return idx


def time_search(index, xq, min_total=0.05, max_iters=300):
    """自适应计时：累计跑到 min_total 秒（或 max_iters 批），返回每 query 毫秒均值。
    小库单批太快时自动多跑，避免微秒级抖动。"""
    iters = 0
    I = None
    t0 = time.perf_counter()
    while True:
        _, I = index.search(xq, K)
        iters += 1
        el = time.perf_counter() - t0
        if el >= min_total or iters >= max_iters:
            break
    return el / iters / len(xq) * 1000, I


def recall_at_k(I_hat, I_true):
    hit = 0
    for i in range(I_true.shape[0]):
        hit += len(set(I_hat[i]) & set(I_true[i]))
    return hit / (I_true.shape[0] * K)


def main():
    results = {"config": {"d": D, "k": K, "nq": NQ, "seed": SEED,
                          "threads": 1, "faiss": faiss.__version__},
               "flat_nscale": [], "hnsw_nscale": [], "hnsw_ef": [],
               "ivf_nprobe": []}

    print(f"配置 d={D} k={K} Nq={NQ} 单核 seed={SEED}")
    print("=" * 78)

    # E1 FLAT 线性 + E2 HNSW 随 n（固定 efSearch=64）
    print("\n[E1/E2] 延迟随 n 变化（HNSW M=16 efSearch=64）")
    print(f"{'n':>8}{'FLAT(ms)':>12}{'HNSW(ms)':>12}{'FLAT/Q(s^-1)':>14}{'HNSW召回':>10}")
    for n in NS:
        xb, xq = make(n)
        gt = gt_index(xb)
        _, I_true = gt.search(xq, K)

        t_flat, _ = time_search(gt, xq)
        q = n / (t_flat / 1000)

        hnsw = faiss.IndexHNSWFlat(D, 16)
        hnsw.hnsw.efConstruction = 40
        hnsw.hnsw.efSearch = 64
        hnsw.add(xb)
        t_hnsw, I_h = time_search(hnsw, xq)
        rec_h = recall_at_k(I_h, I_true)

        results["flat_nscale"].append({"n": n, "ms": round(t_flat, 4),
                                       "q_vec_per_s": int(q)})
        results["hnsw_nscale"].append({"n": n, "ms": round(t_hnsw, 4),
                                       "recall": round(rec_h, 4)})
        print(f"{n:>8}{t_flat:>12.4f}{t_hnsw:>12.4f}{int(q):>14,}{rec_h:>10.3f}")

    # E3 HNSW efSearch 扫描（n=100k）
    n = 100_000
    xb, xq = make(n)
    _, I_true = gt_index(xb).search(xq, K)
    hnsw = faiss.IndexHNSWFlat(D, 16)
    hnsw.hnsw.efConstruction = 40
    hnsw.add(xb)
    print(f"\n[E3a] HNSW 随 efSearch（n={n}）")
    print(f"{'efSearch':>10}{'ms':>10}{'召回':>8}")
    for ef in [16, 32, 64, 128, 256, 512]:
        hnsw.hnsw.efSearch = ef
        t, I_h = time_search(hnsw, xq)
        rec = recall_at_k(I_h, I_true)
        results["hnsw_ef"].append({"ef": ef, "ms": round(t, 4), "recall": round(rec, 4)})
        print(f"{ef:>10}{t:>10.4f}{rec:>8.3f}")

    # E4 IVF nprobe 扫描（n=100k，IVF-Flat vs IVF-PQ）
    nlist = 256
    print(f"\n[E3b] IVF 随 nprobe（n={n} nlist={nlist}）")
    print(f"{'nprobe':>8}{'IVF-Flat(ms)':>14}{'IVF-PQ(ms)':>12}{'Flat召回':>10}{'PQ召回':>9}")
    quantizer = faiss.IndexFlatL2(D)
    ivf_flat = faiss.IndexIVFFlat(quantizer, D, nlist)
    ivf_flat.train(xb)
    ivf_flat.add(xb)
    ivf_pq = faiss.IndexIVFPQ(faiss.IndexFlatL2(D), D, nlist, 8, 8)
    ivf_pq.train(xb)
    ivf_pq.add(xb)
    for nprobe in [1, 4, 8, 16, 32, 64, 128]:
        ivf_flat.nprobe = nprobe
        ivf_pq.nprobe = nprobe
        tf, I_f = time_search(ivf_flat, xq)
        tp, I_p = time_search(ivf_pq, xq)
        rf = recall_at_k(I_f, I_true)
        rp = recall_at_k(I_p, I_true)
        results["ivf_nprobe"].append({"nprobe": nprobe,
                                      "ivfflat_ms": round(tf, 4), "ivfflat_recall": round(rf, 4),
                                      "ivfpq_ms": round(tp, 4), "ivfpq_recall": round(rp, 4)})
        print(f"{nprobe:>8}{tf:>14.4f}{tp:>12.4f}{rf:>10.3f}{rp:>9.3f}")

    out = os.path.join(os.path.dirname(__file__), "..", "results",
                       f"query_latency_d{D}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n已写 {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
