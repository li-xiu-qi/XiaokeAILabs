"""召回率必须用簇结构数据验证：随机均匀数据是 ANN 最坏情况。

verify_query_latency.py 用标准正态数据，召回率被压到很低（HNSW ef=64 仅 0.38）。
真实 embedding 有强簇结构。本脚本用高斯混合数据（模拟 embedding 的簇分布），
证明同样参数下召回率回到正常水平，而延迟缩放律不变。

用法：<venv>/Scripts/python.exe scripts/verify_recall_clustered.py
"""

import json
import os
import time

import faiss
import numpy as np

faiss.omp_set_num_threads(1)
D, K, NQ, N, SEED = 128, 10, 100, 100_000, 42


def make_clustered(n, n_clusters=200, cluster_std=0.3):
    rng = np.random.default_rng(SEED)
    centers = rng.standard_normal((n_clusters, D)).astype("float32")
    assign = rng.integers(0, n_clusters, size=n)
    xb = centers[assign] + cluster_std * rng.standard_normal((n, D)).astype("float32")
    xq_assign = rng.integers(0, n_clusters, size=NQ)
    xq = centers[xq_assign] + cluster_std * rng.standard_normal((NQ, D)).astype("float32")
    return xb.astype("float32"), xq.astype("float32")


def time_search(index, xq):
    best = []
    I = None
    for _ in range(3):
        t0 = time.perf_counter()
        _, I = index.search(xq, K)
        best.append((time.perf_counter() - t0) / NQ * 1000)
    return float(np.median(best)), I


def recall(I_hat, I_true):
    return sum(len(set(a) & set(b)) for a, b in zip(I_hat, I_true)) / (len(I_true) * K)


def main():
    xb, xq = make_clustered(N)
    gt = faiss.IndexFlatL2(D)
    gt.add(xb)
    _, I_true = gt.search(xq, K)
    print(f"簇状数据 n={N} d={D} 200 簇 单核")
    print("=" * 60)

    hnsw = faiss.IndexHNSWFlat(D, 16)
    hnsw.hnsw.efConstruction = 40
    hnsw.add(xb)
    print(f"\n{'HNSW ef':>10}{'ms':>10}{'召回(簇状)':>12}")
    hnsw_rows = []
    for ef in [16, 32, 64, 128, 256, 512]:
        hnsw.hnsw.efSearch = ef
        t, I_h = time_search(hnsw, xq)
        r = recall(I_h, I_true)
        hnsw_rows.append({"ef": ef, "ms": round(t, 4), "recall": round(r, 4)})
        print(f"{ef:>10}{t:>10.4f}{r:>12.3f}")

    nlist = 256
    ivf = faiss.IndexIVFFlat(faiss.IndexFlatL2(D), D, nlist)
    ivf.train(xb)
    ivf.add(xb)
    print(f"\n{'IVF nprobe':>10}{'ms':>10}{'召回(簇状)':>12}")
    ivf_rows = []
    for nprobe in [1, 4, 8, 16, 32, 64, 128]:
        ivf.nprobe = nprobe
        t, I_i = time_search(ivf, xq)
        r = recall(I_i, I_true)
        ivf_rows.append({"nprobe": nprobe, "ms": round(t, 4), "recall": round(r, 4)})
        print(f"{nprobe:>10}{t:>10.4f}{r:>12.3f}")

    out = os.path.join(os.path.dirname(__file__), "..", "results",
                       "recall_clustered_d128.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"data": "gaussian_mixture_200clusters", "n": N, "d": D,
                   "hnsw_ef": hnsw_rows, "ivf_nprobe": ivf_rows}, f,
                  ensure_ascii=False, indent=2)
    print(f"\n已写 {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
