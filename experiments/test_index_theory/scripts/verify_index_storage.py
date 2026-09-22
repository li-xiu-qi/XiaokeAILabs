"""验证索引存储的理论公式 vs faiss 实测。

存储占用的主体是数据结构大小，faiss.serialize_index() 给出索引完整状态的字节数，
它等于写盘大小，也近似内存中的数据结构大小；进程 RSS 增量额外含分配器与临时结构。
压缩类索引（FLAT/SQ/PQ/IVF）的存储由 n、d 和参数决定，理论公式精确；
HNSW 的图边依赖构建结果，理论给期望值，实测揭示真实。

用法（在 test_index_essence 下）：
    <venv>/Scripts/python.exe scripts/verify_index_storage.py
"""

import gc
import json
import os
import sys

import faiss
import numpy as np
import psutil

N = 20000          # 向量数
D = 128            # 维度（小规模验证；规模外推用公式，不跑大数据）
SEED = 42
NLIST = 100        # IVF 聚类数
M_PQ = 8           # PQ 子向量数
NBITS = 8          # PQ 每子向量 bit
M_HNSW = 16        # HNSW 每层邻居上限参数（faiss level0 实际连 2*M）


def make_data():
    rng = np.random.default_rng(SEED)
    xb = rng.standard_normal((N, D)).astype("float32")
    faiss.normalize_L2(xb) if False else None  # 保持 L2 原始数据，不做归一
    return xb


def ser_bytes(index):
    """faiss 序列化字节数 = 索引完整状态大小。"""
    arr = faiss.serialize_index(index)
    return int(arr.nbytes)


def rss_mb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


def theoretical_bytes(name):
    """理论数据结构字节（压缩类精确；HNSW 图为估计）。返回 (字节, 说明)。"""
    n, d = N, D
    if name == "flat":
        return n * d * 4, "n*d*4 原始 float32"
    if name == "sq8":
        return n * d * 1, "n*d*1 每维 1 字节"
    if name == "pq":
        ksub = 2 ** NBITS
        codes = n * M_PQ                       # nbits=8 → 每子向量 1 字节
        codebook = M_PQ * ksub * (d // M_PQ) * 4
        return codes + codebook, "n*m 压缩码 + m*2^b*(d/m)*4 码本"
    if name == "ivf_flat":
        centroids = NLIST * d * 4
        vectors = n * d * 4
        ids = n * 8
        return centroids + vectors + ids, "nlist 中心 + n*d*4 倒排向量 + n*8 id"
    if name == "ivf_pq":
        centroids = NLIST * d * 4
        ksub = 2 ** NBITS
        codebook = M_PQ * ksub * (d // M_PQ) * 4
        codes = n * M_PQ
        ids = n * 8
        return centroids + codebook + codes + ids, "中心 + PQ 码本 + n*m 压缩码 + id"
    if name == "hnsw_flat":
        vectors = n * d * 4
        # 图边估计：level0 上限 2*M 邻居/节点，每边存 1 个邻居 id；
        # 高层邻居与 levels 数组占比小，先按 level0 上限估，实测对照。
        edges_est = n * (2 * M_HNSW) * 8
        return vectors + edges_est, "n*d*4 向量 + 图边(估计 n*2M*8)"
    raise ValueError(name)


def build(name, xb):
    if name == "flat":
        idx = faiss.IndexFlatL2(D)
        idx.add(xb)
        return idx
    if name == "sq8":
        idx = faiss.IndexScalarQuantizer(D, faiss.ScalarQuantizer.QT_8bit)
        idx.train(xb)
        idx.add(xb)
        return idx
    if name == "pq":
        idx = faiss.IndexPQ(D, M_PQ, NBITS)
        idx.train(xb)
        idx.add(xb)
        return idx
    if name == "ivf_flat":
        q = faiss.IndexFlatL2(D)
        idx = faiss.IndexIVFFlat(q, D, NLIST)
        idx.train(xb)
        idx.add(xb)
        return idx
    if name == "ivf_pq":
        q = faiss.IndexFlatL2(D)
        idx = faiss.IndexIVFPQ(q, D, NLIST, M_PQ, NBITS)
        idx.train(xb)
        idx.add(xb)
        return idx
    if name == "hnsw_flat":
        idx = faiss.IndexHNSWFlat(D, M_HNSW)
        idx.hnsw.efConstruction = 40
        idx.add(xb)
        return idx
    raise ValueError(name)


def main():
    xb = make_data()
    raw_bytes = N * D * 4
    print(f"配置: n={N} d={D} seed={SEED} nlist={NLIST} PQ m={M_PQ}/{NBITS}bit HNSW M={M_HNSW}")
    print(f"原始向量 float32: {raw_bytes:>12,} B = {raw_bytes/1024**2:.2f} MiB")
    print("=" * 96)
    print(f"{'索引':<10}{'理论(B)':>14}{'实测serialize(B)':>18}{'理论/实测':>10}{'vs原始':>9}{'RSS增量(MiB)':>14}")
    print("-" * 96)

    results = []
    names = ["flat", "sq8", "pq", "ivf_flat", "ivf_pq", "hnsw_flat"]
    for name in names:
        gc.collect()
        rss0 = rss_mb()
        idx = build(name, xb)
        rss1 = rss_mb()
        theo, theo_note = theoretical_bytes(name)
        meas = ser_bytes(idx)
        ratio = theo / meas if meas else 0
        comp = meas / raw_bytes
        rss_delta = rss1 - rss0
        results.append({
            "index": name, "n": N, "d": D,
            "theory_bytes": theo, "theory_note": theo_note,
            "serialize_bytes": meas, "theory_over_measured": round(ratio, 4),
            "vs_raw": round(comp, 4), "rss_delta_mib": round(rss_delta, 2),
        })
        print(f"{name:<10}{theo:>14,}{meas:>18,}{ratio:>10.3f}{comp:>9.3f}{rss_delta:>14.2f}")
        del idx
        gc.collect()

    print("=" * 96)
    print("注：理论/实测≈1 表示公式准确；HNSW 图边为估计，偏差揭示 faiss 真实图结构。")
    out = os.path.join(os.path.dirname(__file__), "..", "results",
                       f"index_storage_n{N}_d{D}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"config": {"n": N, "d": D, "seed": SEED, "nlist": NLIST,
                              "pq_m": M_PQ, "pq_nbits": NBITS, "hnsw_m": M_HNSW,
                              "faiss": faiss.__version__},
                   "raw_bytes": raw_bytes, "results": results},
                  f, ensure_ascii=False, indent=2)
    print(f"已写 {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
