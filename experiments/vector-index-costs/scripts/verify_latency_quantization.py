"""标定量化档位与 PQ 子向量数的延迟，验证「访问量 × 单位成本」两段式模型。

verify_query_latency.py 已标定 FLAT / IVF-Flat / HNSW 的剪枝缩放律，本脚本补上
量化这一维，并修正原文档一处表述：
  E4 IVF 固定 nprobe，对比量化档位 SQ8 / SQ6 / SQ4 / SQfp16 / PQ。
     实测 SQ8 最快（direct uint8 编码），低位 SQ 因解包反而更慢。
  E5 PQ 子向量数 m 扫描：m 越大召回越高、延迟也越高。
     原文档说「PQ 距离成本与 d 无关」，补充为「成本正比于 m，而 m 必须随 d
     增大才能保住召回，所以实际成本仍随 d 增长，只是常数小于 float32」。
  E6 IVF-PQ 加精排（refine）：粗排召回不足时用精确距离对候选重排，拿回召回。

延迟与数据分布无关，召回必须在簇状数据上测（见 verify_recall_clustered.py 的
方法论说明）。本脚本用高斯混合簇状数据，两项同时有效。

用法：<venv>/Scripts/python.exe scripts/verify_latency_quantization.py
"""

import json
import os
import time

import faiss
import numpy as np

faiss.omp_set_num_threads(1)
D, K, NQ, N, SEED = 128, 10, 100, 100_000, 42
NLIST = 256
SQ = faiss.ScalarQuantizer


def make_clustered(n):
    """高斯混合 200 簇，模拟真实 embedding 的簇分布。"""
    rng = np.random.default_rng(SEED)
    centers = rng.standard_normal((200, D)).astype("float32")
    assign = rng.integers(0, 200, size=n)
    xb = centers[assign] + 0.3 * rng.standard_normal((n, D)).astype("float32")
    xq = centers[rng.integers(0, 200, size=NQ)] + 0.3 * rng.standard_normal((NQ, D)).astype("float32")
    return xb.astype("float32"), xq.astype("float32")


def time_search(index, xq):
    """3 次取中位数，抵消调度抖动。返回每 query 毫秒。"""
    best = []
    I = None
    for _ in range(3):
        t0 = time.perf_counter()
        _, I = index.search(xq, K)
        best.append((time.perf_counter() - t0) / len(xq) * 1000)
    return float(np.median(best)), I


def recall_at_k(I_hat, I_true):
    hit = sum(len(set(a) & set(b)) for a, b in zip(I_hat, I_true))
    return hit / (I_true.shape[0] * K)


def build(mk, xb):
    idx = mk()
    if hasattr(idx, "train"):
        idx.train(xb)
    idx.add(xb)
    return idx


def main():
    xb, xq = make_clustered(N)
    gt = faiss.IndexFlatL2(D)
    gt.add(xb)
    _, I_true = gt.search(xq, K)
    print(f"簇状数据 n={N} d={D} 200 簇 nlist={NLIST} 单核 faiss={faiss.__version__}")
    print("=" * 96)

    # ---- E4 量化档位：访问量固定（同 nprobe），只改单位成本 ----
    variants = {
        "IVF-SQ8": lambda: faiss.IndexIVFScalarQuantizer(
            faiss.IndexFlatL2(D), D, NLIST, SQ.QT_8bit, faiss.METRIC_L2),
        "IVF-SQ6": lambda: faiss.IndexIVFScalarQuantizer(
            faiss.IndexFlatL2(D), D, NLIST, SQ.QT_6bit, faiss.METRIC_L2),
        "IVF-SQ4": lambda: faiss.IndexIVFScalarQuantizer(
            faiss.IndexFlatL2(D), D, NLIST, SQ.QT_4bit, faiss.METRIC_L2),
        "IVF-SQfp16": lambda: faiss.IndexIVFScalarQuantizer(
            faiss.IndexFlatL2(D), D, NLIST, SQ.QT_fp16, faiss.METRIC_L2),
        "IVF-PQ(m=8)": lambda: faiss.IndexIVFPQ(faiss.IndexFlatL2(D), D, NLIST, 8, 8),
    }
    built = {name: build(mk, xb) for name, mk in variants.items()}

    nprobes = [1, 4, 8, 16, 32]
    print(f"\n[E4] IVF 量化档位随 nprobe（每格 ms/召回）")
    print(f"{'nprobe':>8}" + "".join(f"{n:>15}" for n in variants))
    rows = []
    for nprobe in nprobes:
        row = {"nprobe": nprobe}
        cells = f"{nprobe:>8}"
        for name, idx in built.items():
            idx.nprobe = nprobe
            t, I_h = time_search(idx, xq)
            r = recall_at_k(I_h, I_true)
            row[name] = {"ms": round(t, 4), "recall": round(r, 4)}
            cells += f"{t:>10.4f}/{r:<4.2f}"
        rows.append(row)
        print(cells)

    # ---- E5 PQ 子向量数 m：m 越大召回越高，延迟也越高 ----
    ms_list = [8, 16, 32, 64, 128]
    print(f"\n[E5] IVF-PQ 子向量数 m 扫描（nprobe=16，子向量长度 = 128/m 维）")
    print(f"{'m':>6}{'子向量维':>9}{'延迟(ms)':>12}{'召回@10':>10}")
    pq_rows = []
    for m in ms_list:
        idx = build(lambda m=m: faiss.IndexIVFPQ(
            faiss.IndexFlatL2(D), D, NLIST, m, 8), xb)
        idx.nprobe = 16
        t, I_h = time_search(idx, xq)
        r = recall_at_k(I_h, I_true)
        pq_rows.append({"m": m, "sub_dim": D // m, "ms": round(t, 4),
                        "recall": round(r, 4)})
        print(f"{m:>6}{D // m:>9}{t:>12.4f}{r:>10.3f}")

    # ---- E6 IVF-PQ 精排：粗排召回不足时用精确距离重排候选 ----
    coarse = faiss.IndexIVFPQ(faiss.IndexFlatL2(D), D, NLIST, 8, 8)
    coarse.train(xb)
    coarse.add(xb)
    fine = faiss.IndexFlatL2(D)
    fine.add(xb)
    refine = faiss.IndexRefine(coarse, fine)
    refine.k_factor = 4
    print(f"\n[E6] IVF-PQ(m=8) 精排 k_factor=4 vs 无精排")
    print(f"{'nprobe':>8}{'PQ(ms)':>11}{'PQ召回':>10}{'Refine(ms)':>13}{'Refine召回':>12}{'增益':>9}")
    refine_rows = []
    for nprobe in nprobes:
        coarse.nprobe = nprobe
        tp, I_p = time_search(coarse, xq)
        tr, I_r = time_search(refine, xq)
        rp, rr = recall_at_k(I_p, I_true), recall_at_k(I_r, I_true)
        refine_rows.append({"nprobe": nprobe, "pq_ms": round(tp, 4), "pq_recall": round(rp, 4),
                            "refine_ms": round(tr, 4), "refine_recall": round(rr, 4),
                            "gain": round(rr - rp, 4)})
        print(f"{nprobe:>8}{tp:>11.4f}{rp:>10.3f}{tr:>13.4f}{rr:>12.3f}{rr - rp:>+9.3f}")

    out = os.path.join(os.path.dirname(__file__), "..", "results",
                       f"latency_quantization_d{D}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"config": {"d": D, "k": K, "nq": NQ, "n": N, "nlist": NLIST,
                              "threads": 1, "faiss": faiss.__version__,
                              "data": "gaussian_mixture_200clusters"},
                   "ivf_quant_nprobe": rows, "pq_subdim_m": pq_rows,
                   "pq_refine": refine_rows}, f, ensure_ascii=False, indent=2)
    print(f"\n已写 {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
