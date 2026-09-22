"""在 WSL 里做跨环境对照，标定量化索引在两个平台上的定性是否一致。

注意：本脚本第 84 行的 `faiss.IndexHNSWPQ(D, 16, 32, 8)` 用的是错的参数顺序。
C++ 签名是 `(d, pq_M, M, pq_nbits)`，这里实际取到 pq_M=16，对 d=128 而言
每子向量 8 维，召回只有 0.04 左右。所以本脚本产出的 HNSW-PQ 一列不可用，
已由 `verify_latency_graph_quant_win.py` 用 pq_M=128 重采。其余各列
（HNSW-Flat、HNSW-SQ8、HNSW-SQ4、IVF 各量化档）不受影响。

用法（WSL Ubuntu）：python3 verify_latency_wsl_crosscheck.py
"""

import json
import os
import subprocess
import sys
import time

import faiss
import numpy as np

faiss.omp_set_num_threads(1)
D, K, NQ, N, SEED = 128, 10, 100, 100_000, 42
NLIST = 256
SQ = faiss.ScalarQuantizer


def make_clustered(n):
    rng = np.random.default_rng(SEED)
    centers = rng.standard_normal((200, D)).astype("float32")
    assign = rng.integers(0, 200, size=n)
    xb = centers[assign] + 0.3 * rng.standard_normal((n, D)).astype("float32")
    xq = centers[rng.integers(0, 200, size=NQ)] + 0.3 * rng.standard_normal((NQ, D)).astype("float32")
    return xb.astype("float32"), xq.astype("float32")


def time_search(index, xq):
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


def run_isolated(code, timeout=600):
    """子进程跑一段代码，返回 stdout 末行；C++ terminate 不连坐主脚本。"""
    p = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, timeout=timeout)
    if p.returncode != 0:
        err = (p.stderr or "").strip().splitlines()
        return None, (err[-1][:100] if err else "unknown")
    lines = [l for l in p.stdout.splitlines() if l.strip()]
    return (lines[-1] if lines else None), None


xb, xq = make_clustered(N)
gt = faiss.IndexFlatL2(D)
gt.add(xb)
_, I_true = gt.search(xq, K)
print(f"WSL 簇状数据 n={N} d={D} nlist={NLIST} 单核 faiss={faiss.__version__}")
print("=" * 96)

# ---- E7 图 + 量化复合：同一 efSearch 下改单位成本 ----
print("\n[E7] 图索引随 efSearch（HNSW-Flat vs HNSW-SQ8 vs HNSW-PQ）")
hnsw_flat = faiss.IndexHNSWFlat(D, 16)
hnsw_flat.hnsw.efConstruction = 40
hnsw_flat.add(xb)
# 正确参数顺序：(d, qtype, M)
hnsw_sq8 = faiss.IndexHNSWSQ(D, SQ.QT_8bit, 16)
hnsw_sq8.hnsw.efConstruction = 40
hnsw_sq8.train(xb)
hnsw_sq8.add(xb)
hnsw_sq4 = faiss.IndexHNSWSQ(D, SQ.QT_4bit, 16)
hnsw_sq4.hnsw.efConstruction = 40
hnsw_sq4.train(xb)
hnsw_sq4.add(xb)
hnsw_pq = faiss.IndexHNSWPQ(D, 16, 32, 8)
hnsw_pq.hnsw.efConstruction = 40
hnsw_pq.train(xb)
hnsw_pq.add(xb)

print(f"{'efSearch':>10}{'Flat(ms)':>11}{'SQ8(ms)':>11}{'SQ4(ms)':>11}{'PQ(ms)':>11}"
      f"{'Flat召回':>10}{'SQ8召回':>10}{'PQ召回':>9}")
graph_rows = []
for ef in [16, 32, 64, 128, 256]:
    hnsw_flat.hnsw.efSearch = ef
    hnsw_sq8.hnsw.efSearch = ef
    hnsw_sq4.hnsw.efSearch = ef
    hnsw_pq.hnsw.efSearch = ef
    tf, I_f = time_search(hnsw_flat, xq)
    ts8, I_s8 = time_search(hnsw_sq8, xq)
    ts4, I_s4 = time_search(hnsw_sq4, xq)
    tp, I_p = time_search(hnsw_pq, xq)
    rf = recall_at_k(I_f, I_true)
    rs8 = recall_at_k(I_s8, I_true)
    rs4 = recall_at_k(I_s4, I_true)
    rp = recall_at_k(I_p, I_true)
    graph_rows.append({"ef": ef, "flat_ms": round(tf, 4), "sq8_ms": round(ts8, 4),
                       "sq4_ms": round(ts4, 4), "pq_ms": round(tp, 4),
                       "flat_recall": round(rf, 4), "sq8_recall": round(rs8, 4),
                       "sq4_recall": round(rs4, 4), "pq_recall": round(rp, 4)})
    print(f"{ef:>10}{tf:>11.4f}{ts8:>11.4f}{ts4:>11.4f}{tp:>11.4f}"
          f"{rf:>10.3f}{rs8:>10.3f}{rp:>9.3f}")

# ---- E8 RaBitQ（仅 qb=1 可用，qb>=2 隔离跑）----
print(f"\n[E8] IVF-RaBitQ 随 nprobe（子进程隔离，qb>=2 会 abort）")
rabitq_rows = []
for qb in [1, 2, 4]:
    out, err = run_isolated(f"""
import faiss, numpy as np, time
faiss.omp_set_num_threads(1)
D=128; NLIST=256; NPROBE={8}; K=10
rng=np.random.default_rng(42)
c=rng.standard_normal((200,D)).astype('float32')
xb=(c[rng.integers(0,200,100000)]+0.3*rng.standard_normal((100000,D)).astype('float32')).astype('float32')
xq=(c[rng.integers(0,200,100)]+0.3*rng.standard_normal((100,D)).astype('float32')).astype('float32')
idx=faiss.IndexIVFRaBitQ(faiss.IndexFlatL2(D),D,NLIST,{qb})
idx.train(xb); idx.add(xb); idx.nprobe=NPROBE
t0=time.perf_counter()
for _ in range(3): idx.search(xq,K)
t=(time.perf_counter()-t0)/3/100*1000
gt=faiss.IndexFlatL2(D); gt.add(xb); _,It=gt.search(xq,K)
_,I=idx.search(xq,K)
r=sum(len(set(a)&set(b)) for a,b in zip(I,It))/1000
print('%.4f %.4f'%(t,r))
""")
    if out:
        t, r = out.split()
        rabitq_rows.append({"qb": qb, "nprobe": 8, "ms": float(t), "recall": float(r)})
        print(f"  qb={qb}  {t} ms  recall={r}")
    else:
        rabitq_rows.append({"qb": qb, "nprobe": 8, "error": err})
        print(f"  qb={qb}  FAIL :: {err[:70]}")

# ---- E9 量化档位（含 IVF-Flat 对照）----
print(f"\n[E9] IVF 量化档位随 nprobe（每格 ms/召回）")
print(f"{'nprobe':>8}{'Flat':>13}{'SQ8':>13}{'SQ6':>13}{'SQ4':>13}{'fp16':>13}{'PQ(m=8)':>13}")
variants = {
    "SQ8": lambda: faiss.IndexIVFScalarQuantizer(faiss.IndexFlatL2(D), D, NLIST, SQ.QT_8bit, faiss.METRIC_L2),
    "SQ6": lambda: faiss.IndexIVFScalarQuantizer(faiss.IndexFlatL2(D), D, NLIST, SQ.QT_6bit, faiss.METRIC_L2),
    "SQ4": lambda: faiss.IndexIVFScalarQuantizer(faiss.IndexFlatL2(D), D, NLIST, SQ.QT_4bit, faiss.METRIC_L2),
    "fp16": lambda: faiss.IndexIVFScalarQuantizer(faiss.IndexFlatL2(D), D, NLIST, SQ.QT_fp16, faiss.METRIC_L2),
    "PQ": lambda: faiss.IndexIVFPQ(faiss.IndexFlatL2(D), D, NLIST, 8, 8),
}
built = {"Flat": faiss.IndexIVFFlat(faiss.IndexFlatL2(D), D, NLIST)}
built["Flat"].train(xb)
built["Flat"].add(xb)
for name, mk in variants.items():
    idx = mk()
    idx.train(xb)
    idx.add(xb)
    built[name] = idx

quant_rows = []
for nprobe in [1, 4, 8, 16, 32]:
    row = {"nprobe": nprobe}
    cells = f"{nprobe:>8}"
    for name, idx in built.items():
        idx.nprobe = nprobe
        t, I_h = time_search(idx, xq)
        r = recall_at_k(I_h, I_true)
        row[name] = {"ms": round(t, 4), "recall": round(r, 4)}
        cells += f"{t:>8.4f}/{r:<4.2f}"
    quant_rows.append(row)
    print(cells)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   f"wsl_latency_quant_d{D}.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump({"env": {"faiss": faiss.__version__, "py": "3.10.12",
                       "os": "WSL2 Ubuntu-22.04"},
               "config": {"d": D, "k": K, "nq": NQ, "n": N, "nlist": NLIST,
                          "threads": 1, "data": "gaussian_mixture_200clusters"},
               "graph_quant_ef": graph_rows, "rabitq": rabitq_rows,
               "ivf_quant_nprobe": quant_rows}, f, ensure_ascii=False, indent=2)
print(f"\n已写 {out}")
