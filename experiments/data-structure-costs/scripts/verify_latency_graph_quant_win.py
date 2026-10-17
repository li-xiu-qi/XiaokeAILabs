"""Windows 原生图量化索引标定。

标定 faiss-cpu 1.15.0 上的 HNSW-SQ / HNSW-PQ / IVF-RaBitQ，双平台对照的
Windows 一侧。与 `verify_latency_wsl_crosscheck.py` 的数据集、参数、指标
口径完全一致（n=10 万、d=128、单核、ef/nprobe 同值），结果落盘
`results/win_graph_quant_d128.json`。

注意 `IndexHNSWPQ` 的 C++ 签名是 `(d, pq_M, M, pq_nbits)`，与 faiss 自带的
`__init__.pyi` stub 写的 `(d, pq, M, metric)` 不符，stub 是错的。把第二、三
个参数传反（写成 `(d, M, pq_M, nbits)`）不会报错，但 pq_M 会取到 16，召回
钉在 0.04 且延迟不随 pq_M 变化，看起来像「PQ 失效」，实际是参数语义错。

每个 case 用独立子进程跑，隔离 faiss 的 C++ 断言与访问违例，避免一个
失败 case 连带整个脚本退出。
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
D = 128
N = 100000
K = 10
NQ = 100
REPEAT = 3

HEAD = """
import faiss, numpy as np, time
faiss.omp_set_num_threads(1)
D={d}; N={n}; K={k}; NQ={nq}
rng = np.random.default_rng(42)
c = rng.standard_normal((200, D)).astype('float32')
xb = (c[rng.integers(0, 200, N)] + 0.3 * rng.standard_normal((N, D)).astype('float32')).astype('float32')
xq = (c[rng.integers(0, 200, NQ)] + 0.3 * rng.standard_normal((NQ, D)).astype('float32')).astype('float32')
gt = faiss.IndexFlatL2(D); gt.add(xb); _, It = gt.search(xq, K)
def rec(I):
    return sum(len(set(a) & set(b)) for a, b in zip(I, It)) / (NQ * K)
def ms(idx):
    t0 = time.perf_counter()
    for _ in range({rep}):
        idx.search(xq, K)
    return (time.perf_counter() - t0) / {rep} / NQ * 1000
""".format(d=D, n=N, k=K, nq=NQ, rep=REPEAT)

TAIL = """
if hasattr(i, 'train'):
    i.train(xb)
i.add(xb)
t = ms(i)
_, I = i.search(xq, K)
print('RESULT', round(t, 4), round(rec(I), 4))
"""

CASES = [
    ("HNSW-Flat", "faiss.IndexHNSWFlat(D, 16)",
     "i.hnsw.efConstruction=40; i.hnsw.efSearch=64"),
    ("HNSW-SQ8", "faiss.IndexHNSWSQ(D, faiss.ScalarQuantizer.QT_8bit, 16)",
     "i.hnsw.efConstruction=40; i.hnsw.efSearch=64"),
    ("HNSW-SQ4", "faiss.IndexHNSWSQ(D, faiss.ScalarQuantizer.QT_4bit, 16)",
     "i.hnsw.efConstruction=40; i.hnsw.efSearch=64"),
    # 对照组：参数顺序传反，验证失败形态（Windows 上是进程崩溃，无 Python 异常）
    ("HNSW-SQ8-wrong-order", "faiss.IndexHNSWSQ(D, 16, faiss.ScalarQuantizer.QT_8bit)",
     "i.hnsw.efConstruction=40; i.hnsw.efSearch=64"),
    # pq_M 扫描（M=16、nbits=8、ef=64）。pq_M 必须整除 d，d=128 时合法值
    # 只有 16、32、64、128，传 48 会抛 `!(d % M == 0)`。
    ("HNSW-PQ-pqM16", "faiss.IndexHNSWPQ(D, 16, 16, 8)",
     "i.hnsw.efConstruction=40; i.hnsw.efSearch=64"),
    ("HNSW-PQ-pqM32", "faiss.IndexHNSWPQ(D, 32, 16, 8)",
     "i.hnsw.efConstruction=40; i.hnsw.efSearch=64"),
    ("HNSW-PQ-pqM64", "faiss.IndexHNSWPQ(D, 64, 16, 8)",
     "i.hnsw.efConstruction=40; i.hnsw.efSearch=64"),
    ("HNSW-PQ-pqM128", "faiss.IndexHNSWPQ(D, 128, 16, 8)",
     "i.hnsw.efConstruction=40; i.hnsw.efSearch=64"),
    # ef 扫描（pq_M=128、M=16、nbits=8），用于填主文档的 ef 对比表
    ("HNSW-PQ-pqM128-ef16", "faiss.IndexHNSWPQ(D, 128, 16, 8)",
     "i.hnsw.efConstruction=40; i.hnsw.efSearch=16"),
    ("HNSW-PQ-pqM128-ef32", "faiss.IndexHNSWPQ(D, 128, 16, 8)",
     "i.hnsw.efConstruction=40; i.hnsw.efSearch=32"),
    ("HNSW-PQ-pqM128-ef128", "faiss.IndexHNSWPQ(D, 128, 16, 8)",
     "i.hnsw.efConstruction=40; i.hnsw.efSearch=128"),
    ("HNSW-PQ-pqM128-ef256", "faiss.IndexHNSWPQ(D, 128, 16, 8)",
     "i.hnsw.efConstruction=40; i.hnsw.efSearch=256"),
    # HNSW-Flat / HNSW-SQ8 / HNSW-SQ4 的 ef 扫描，给 HNSW-PQ 的 ef 表提供
    # 同环境分母。ef=64 已由上面的基础 case 覆盖，这里只补 16/32/128/256，
    # 四列齐了 ef 对比表才能整表来自同一次运行、同一平台。
    ("HNSW-Flat-ef16", "faiss.IndexHNSWFlat(D, 16)",
     "i.hnsw.efConstruction=40; i.hnsw.efSearch=16"),
    ("HNSW-Flat-ef32", "faiss.IndexHNSWFlat(D, 16)",
     "i.hnsw.efConstruction=40; i.hnsw.efSearch=32"),
    ("HNSW-Flat-ef128", "faiss.IndexHNSWFlat(D, 16)",
     "i.hnsw.efConstruction=40; i.hnsw.efSearch=128"),
    ("HNSW-Flat-ef256", "faiss.IndexHNSWFlat(D, 16)",
     "i.hnsw.efConstruction=40; i.hnsw.efSearch=256"),
    ("HNSW-SQ8-ef16", "faiss.IndexHNSWSQ(D, faiss.ScalarQuantizer.QT_8bit, 16)",
     "i.hnsw.efConstruction=40; i.hnsw.efSearch=16"),
    ("HNSW-SQ8-ef32", "faiss.IndexHNSWSQ(D, faiss.ScalarQuantizer.QT_8bit, 16)",
     "i.hnsw.efConstruction=40; i.hnsw.efSearch=32"),
    ("HNSW-SQ8-ef128", "faiss.IndexHNSWSQ(D, faiss.ScalarQuantizer.QT_8bit, 16)",
     "i.hnsw.efConstruction=40; i.hnsw.efSearch=128"),
    ("HNSW-SQ8-ef256", "faiss.IndexHNSWSQ(D, faiss.ScalarQuantizer.QT_8bit, 16)",
     "i.hnsw.efConstruction=40; i.hnsw.efSearch=256"),
    ("HNSW-SQ4-ef16", "faiss.IndexHNSWSQ(D, faiss.ScalarQuantizer.QT_4bit, 16)",
     "i.hnsw.efConstruction=40; i.hnsw.efSearch=16"),
    ("HNSW-SQ4-ef32", "faiss.IndexHNSWSQ(D, faiss.ScalarQuantizer.QT_4bit, 16)",
     "i.hnsw.efConstruction=40; i.hnsw.efSearch=32"),
    ("HNSW-SQ4-ef128", "faiss.IndexHNSWSQ(D, faiss.ScalarQuantizer.QT_4bit, 16)",
     "i.hnsw.efConstruction=40; i.hnsw.efSearch=128"),
    ("HNSW-SQ4-ef256", "faiss.IndexHNSWSQ(D, faiss.ScalarQuantizer.QT_4bit, 16)",
     "i.hnsw.efConstruction=40; i.hnsw.efSearch=256"),
    # IVF_RaBitQ 签名 (quantizer, d, nlist, metric, own_invlists, nb_bits)。
    # 第 4 位是 MetricType 不是 nb_bits，传 2/3/4 等于选 L1/Linf/Lp，RaBitQ
    # 只实现 L2 与内积，会触发 metric 断言。曾据此误判成「qb>=2 不可用」。
    ("IVF-RaBitQ-nb1", "faiss.IndexIVFRaBitQ(faiss.IndexFlatL2(D), D, 256, faiss.METRIC_L2, True, 1)",
     "i.nprobe=8"),
    ("IVF-RaBitQ-nb2", "faiss.IndexIVFRaBitQ(faiss.IndexFlatL2(D), D, 256, faiss.METRIC_L2, True, 2)",
     "i.nprobe=8"),
    ("IVF-RaBitQ-nb4", "faiss.IndexIVFRaBitQ(faiss.IndexFlatL2(D), D, 256, faiss.METRIC_L2, True, 4)",
     "i.nprobe=8"),
    ("IVF-RaBitQ-nb8", "faiss.IndexIVFRaBitQ(faiss.IndexFlatL2(D), D, 256, faiss.METRIC_L2, True, 8)",
     "i.nprobe=8"),
    # 查询量化位数 qb（属性，默认 4）。qb=0 用原始 fp32，慢一个数量级。
    ("IVF-RaBitQ-qb0", "faiss.IndexIVFRaBitQ(faiss.IndexFlatL2(D), D, 256, faiss.METRIC_L2, True, 1)",
     "i.qb=0; i.nprobe=8"),
    ("IVF-RaBitQ-qb1", "faiss.IndexIVFRaBitQ(faiss.IndexFlatL2(D), D, 256, faiss.METRIC_L2, True, 1)",
     "i.qb=1; i.nprobe=8"),
    ("IVF-RaBitQ-qb8", "faiss.IndexIVFRaBitQ(faiss.IndexFlatL2(D), D, 256, faiss.METRIC_L2, True, 1)",
     "i.qb=8; i.nprobe=8"),
    # 对照组：第 4 位误传 metric=2（METRIC_L1），RaBitQ 不支持
    ("IVF-RaBitQ-metric-L1", "faiss.IndexIVFRaBitQ(faiss.IndexFlatL2(D), D, 256, faiss.METRIC_L1)",
     "i.nprobe=8"),
]


def run_case(ctor, extra):
    code = "import faiss\n" + HEAD + "\ni = " + ctor + "\n" + extra + "\n" + TAIL
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=900)
    line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
    if line:
        parts = line.split()
        return {"status": "ok", "latency_ms": float(parts[1]), "recall": float(parts[2])}
    err = (p.stderr or "").strip().splitlines()
    tail = next((l for l in reversed(err) if l.strip()), "")
    return {"status": "fail", "returncode": p.returncode, "error": tail[:200]}


def main():
    try:
        import faiss
        ver = faiss.__version__
    except ImportError as exc:
        print("faiss 不可用:", exc)
        return 1

    results = {"platform": "windows", "faiss_version": ver, "n": N, "d": D, "k": K, "nq": NQ,
               "repeat": REPEAT, "cases": {}}
    for name, ctor, extra in CASES:
        t0 = time.perf_counter()
        r = run_case(ctor, extra)
        r["wall_s"] = round(time.perf_counter() - t0, 1)
        results["cases"][name] = r
        if r["status"] == "ok":
            print(f"{name:24s} {r['latency_ms']:8.4f} ms  recall={r['recall']:.4f}")
        else:
            print(f"{name:24s} FAIL rc={r['returncode']} :: {r['error'][:70]}")

    out = ROOT / "results" / "win_graph_quant_d128.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
