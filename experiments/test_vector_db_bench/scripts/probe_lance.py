"""LanceDB IVF_PQ 参数对照：定位 30000/768 召回 0.04 的根因。"""
import shutil
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import lancedb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
z = np.load(ROOT / "data" / "20news_emb.npz", allow_pickle=True)
vecs = z["train_emb"].astype("float32")
q = z["test_emb"].astype("float32")[:50]


def gt(vecs, q, k=10):
    vv = np.sum(vecs**2, 1)
    qq = np.sum(q**2, 1)
    out = []
    for i in range(0, len(q), 25):
        d = qq[i:i + 25, None] + vv[None, :] - 2 * (q[i:i + 25] @ vecs.T)
        out.append(np.argsort(d, 1)[:, :k])
    return np.concatenate(out, 0)


G = gt(vecs, q)


def recall_at(tbl, q, G, efs=(16, 64, 256), sub=32, part=256, refine=0):
    tbl.create_index(metric="l2", num_partitions=part, num_sub_vectors=sub)
    for ef in efs:
        lat, rc = [], []
        for i, qq in enumerate(q):
            t1 = time.perf_counter()
            s = tbl.search(qq.tolist()).limit(10).nprobes(ef)
            if refine:
                s = s.refine_factor(refine)
            r = s.to_list()
            lat.append((time.perf_counter() - t1) * 1000)
            rc.append(len({x["pk"] for x in r} & set(G[i].tolist())) / 10)
        print(f"  ef={ef:<4} p50={np.median(lat):7.1f}ms recall={np.mean(rc):.4f}", flush=True)


for name, sub, part, refine in [
    ("sub=32 part=256 (旧参数)", 32, 256, 0),
    ("sub=48 part=128", 48, 128, 0),
    ("sub=64 part=128", 64, 128, 0),
    ("sub=64 part=128 refine=10", 64, 128, 10),
    ("sub=96 part=64", 96, 64, 0),
]:
    print(f"\n== {name}", flush=True)
    p = ROOT / "data" / ("lance_probe_%d_%d_%d" % (sub, part, refine))
    if p.exists():
        shutil.rmtree(p, ignore_errors=True)
    db = lancedb.connect(str(p))
    tbl = db.create_table("vec",
                          data=[{"pk": int(i), "vector": vecs[i].tolist()} for i in range(len(vecs))])
    recall_at(tbl, q, G, sub=sub, part=part, refine=refine)
    shutil.rmtree(p, ignore_errors=True)
