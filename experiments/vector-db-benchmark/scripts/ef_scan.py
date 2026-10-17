"""ef / EFC 参数扫描：确认召回率低是参数问题还是高维数据本身的难度。

同时对比均匀随机数据与聚簇数据（后者更像真实 embedding）。
"""
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from bench import SurrealBench, MilvusBench, make_dataset, ground_truth  # noqa: E402


def make_clustered(n, dim, seed=7, n_clusters=50):
    """聚簇分布，模拟真实 embedding 的语义聚集特征。"""
    rng = np.random.default_rng(seed)
    centers = rng.random((n_clusters, dim), dtype="float32")
    labels = rng.integers(0, n_clusters, size=n)
    vecs = centers[labels] + rng.normal(0, 0.08, (n, dim)).astype("float32")
    q = centers[rng.integers(0, n_clusters, 200)] + rng.normal(
        0, 0.08, (200, dim)
    ).astype("float32")
    return vecs, q


def scan(cls, vecs, queries, gt, dim, k, efs, rounds=3):
    print(f"\n--- {cls.__name__} ---")
    b = cls(dim=dim, k_nn=k)
    try:
        b.reset()
        b.create_index()
        b.write(vecs)
        time.sleep(2)
        for ef in efs:
            lat = []
            rc = []
            for _ in range(rounds):
                for i, q in enumerate(queries):
                    pks, dt = b.query_once(q, ef=ef)
                    lat.append(dt)
                    rc.append(len(set(pks) & set(gt[i].tolist())) / k)
            print(f"  ef={ef:<5} p50={statistics.median(lat):6.1f}ms "
                  f"recall={statistics.mean(rc):.4f}")
    finally:
        b.drop()
        b.close()


def main():
    dim = 768
    k = 10
    n = 10000
    efs = [32, 64, 128, 256, 512]

    for label, gen in (("uniform", "uniform"), ("clustered", "clustered")):
        print(f"\n{'=' * 60}\n{label} data  n={n} dim={dim}\n{'=' * 60}")
        if gen == "uniform":
            vecs, qs, _ = make_dataset(n, dim)
        else:
            vecs, qs = make_clustered(n, dim)
        gt = ground_truth(vecs, qs, k)
        for cls in (MilvusBench, SurrealBench):
            scan(cls, vecs, qs, gt, dim, k, efs)


if __name__ == "__main__":
    main()
