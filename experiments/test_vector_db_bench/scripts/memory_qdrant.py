# -*- coding: utf-8 -*-
"""Qdrant 索引级内存（全 SPECS）：用 docker stats 测容器内存增量。

Qdrant 有 6 种索引配置（见 index_bench.py QdrantAdapter.SPECS）：
  HNSW_m16 / HNSW_m32 / FLAT / HNSW_ScalarI8 / HNSW_PQ / HNSW_Binary

本脚本用 QdrantAdapter 建 collection（支持全部量化配置），
用 docker stats 多次采样取中位数，减少瞬时波动。

用法:
  ./.venv/Scripts/python.exe scripts/memory_qdrant.py --sizes 50000 100000
  ./.venv/Scripts/python.exe scripts/memory_qdrant.py --specs HNSW_m16 HNSW_PQ HNSW_Binary --sizes 100000
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
QDRANT_CONTAINER = "qdrant"
sys.path.insert(0, str(ROOT / "scripts"))

from index_bench import QdrantAdapter, load_real  # noqa: E402


def docker_stats_mb():
    """Qdrant 容器内存（MB）。"""
    try:
        out = subprocess.check_output(
            ["docker", "stats", "--no-stream", "--format",
             "{{.MemUsage}}", QDRANT_CONTAINER],
            timeout=10, stderr=subprocess.DEVNULL
        ).decode().strip()
        used = out.split("/")[0].strip()
        if "GiB" in used:
            return float(used.replace("GiB", "").strip()) * 1024
        elif "MiB" in used:
            return float(used.replace("MiB", "").strip())
        elif "KiB" in used:
            return float(used.replace("KiB", "").strip()) / 1024
    except Exception:
        pass
    return None


def sample_container_mem(samples=3, interval=2):
    """多次采样取中位数，减少瞬时波动。"""
    vals = []
    for _ in range(samples):
        v = docker_stats_mb()
        if v is not None:
            vals.append(v)
        time.sleep(interval)
    if not vals:
        return None
    vals.sort()
    return vals[len(vals) // 2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[50000, 100000])
    parser.add_argument("--specs", nargs="+", default=None,
                        help="要测的索引配置，默认全部 SPECS")
    args = parser.parse_args()

    all_specs = list(QdrantAdapter.SPECS.keys())
    specs = args.specs or all_specs
    specs = [s for s in specs if s in all_specs]
    print("Qdrant SPECS: %s" % specs, flush=True)

    all_vecs, _, dim = load_real()
    print("dataset: %d x %d" % (len(all_vecs), dim), flush=True)

    results = {}
    for spec in specs:
        spec_results = []
        for size in args.sizes:
            print("  %s @ %d ..." % (spec, size), flush=True)

            # 用 QdrantAdapter 建 collection（支持量化配置）
            a = QdrantAdapter(dim=dim, ns="membench3")
            a.setup(spec)

            # 采样基线（空 collection）
            before = sample_container_mem(samples=3, interval=2)

            # 加载向量
            if size <= len(all_vecs):
                vecs = all_vecs[:size]
            else:
                rng = np.random.default_rng(42)
                extra = rng.standard_normal((size - len(all_vecs), dim)).astype("float32")
                vecs = np.vstack([all_vecs, extra])[:size]

            # 写入
            a.write(vecs)

            # 等索引构建完成 + 内存稳定
            time.sleep(5)
            after = sample_container_mem(samples=3, interval=2)

            delta = (after - before) if (before is not None and after is not None) else None
            r = {
                "spec": spec,
                "n": size,
                "before_mb": round(before, 1) if before else None,
                "after_mb": round(after, 1) if after else None,
                "delta_mb": round(delta, 1) if delta is not None else None,
            }
            spec_results.append(r)
            print("    total=%sMB (before=%.1f after=%.1f)" % (
                "%+.1f" % delta if delta is not None else "N/A",
                before or 0, after or 0), flush=True)

            # 清理
            try:
                a.teardown()
            except Exception:
                pass
            time.sleep(2)

        results[spec] = spec_results

    out = RESULTS / ("membench-qdrant-%s.json" % time.strftime("%Y%m%d-%H%M%S"))
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print("\nsaved:", out)


if __name__ == "__main__":
    main()
