# -*- coding: utf-8 -*-
"""向量库内存占用扩展基准：按索引类型 × 数据规模两维测量。

嵌入式库测进程 RSS 增量，Docker 库用 docker stats。

用法:
  ./.venv/Scripts/python.exe scripts/memory_bench2.py
  ./.venv/Scripts/python.exe scripts/memory_bench2.py --only lance milvus qdrant
  ./.venv/Scripts/python.exe scripts/memory_bench2.py --sizes 1000 5000 10000 50000 100000
"""
import argparse
import json
import subprocess
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import psutil

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

MILVUS_CONTAINER = "milvus-standalone"
QDRANT_CONTAINER = "qdrant"


def rss_mb():
    return psutil.Process().memory_info().rss / 1e6


def docker_stats_mb(container):
    try:
        out = subprocess.check_output(
            ["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", container],
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


def measure_embedded_index(name, adapter_cls, vecs, dim, spec):
    """嵌入式库：测指定索引在 setup/write/build_index 的 RSS 增量。"""
    a = adapter_cls(dim=dim, ns="membench2")
    try:
        gc_before = rss_mb()
        a.setup(spec)
        after_setup = rss_mb()
        a.write(vecs)
        after_write = rss_mb()
        a.build_index()
        after_index = rss_mb()
        return {
            "spec": spec,
            "setup_delta": round(after_setup - gc_before, 1),
            "write_delta": round(after_write - after_setup, 1),
            "index_delta": round(after_index - after_write, 1),
            "total_delta": round(after_index - gc_before, 1),
        }
    finally:
        try:
            a.teardown()
        except Exception:
            pass


def sample_docker_mem(container, samples=3, interval=3):
    """多次采样取中位数，减少瞬时波动。"""
    vals = []
    for _ in range(samples):
        v = docker_stats_mb(container)
        if v is not None:
            vals.append(v)
        time.sleep(interval)
    if not vals:
        return None
    vals.sort()
    return vals[len(vals) // 2]


def measure_docker_index(name, container, adapter_cls, vecs, dim, spec):
    """Docker 库：测指定索引的容器内存增量（多次采样取中位数）。"""
    a = adapter_cls(dim=dim, ns="membench2")
    try:
        before = sample_docker_mem(container, samples=3, interval=3)
        a.setup(spec)
        a.write(vecs)
        a.build_index()
        time.sleep(10)  # 等容器内存稳定（Milvus 建索引后有 GC 行为）
        after = sample_docker_mem(container, samples=3, interval=3)
        delta = (after - before) if (before is not None and after is not None) else None
        return {
            "spec": spec,
            "before_mb": round(before, 1) if before else None,
            "after_mb": round(after, 1) if after else None,
            "delta_mb": round(delta, 1) if delta is not None else None,
        }
    finally:
        try:
            a.teardown()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", default=None)
    parser.add_argument("--sizes", nargs="+", type=int,
                        default=[1000, 5000, 11293, 50000, 100000])
    args = parser.parse_args()

    from index_bench import (MilvusAdapter, QdrantAdapter, LanceAdapter,
                             SqliteVecAdapter, ChromaAdapter, load_real)
    import gen_extra_vecs

    all_vecs, _, dim = load_real()
    print("dataset: %d x %d" % (all_vecs.shape[0], dim), flush=True)

    targets = args.only or ["milvus", "qdrant", "lance", "sqlite", "chroma"]
    adapter_map = {
        "milvus": (MilvusAdapter, MILVUS_CONTAINER),
        "qdrant": (QdrantAdapter, QDRANT_CONTAINER),
        "lance": (LanceAdapter, None),
        "sqlite": (SqliteVecAdapter, None),
        "chroma": (ChromaAdapter, None),
    }

    all_results = {}

    for name in targets:
        if name not in adapter_map:
            continue
        cls, container = adapter_map[name]
        specs = list(cls.SPECS.keys())
        print("\n%s [%s] specs: %s" % ("=" * 40, name, specs), flush=True)

        db_results = {"type": "docker" if container else "embedded", "specs": {}}

        for spec in specs:
            spec_results = []
            for size in args.sizes:
                if size > len(all_vecs):
                    # 生成随机向量补充
                    extra = gen_extra_vecs.gen(size - len(all_vecs), dim)
                    vecs = np.vstack([all_vecs, extra])[:size]
                else:
                    vecs = all_vecs[:size]

                print("  %s @ %d vectors..." % (spec, size), flush=True)
                if container:
                    r = measure_docker_index(name, container, cls, vecs, dim, spec)
                else:
                    r = measure_embedded_index(name, cls, vecs, dim, spec)
                r["n"] = size
                spec_results.append(r)
                print("    -> total=%.1fMB" % r.get("total_delta", r.get("delta_mb", 0)), flush=True)

            db_results["specs"][spec] = spec_results

        all_results[name] = db_results

    out = RESULTS / ("membench2-index-scale-%s.json" % time.strftime("%Y%m%d-%H%M%S"))
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(all_results, ensure_ascii=False, indent=2))
    print("\nsaved:", out)


if __name__ == "__main__":
    main()
