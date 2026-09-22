# -*- coding: utf-8 -*-
"""向量库内存占用基准：嵌入式库测进程 RSS 增量，Docker 库用 docker stats。

用法:
  ./.venv/Scripts/python.exe scripts/memory_bench.py
  ./.venv/Scripts/python.exe scripts/memory_bench.py --only lance sqlite chroma
"""
import argparse
import importlib
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
    """当前进程 RSS（MB）。"""
    return psutil.Process().memory_info().rss / 1e6


def docker_stats_mb(container):
    """Docker 容器内存占用（MB），失败返回 None。"""
    try:
        out = subprocess.check_output(
            ["docker", "stats", "--no-stream", "--format",
             "{{.MemUsage}}", container],
            timeout=10, stderr=subprocess.DEVNULL
        ).decode().strip()
        # 格式如 "1.234GiB / 60GiB" 或 "512MiB / 60GiB"
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


def measure_embedded(name, adapter_cls, vecs, dim):
    """嵌入式库：测进程 RSS 在 setup/write/build_index 三个阶段的增量。"""
    print("\n%s [%s]" % ("-" * 24, name), flush=True)
    a = adapter_cls(dim=dim, ns="membench")
    try:
        gc_before = rss_mb()
        a.setup(list(adapter_cls.SPECS.keys())[0])
        after_setup = rss_mb()
        a.write(vecs)
        after_write = rss_mb()
        a.build_index()
        after_index = rss_mb()
        result = {
            "db": name,
            "type": "embedded",
            "gc_before_mb": round(gc_before, 1),
            "after_setup_mb": round(after_setup, 1),
            "after_write_mb": round(after_write, 1),
            "after_index_mb": round(after_index, 1),
            "setup_delta_mb": round(after_setup - gc_before, 1),
            "write_delta_mb": round(after_write - after_setup, 1),
            "index_delta_mb": round(after_index - after_write, 1),
            "total_delta_mb": round(after_index - gc_before, 1),
        }
        print("  setup: +%.1fMB  write: +%.1fMB  index: +%.1fMB  total: +%.1fMB"
              % (result["setup_delta_mb"], result["write_delta_mb"],
                 result["index_delta_mb"], result["total_delta_mb"]), flush=True)
        return result
    finally:
        try:
            a.teardown()
        except Exception:
            pass


def measure_docker(name, container, adapter_cls, vecs, dim):
    """Docker 库：用 docker stats 测容器内存。"""
    print("\n%s [%s]" % ("-" * 24, name), flush=True)
    a = adapter_cls(dim=dim, ns="membench")
    try:
        before = docker_stats_mb(container)
        a.setup(list(adapter_cls.SPECS.keys())[0])
        a.write(vecs)
        a.build_index()
        time.sleep(2)  # 等容器内存稳定
        after = docker_stats_mb(container)
        delta = (after - before) if (before is not None and after is not None) else None
        result = {
            "db": name,
            "type": "docker",
            "container": container,
            "before_mb": round(before, 1) if before else None,
            "after_mb": round(after, 1) if after else None,
            "delta_mb": round(delta, 1) if delta is not None else None,
        }
        print("  container %s: before=%.1fMB after=%.1fMB delta=%s"
              % (container, before or 0, after or 0,
                 "%+.1fMB" % delta if delta is not None else "N/A"), flush=True)
        return result
    finally:
        try:
            a.teardown()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", default=None)
    args = parser.parse_args()

    from index_bench import (MilvusAdapter, QdrantAdapter, LanceAdapter,
                             SqliteVecAdapter, ChromaAdapter, load_real)

    vecs, queries, dim = load_real()
    print("dataset: %d x %d, mode=real-20news" % (vecs.shape[0], dim), flush=True)

    targets = args.only or ["milvus", "qdrant", "lance", "sqlite", "chroma"]
    results = []

    adapter_map = {
        "milvus": (MilvusAdapter, MILVUS_CONTAINER),
        "qdrant": (QdrantAdapter, QDRANT_CONTAINER),
        "lance": (LanceAdapter, None),
        "sqlite": (SqliteVecAdapter, None),
        "chroma": (ChromaAdapter, None),
    }

    for name in targets:
        if name not in adapter_map:
            print("skip unknown:", name)
            continue
        cls, container = adapter_map[name]
        if container:
            r = measure_docker(name, container, cls, vecs, dim)
        else:
            r = measure_embedded(name, cls, vecs, dim)
        results.append(r)

    out = RESULTS / ("membench-real-20news-%s.json" % time.strftime("%Y%m%d-%H%M%S"))
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print("\nsaved:", out)


if __name__ == "__main__":
    main()
