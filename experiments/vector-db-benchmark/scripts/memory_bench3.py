# -*- coding: utf-8 -*-
"""向量库内存占用 v3：每轮独立子进程，测绝对内存，消除跨轮次污染。

每轮启动一个子进程测一个 (库, 索引, 规模) 组合，测完后进程退出，
OS 回收全部内存。父进程收集结果。

- 嵌入式库（lance / sqlite / chroma）：测 Python 进程 RSS 增量
- Docker 库（milvus / qdrant）：测容器内存增量，每轮前 drop 全部已有集合

用法:
  ./.venv/Scripts/python.exe scripts/memory_bench3.py
  ./.venv/Scripts/python.exe scripts/memory_bench3.py --only lance milvus --sizes 50000 100000
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"

CHILD_SCRIPT = r'''
import sys, json, gc, time, subprocess
sys.path.insert(0, "scripts")
import numpy as np
import psutil

db, spec, n, dim = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])

from index_bench import (MilvusAdapter, QdrantAdapter, LanceAdapter,
                         SqliteVecAdapter, ChromaAdapter, load_real)
from gen_extra_vecs import gen

# docker 库 -> 容器名列表（服务容器 + 依赖容器）
DOCKER_CONTAINERS = {
    "milvus": ["milvus-standalone", "milvus-etcd", "milvus-minio"],
    "qdrant": ["qdrant"],
}

def docker_mb(containers):
    """容器总内存（MB），失败返回 None。"""
    total = 0.0
    for c in containers:
        try:
            out = subprocess.check_output(
                ["docker", "stats", "--no-stream", "--format",
                 "{{.MemUsage}}", c],
                timeout=15, stderr=subprocess.DEVNULL
            ).decode().strip()
            used = out.split("/")[0].strip()
            if "GiB" in used:
                total += float(used.replace("GiB", "").strip()) * 1024
            elif "MiB" in used:
                total += float(used.replace("MiB", "").strip())
            elif "KiB" in used:
                total += float(used.replace("KiB", "").strip()) / 1024
        except Exception:
            return None
    return total

def drop_all(adapter):
    """drop 该库所有已存在的集合，等容器内存释放。"""
    try:
        if hasattr(adapter, "c"):
            for c in adapter.c.list_collections():
                adapter.c.drop_collection(c)
    except Exception:
        pass

adapter_map = {
    "milvus": MilvusAdapter, "qdrant": QdrantAdapter,
    "lance": LanceAdapter, "sqlite": SqliteVecAdapter, "chroma": ChromaAdapter,
}

all_vecs, _, d = load_real()
if n <= len(all_vecs):
    vecs = all_vecs[:n]
else:
    extra = gen(n - len(all_vecs), d)
    vecs = np.vstack([all_vecs, extra])[:n]

cls = adapter_map[db]
is_docker = db in DOCKER_CONTAINERS

if is_docker:
    a = cls(dim=dim, ns="membench3")
    containers = DOCKER_CONTAINERS[db]
    drop_all(a)
    time.sleep(5)
    before = docker_mb(containers)
    a.setup(spec)
    time.sleep(2)
    after_setup = docker_mb(containers)
    a.write(vecs)
    time.sleep(3)
    after_write = docker_mb(containers)
    a.build_index()
    time.sleep(5)
    after_index = docker_mb(containers)
else:
    a = cls(dim=dim, ns="membench3")
    gc.collect()
    before = psutil.Process().memory_info().rss / 1e6
    a.setup(spec)
    gc.collect()
    after_setup = psutil.Process().memory_info().rss / 1e6
    a.write(vecs)
    gc.collect()
    after_write = psutil.Process().memory_info().rss / 1e6
    a.build_index()
    gc.collect()
    time.sleep(1)
    after_index = psutil.Process().memory_info().rss / 1e6

result = {
    "db": db, "spec": spec, "n": n, "type": "docker" if is_docker else "embedded",
    "before_mb": round(before, 1),
    "after_setup_mb": round(after_setup, 1),
    "after_write_mb": round(after_write, 1),
    "after_index_mb": round(after_index, 1),
    "setup_delta": round(after_setup - before, 1),
    "write_delta": round(after_write - after_setup, 1),
    "index_delta": round(after_index - after_write, 1),
    "total_delta": round(after_index - before, 1),
}
print("RESULT:" + json.dumps(result))
'''


def run_child(db, spec, n, dim=384):
    """在子进程里跑一个组合，返回结果 dict。"""
    proc = subprocess.run(
        [str(PYTHON), "-c", CHILD_SCRIPT, db, spec, str(n), str(dim)],
        capture_output=True, text=True, timeout=600,
        cwd=str(ROOT),
    )
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT:"):
            return json.loads(line[7:])
    if proc.returncode != 0:
        print("  CHILD FAILED: %s" % proc.stderr[-300:], flush=True)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", default=None)
    parser.add_argument("--sizes", nargs="+", type=int, default=[50000, 100000])
    args = parser.parse_args()

    from index_bench import (MilvusAdapter, QdrantAdapter, LanceAdapter,
                             SqliteVecAdapter, ChromaAdapter, load_real)

    targets = args.only or ["lance", "milvus"]
    adapter_map = {
        "milvus": MilvusAdapter, "qdrant": QdrantAdapter,
        "lance": LanceAdapter, "sqlite": SqliteVecAdapter, "chroma": ChromaAdapter,
    }

    all_results = {}
    for name in targets:
        if name not in adapter_map:
            continue
        cls = adapter_map[name]
        specs = list(cls.SPECS.keys())
        print("\n%s [%s] specs: %s" % ("=" * 40, name, specs), flush=True)
        db_results = {}
        for spec in specs:
            spec_results = []
            for size in args.sizes:
                print("  %s @ %d ..." % (spec, size), flush=True)
                r = run_child(name, spec, size)
                if r:
                    spec_results.append(r)
                    print("    total=%+.1fMB (setup=%+.1f write=%+.1f index=%+.1f)" % (
                        r["total_delta"], r["setup_delta"],
                        r["write_delta"], r["index_delta"]), flush=True)
            db_results[spec] = spec_results
        all_results[name] = db_results

    out = RESULTS / ("membench3-index-scale-%s.json" % time.strftime("%Y%m%d-%H%M%S"))
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(all_results, ensure_ascii=False, indent=2))
    print("\nsaved:", out)


if __name__ == "__main__":
    main()
