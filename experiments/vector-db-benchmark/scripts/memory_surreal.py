# -*- coding: utf-8 -*-
"""SurrealDB 索引级内存：用 Windows WorkingSet API 测 SurrealDB 进程内存。

SurrealDB 是独立进程，无法用 psutil 测 Python 子进程。
用 ctypes 调 GetProcessMemoryInfo 拿 WorkingSetSize（任务管理器同款口径）。

用法:
  ./.venv/Scripts/python.exe scripts/memory_surreal.py --sizes 50000 100000
"""
import argparse
import ctypes
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT / "scripts"))

from index_bench import load_real  # noqa: E402

SURREAL_PORT = 8000
SURREAL_BIN = ROOT / "bin" / "surreal.exe"
SURREAL_DATA = ROOT / "data" / "surreal_bench.db"


def find_surreal_pid():
    """找监听 8000 端口的 SurrealDB 进程 PID。"""
    out = subprocess.check_output(
        ["netstat", "-ano"], text=True, timeout=10
    )
    for line in out.splitlines():
        if ":%d " % SURREAL_PORT in line and "LISTENING" in line:
            return int(line.split()[-1])
    return None


def get_working_set_mb(pid):
    """Windows WorkingSetSize（MB），任务管理器「内存」列同款口径。"""
    PROCESS_QUERY_INFORMATION = 0x0400
    PROCESS_VM_READ = 0x0010
    handle = ctypes.windll.psapi.GetProcessMemoryInfo
    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, pid)
    if not h:
        return None
    try:
        class PMC(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        ok = handle(h, ctypes.byref(pmc), ctypes.sizeof(PMC))
        if not ok:
            return None
        return pmc.WorkingSetSize / 1e6
    finally:
        k32.CloseHandle(h)


def surreal_mem_mb():
    pid = find_surreal_pid()
    if pid is None:
        return None
    return get_working_set_mb(pid)


def setup_surreal(spec_name, dim):
    """通过 surrealdb Python SDK 建表+建索引，返回 (db, coll) 信息。"""
    from surrealdb import Surreal
    db = Surreal("ws://127.0.0.1:8000/rpc")
    db.signin({"username": "root", "password": "root"})
    ns = "membench3"
    db.use(ns, "main")
    coll = "vec_%s" % spec_name.lower()
    # 清理
    db.query("REMOVE TABLE IF EXISTS %s;" % coll)
    db.query("REMOVE INDEX IF EXISTS idx ON %s;" % coll)
    # 建表 + 向量字段
    db.query("DEFINE TABLE %s SCHEMALESS;" % coll)
    db.query("DEFINE FIELD emb ON %s TYPE array<float>;" % coll)
    # 建索引（SurrealDB 只支持 HNSW）
    if spec_name == "HNSW":
        db.query("DEFINE INDEX idx ON %s FIELDS emb HNSW DIMENSION %d DIST EUCLIDEAN;" % (coll, dim))
    return db, coll


def insert_vecs(db, coll, vecs):
    """批量插入向量。"""
    n = len(vecs)
    # SurrealDB 的 INSERT 一次不要太多，分批
    batch = 1000
    for s in range(0, n, batch):
        e = min(s + batch, n)
        rows = []
        for i in range(s, e):
            rows.append({
                "pk": i,
                "emb": vecs[i].tolist(),
            })
        db.insert(coll, rows)
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[50000, 100000])
    parser.add_argument("--only", nargs="+", default=["HNSW"])
    args = parser.parse_args()

    # SurrealDB 只支持 HNSW 一种向量索引
    specs = [s for s in args.only if s in ("HNSW", "MTREE")]

    results = {}
    for spec in specs:
        spec_results = []
        for size in args.sizes:
            print("  %s @ %d ..." % (spec, size), flush=True)
            # 重新连接建表
            db, coll = setup_surreal(spec, 384)
            time.sleep(1)  # 等索引定义生效
            before = surreal_mem_mb()
            # 加载向量
            all_vecs, _, d = load_real()
            if size <= len(all_vecs):
                vecs = all_vecs[:size]
            else:
                # 生成额外向量
                rng = np.random.default_rng(42)
                extra = rng.standard_normal((size - len(all_vecs), d)).astype("float32")
                vecs = np.vstack([all_vecs, extra])[:size]
            # 写入
            insert_vecs(db, coll, vecs)
            time.sleep(2)  # 等索引构建 + 内存稳定
            after = surreal_mem_mb()
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
                db.query("REMOVE TABLE IF EXISTS %s;" % coll)
            except Exception:
                pass
            time.sleep(1)
        results[spec] = spec_results

    out = RESULTS / ("membench-surreal-%s.json" % time.strftime("%Y%m%d-%H%M%S"))
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print("\nsaved:", out)


if __name__ == "__main__":
    main()
