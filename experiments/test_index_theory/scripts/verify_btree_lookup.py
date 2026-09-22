# -*- coding: utf-8 -*-
"""
B+树点查延迟的介质常数标定

B+树点查延迟 = 树高(IO 次数) x 每次 IO 的延迟常数。
树高由 btree_model 精确给出（IO 次数 = height），本脚本标定"每次 IO 延迟"这个介质常数。

为什么不用 sqlite 实测点查延迟：sqlite 点查被 SQL 解析与 Python 驱动开销主导（微秒级），
而树高在 vlen=100 时只有 3 层、纯树遍历是纳秒级，信号被淹没。所以这里直接标定介质常数。

用指针追踪（pointer chasing）微基准测本机内存随机访问延迟：在超出 L3 的大缓冲里
沿随机指针链跳转，每步是一次 cache miss 的内存随机访问，对应 DB 缓冲池命中时的树遍历。
小数组（L1 内）追踪作对照，扣除 Python 循环开销，得到净内存随机访问延迟。

结果写入 results/btree_latency_<timestamp>.json，给出各 n 的点查延迟换算表。
"""
import os
import sys
import json
import time
import random
from array import array
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from btree_model import BTreeSpec, compute

# 缓冲池命中场景（热数据，内存随机访问）。磁盘 IO 作为边界在文档里给量级。
SPEC = BTreeSpec(key_size=4, value_size=100, page_size=4096,
                 page_overhead=100, fill_factor=0.84)  # fill 用随机插入标定值
N_LIST = [10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]

STEPS = 2_000_000
SEED = 42


def chase(size_bytes, page=4096, steps=STEPS):
    """指针追踪：返回每步纳秒数。大缓冲->cache miss，小缓冲->缓存命中。"""
    n_pages = size_bytes // page
    perm = list(range(n_pages))
    random.seed(SEED)
    random.shuffle(perm)
    ptr = array("q", perm)  # int64 C 数组，双射无短环
    idx = 0
    # warmup，让分配落地、分支预测稳定
    for _ in range(min(steps, 100_000)):
        idx = ptr[idx]
    start = time.perf_counter_ns()
    for _ in range(steps):
        idx = ptr[idx]
    elapsed = time.perf_counter_ns() - start
    return elapsed / steps  # ns/step


def benchmark():
    """小数组(缓存命中)对照 + 逐级增大缓冲，找 cache miss 的平台区。"""
    # 小数组：32 KiB，在 L1/L2 内，测得近似 Python 循环开销
    small = chase(32 * 1024)
    results = {"python_overhead_ns": small, "sizes": []}
    print(f"Python 循环开销(32KiB, 缓存命中): {small:.2f} ns/step")
    for mb in [1, 8, 64, 256]:
        size = mb * 1024 * 1024
        ns = chase(size)
        net = ns - small  # 扣除 Python 开销
        results["sizes"].append({"mb": mb, "raw_ns": ns, "net_ns": net})
        print(f"{mb:>4} MiB 缓冲: {ns:.2f} ns/step  净内存随机访问: {net:.2f} ns")
    # 取最大缓冲(256MiB，稳出 L3)的净值作为内存随机访问常数
    mem_ns = results["sizes"][-1]["net_ns"]
    results["mem_random_access_ns"] = mem_ns
    return results, mem_ns


def main():
    bench, mem_ns = benchmark()
    print(f"\n内存随机访问延迟常数: {mem_ns:.2f} ns（缓冲池命中时的每次树 IO）")

    # 点查延迟换算表：树高 x 内存随机访问延迟
    table = []
    for n in N_LIST:
        m = compute(SPEC, n)
        # 树遍历 = height 次内存随机访问；另加少量固定开销（CPU 比较键等）
        latency_ns = m.height * mem_ns
        table.append({
            "n": n,
            "height": m.height,
            "point_io": m.point_io,
            "lookup_ns_hot": round(latency_ns, 1),
        })
        print(f"n={n:>13,}  height={m.height}  point_io={m.point_io}  "
              f"热数据点查≈{latency_ns:.0f} ns ({latency_ns/1000:.2f} µs)")

    result = {
        "meta": {
            "steps": STEPS,
            "python": sys.version.split()[0],
            "spec_fill_factor": SPEC.fill_factor,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "note": "内存随机访问常数，对应 DB 缓冲池命中(热数据)场景；磁盘 IO 另给量级",
        },
        "benchmark": bench,
        "mem_random_access_ns": mem_ns,
        "lookup_table": table,
    }
    out = os.path.join(os.path.dirname(__file__), "..", "results",
                       f"btree_latency_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n结果写入 {out}")


if __name__ == "__main__":
    main()
