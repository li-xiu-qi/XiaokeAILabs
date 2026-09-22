# -*- coding: utf-8 -*-
"""
树状数组（Fenwick Tree）实测验证

用纯 Python 自制迷你树状数组（1 开始下标，tree[i] 存区间 (i-lowbit(i), i] 的和），实测：
1. 单点更新延迟与迭代次数
2. 前缀和查询延迟与迭代次数
3. 区间和查询延迟（prefix(r) - prefix(l-1)）
4. 正确性：与暴力真值逐区间核对
5. 同 n 下与线段树的延迟与内存对比

延迟的理论换算是"节点常数"法：先由建树标定"每访问一个节点的纳秒数"，
再用 迭代次数 × 常数 预测各操作延迟，与实测对比。

结果写入 results/fenwick_tree_<timestamp>.json
"""
import os
import sys
import json
import random
import time
from array import array
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from fenwick_tree_model import (
    FenwickTreeSpec, compute, simulate_prefix_iters, simulate_update_iters,
)
from verify_segment_tree import SegmentTree

N_LIST = [1_000, 10_000, 100_000, 1_000_000]
SEED = 42


class FenwickTree:
    """自制迷你树状数组。1 开始下标，tree[i] 存区间 (i - lowbit(i), i] 的和。"""

    def __init__(self, n: int):
        self.n = n
        self.tree = array('q', bytes(8 * (n + 1)))

    @staticmethod
    def _lowbit(i: int) -> int:
        return i & (-i)

    def build(self, data):
        """建树。逐个单点更新，O(n log n)。

        也可以 O(n) 建树（先填 tree[i]=a[i]，再 tree[i] += tree[i-lowbit(i)]），
        这里用简单写法，测量的是查询与更新的常数，不是建树。
        """
        for i, v in enumerate(data, start=1):
            self.add(i, v)

    def add(self, i: int, delta: int):
        """单点更新：位置 i 加 delta。反复 i += lowbit(i) 直到越界。"""
        t = self.tree
        n = self.n
        while i <= n:
            t[i] += delta
            i += i & (-i)

    def prefix(self, i: int) -> int:
        """前缀和：a[1..i] 的和。反复 i -= lowbit(i) 直到 0。"""
        t = self.tree
        s = 0
        while i > 0:
            s += t[i]
            i -= i & (-i)
        return s

    def range_query(self, l: int, r: int) -> int:
        """区间和 [l, r]（闭区间，1 开始）。= prefix(r) - prefix(l-1)。"""
        return self.prefix(r) - self.prefix(l - 1)

    def count_prefix_iters(self, i: int) -> int:
        """只数前缀和查询的迭代次数，用于核对理论。"""
        steps = 0
        while i > 0:
            i -= i & (-i)
            steps += 1
        return steps

    def storage_bytes(self) -> int:
        """实际存储占用。array 的缓冲区字节数。"""
        return self.tree.buffer_info()[1] * self.tree.itemsize


def time_ns(fn, repeats: int) -> float:
    """测 fn 的平均耗时（纳秒）。先 warmup 再计时。"""
    for _ in range(min(repeats // 10 + 1, 100)):
        fn()
    start = time.perf_counter_ns()
    for _ in range(repeats):
        fn()
    return (time.perf_counter_ns() - start) / repeats


def make_data(n: int):
    rng = random.Random(SEED)
    return [rng.randrange(0, 10_000) for _ in range(n)]


def measure_build(fw: FenwickTree, spec: FenwickTreeSpec) -> dict:
    """实测建树，并标定"每访问一个节点的纳秒数"。"""
    data = make_data(spec.n)
    start = time.perf_counter_ns()
    fw.build(data)
    elapsed_ns = time.perf_counter_ns() - start
    # 建树做 n 次 add，每次 add 平均约 (1/2)log2(n) 次迭代
    iters = compute(spec).update_iters
    per_node_ns = elapsed_ns / (spec.n * iters)
    return {
        "build_ms": round(elapsed_ns / 1e6, 3),
        "per_node_ns": per_node_ns,
    }


def measure_update(fw: FenwickTree, spec: FenwickTreeSpec, per_node_ns: float) -> dict:
    """实测单点更新。"""
    rng = random.Random(SEED + 1)
    indices = [rng.randrange(1, spec.n + 1) for _ in range(2000)]

    actual_ns = time_ns(lambda: [fw.add(i, 1) for i in indices[:200]], 5)

    # 实际迭代次数（同一批下标）
    sample = indices[:500]
    actual_iters = sum(simulate_update_iters(i, spec.n) for i in sample) / len(sample)

    m = compute(spec)
    return {
        "actual_ns_per_op": round(actual_ns / 200, 1),
        "iters_actual": round(actual_iters, 2),
        "iters_theory": round(m.update_iters, 2),
        "iters_worst": m.worst_iters,
        "predicted_ns_per_op": round(m.update_iters * per_node_ns, 1),
    }


def measure_prefix(fw: FenwickTree, spec: FenwickTreeSpec, per_node_ns: float) -> dict:
    """实测前缀和查询。"""
    rng = random.Random(SEED + 2)
    indices = [rng.randrange(1, spec.n + 1) for _ in range(2000)]

    actual_ns = time_ns(lambda: [fw.prefix(i) for i in indices[:200]], 5)

    sample = indices[:500]
    actual_iters = sum(fw.count_prefix_iters(i) for i in sample) / len(sample)

    m = compute(spec)
    return {
        "actual_ns_per_op": round(actual_ns / 200, 1),
        "iters_actual": round(actual_iters, 2),
        "iters_theory": round(m.prefix_iters, 2),
        "iters_worst": m.worst_iters,
        "predicted_ns_per_op": round(m.prefix_iters * per_node_ns, 1),
    }


def measure_range(fw: FenwickTree, spec: FenwickTreeSpec, per_node_ns: float) -> dict:
    """实测区间和查询。"""
    rng = random.Random(SEED + 3)
    ranges = []
    for _ in range(2000):
        l = rng.randrange(1, spec.n + 1)
        r = rng.randrange(l, spec.n + 1)
        ranges.append((l, r))

    actual_ns = time_ns(lambda: [fw.range_query(l, r) for l, r in ranges[:200]], 5)

    m = compute(spec)
    return {
        "actual_ns_per_op": round(actual_ns / 200, 1),
        "iters_theory": round(m.range_query_iters, 2),
        "predicted_ns_per_op": round(m.range_query_iters * per_node_ns, 1),
    }


def verify_correctness(spec: FenwickTreeSpec) -> dict:
    """与暴力真值核对。随机区间上逐一比对。"""
    n = spec.n
    data = make_data(n)
    fw = FenwickTree(n)
    fw.build(data)
    rng = random.Random(SEED + 5)

    checked = 0
    mismatches = 0
    for _ in range(300):
        l = rng.randrange(1, n + 1)
        r = rng.randrange(l, n + 1)
        checked += 1
        if fw.range_query(l, r) != sum(data[l - 1:r]):
            mismatches += 1

    # 单点更新后再核对一次（验证更新路径）
    for _ in range(200):
        i = rng.randrange(1, n + 1)
        delta = rng.randrange(-100, 100)
        fw.add(i, delta)
        data[i - 1] += delta
    checked2 = 0
    mismatches2 = 0
    for _ in range(300):
        l = rng.randrange(1, n + 1)
        r = rng.randrange(l, n + 1)
        checked2 += 1
        if fw.range_query(l, r) != sum(data[l - 1:r]):
            mismatches2 += 1

    return {
        "build_checked": checked, "build_mismatches": mismatches,
        "after_update_checked": checked2, "after_update_mismatches": mismatches2,
    }


def main():
    print("=== 树状数组实测验证 ===\n")

    rows = []
    for n in N_LIST:
        spec = FenwickTreeSpec(n=n)
        print(f"--- n={n:,} ---")

        fw = FenwickTree(n)
        build = measure_build(fw, spec)

        upd = measure_update(fw, spec, build["per_node_ns"])
        pre = measure_prefix(fw, spec, build["per_node_ns"])
        rng_q = measure_range(fw, spec, build["per_node_ns"])
        corr = verify_correctness(spec)

        # 同 n 线段树对照
        st = SegmentTree(n)
        st.build(make_data(n))
        seg_point = time_ns(lambda: st.point_update(0, 0), 2000)
        seg_ranges = [(0, n // 2)]
        seg_range_ns = time_ns(lambda: [st.range_query(l, r) for l, r in seg_ranges], 2000)
        seg_mem = st.storage_bytes()
        fw_mem = fw.storage_bytes()

        m = compute(spec)
        print(f"  建树: {build['build_ms']:.2f} ms  每节点 {build['per_node_ns']:.2f} ns")
        print(f"  单点更新: 实测 {upd['actual_ns_per_op']:.0f} ns  迭代 "
              f"实测 {upd['iters_actual']:.1f} / 理论 {upd['iters_theory']:.1f} "
              f"(最坏 {upd['iters_worst']})")
        print(f"  前缀和:   实测 {pre['actual_ns_per_op']:.0f} ns  迭代 "
              f"实测 {pre['iters_actual']:.1f} / 理论 {pre['iters_theory']:.1f}")
        print(f"  区间和:   实测 {rng_q['actual_ns_per_op']:.0f} ns  迭代理论 {rng_q['iters_theory']:.1f}")
        print(f"  正确性: 建后不一致 {corr['build_mismatches']}/{corr['build_checked']}，"
              f"更新后不一致 {corr['after_update_mismatches']}/{corr['after_update_checked']}")
        print(f"  内存: 树状数组 {fw_mem/1024/1024:.2f} MiB vs 线段树 {seg_mem/1024/1024:.2f} MiB "
              f"(比值 {seg_mem/fw_mem:.1f}x)")
        print(f"  单点更新延迟: 树状数组 {upd['actual_ns_per_op']:.0f} ns vs "
              f"线段树 {seg_point:.0f} ns")

        rows.append({
            "n": n,
            "build": build,
            "point_update": upd,
            "prefix_query": pre,
            "range_query": rng_q,
            "correctness": corr,
            "comparison": {
                "fenwick_memory_bytes": fw_mem,
                "segment_tree_memory_bytes": seg_mem,
                "memory_ratio": round(seg_mem / fw_mem, 2),
                "fenwick_point_update_ns": round(upd["actual_ns_per_op"], 1),
                "segment_tree_point_update_ns": round(seg_point, 1),
                "fenwick_range_query_ns": round(rng_q["actual_ns_per_op"], 1),
                "segment_tree_range_query_ns": round(seg_range_ns, 1),
            },
            "memory_theory_bytes": m.memory_bytes,
        })
        print()

    result = {
        "meta": {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "python": sys.version.split()[0],
            "note": "自制树状数组（1 开始下标，tree[i] 存 (i-lowbit(i), i] 的和），实测更新/查询延迟与内存",
            "seed": SEED,
        },
        "per_n": rows,
    }

    out = os.path.join(os.path.dirname(__file__), "..", "results",
                       f"fenwick_tree_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"结果写入 {out}")


if __name__ == "__main__":
    main()
