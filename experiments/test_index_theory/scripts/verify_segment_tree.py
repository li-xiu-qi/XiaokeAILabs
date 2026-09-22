# -*- coding: utf-8 -*-
"""
线段树实测验证

用纯 Python 自制迷你线段树（array 迭代实现，即 Al.Cash 风格的自底向上版本），实测：
1. 建树时间与吞吐
2. 单点更新延迟
3. 区间查询延迟与实际访问节点数
4. 区间查询访问节点数与精确理论（simulate_query_nodes）的吻合度
5. 懒标记区间更新 vs 朴素逐点更新的性能差
6. 存储占用

延迟的理论换算是"节点常数"法：先由建树标定"每访问一个节点的纳秒数"，
再用 节点数 × 常数 预测各操作的延迟，与实测对比。这样能区分
"O(log n) 的节点数对不对"和"本机每节点多快"两件事。

方法：array('q') 存树，叶子落在下标 n..2n-1，自底向上递推。
结果写入 results/segment_tree_<timestamp>.json
"""
import os
import sys
import json
import random
import time
from array import array
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from segment_tree_model import (
    SegmentTreeSpec, compute, simulate_query_nodes, _sample_query_nodes,
)

N_LIST = [1_000, 10_000, 100_000, 1_000_000]
SEED = 42


class SegmentTree:
    """自制迷你线段树。迭代数组实现，支持单点更新、区间查询。

    叶子落在下标 n..2n-1，内部节点 i 的儿子是 2i 和 2i+1，
    根在 1，0 号位闲置。建树自底向上递推。
    """

    def __init__(self, n: int):
        self.n = n
        # 4n 足够容纳任意 n 的结构（n 不是 2 的幂时也不会越界）
        self.tree = array('q', bytes(8 * (4 * n)))
        self.nodes_visited = 0  # 累计访问节点数（用于核对理论）

    def build(self, data):
        """建树。先放叶子，再自底向上合并，O(n)。"""
        t = self.tree
        n = self.n
        for i in range(n):
            t[n + i] = data[i]
        for i in range(n - 1, 0, -1):
            t[i] = t[2 * i] + t[2 * i + 1]

    def point_update(self, i: int, value: int):
        """单点更新。改叶子后沿父节点链回退到根，O(log n)。"""
        t = self.tree
        i += self.n
        t[i] = value
        i >>= 1
        while i >= 1:
            t[i] = t[2 * i] + t[2 * i + 1]
            i >>= 1

    def range_query(self, l: int, r: int) -> int:
        """区间查询 [l, r)。两端逐层上跳，O(log n)。"""
        t = self.tree
        l += self.n
        r += self.n
        s = 0
        while l < r:
            if l & 1:
                s += t[l]
                l += 1
            if r & 1:
                r -= 1
                s += t[r]
            l >>= 1
            r >>= 1
        return s

    def query_nodes(self, l: int, r: int) -> int:
        """只数访问节点数，不做加法。用于核对理论。"""
        l += self.n
        r += self.n
        cnt = 0
        while l < r:
            if l & 1:
                cnt += 1
                l += 1
            if r & 1:
                r -= 1
                cnt += 1
            l >>= 1
            r >>= 1
        return cnt

    def naive_range_update(self, l: int, r: int, delta: int):
        """朴素区间更新：逐个点改，每个点 O(log n)，总 O(k log n)。"""
        for i in range(l, r):
            self.point_update(i, self.tree[self.n + i] + delta)

    def storage_bytes(self) -> int:
        """实际存储占用。array 的缓冲区字节数。"""
        return self.tree.buffer_info()[1] * self.tree.itemsize


class LazySegmentTree:
    """懒标记线段树。递归实现，区间更新 O(log n)。

    每个节点存一个待下传标记，update 时若当前节点对应的区间完全被覆盖，
    直接打标记不再下探。
    """

    def __init__(self, n: int):
        self.n = n
        self.tree = array('q', bytes(8 * (4 * n)))
        self.lazy = array('q', bytes(8 * (4 * n)))

    def build(self, data):
        """建树。递归折半填值，O(n)。

        注意必须用与 _update/_query 同一套区间假设（递归折半）来填，
        不能套用迭代数组布局（叶子落在 n..2n-1）。后者在 n 不是 2 的幂时，
        节点 1 的索引子树覆盖 2^ceil(log2 n) 个叶子槽位而不是 n 个，
        与折半假设不一致，会导致聚合值算错。这是实现层面的真实陷阱。
        """
        self._build(1, 0, self.n, data)

    def _build(self, node: int, nl: int, nr: int, data):
        if nr - nl == 1:
            self.tree[node] = data[nl]
            return
        mid = (nl + nr) // 2
        self._build(2 * node, nl, mid, data)
        self._build(2 * node + 1, mid, nr, data)
        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def _apply(self, node: int, length: int, delta: int):
        self.tree[node] += delta * length
        self.lazy[node] += delta

    def _push(self, node: int, length: int):
        if self.lazy[node] != 0:
            half = length // 2
            self._apply(2 * node, half, self.lazy[node])
            self._apply(2 * node + 1, length - half, self.lazy[node])
            self.lazy[node] = 0

    def range_update(self, l: int, r: int, delta: int):
        """区间更新 [l, r)。O(log n)。"""
        self._update(1, 0, self.n, l, r, delta)

    def _update(self, node: int, nl: int, nr: int, l: int, r: int, delta: int):
        if l <= nl and nr <= r:
            self._apply(node, nr - nl, delta)
            return
        self._push(node, nr - nl)
        mid = (nl + nr) // 2
        if l < mid:
            self._update(2 * node, nl, mid, l, r, delta)
        if r > mid:
            self._update(2 * node + 1, mid, nr, l, r, delta)
        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def range_query(self, l: int, r: int) -> int:
        return self._query(1, 0, self.n, l, r)

    def _query(self, node: int, nl: int, nr: int, l: int, r: int) -> int:
        if l <= nl and nr <= r:
            return self.tree[node]
        self._push(node, nr - nl)
        mid = (nl + nr) // 2
        s = 0
        if l < mid:
            s += self._query(2 * node, nl, mid, l, r)
        if r > mid:
            s += self._query(2 * node + 1, mid, nr, l, r)
        return s


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


def measure_build(spec: SegmentTreeSpec) -> dict:
    """实测建树。返回耗时、吞吐、以及标定出的"每节点纳秒数"。"""
    data = make_data(spec.n)
    st = SegmentTree(spec.n)
    repeats = 3 if spec.n <= 100_000 else 1

    start = time.perf_counter_ns()
    for _ in range(repeats):
        st.build(data)
    elapsed_ns = (time.perf_counter_ns() - start) / repeats

    # 建树做 n-1 次合并 + n 次叶子写入，总节点访问约 2n-1
    build_merges = compute(spec).build_ops
    per_node_ns = elapsed_ns / (2 * spec.n - 1)
    return {
        "build_ns": round(elapsed_ns),
        "build_ms": round(elapsed_ns / 1e6, 3),
        "throughput_m_per_s": round(spec.n / (elapsed_ns / 1e9) / 1e6, 3),
        "build_merges": build_merges,
        "per_node_ns": round(per_node_ns, 2),
    }


def measure_point_update(spec: SegmentTreeSpec, per_node_ns: float) -> dict:
    """实测单点更新。随机位置，多轮取平均。"""
    data = make_data(spec.n)
    st = SegmentTree(spec.n)
    st.build(data)
    rng = random.Random(SEED + 1)
    indices = [rng.randrange(0, spec.n) for _ in range(2000)]

    actual_ns = time_ns(lambda: [st.point_update(i, 0) for i in indices[:200]], 5)

    nodes = compute(spec).point_update_nodes
    return {
        "actual_ns": round(actual_ns, 1),
        "actual_ns_per_op": round(actual_ns / 200, 1),
        "nodes_theory": nodes,
        "predicted_ns_per_op": round(nodes * per_node_ns, 1),
        "iterations": 1000,
    }


def measure_range_query(spec: SegmentTreeSpec, per_node_ns: float) -> dict:
    """实测区间查询。随机区间，同时统计实际访问节点数。"""
    data = make_data(spec.n)
    st = SegmentTree(spec.n)
    st.build(data)
    rng = random.Random(SEED + 2)

    ranges = []
    for _ in range(2000):
        l = rng.randrange(0, spec.n)
        r = rng.randrange(l + 1, spec.n + 1)
        ranges.append((l, r))

    # 实际访问节点数
    actual_nodes = []
    for l, r in ranges[:500]:
        actual_nodes.append(st.query_nodes(l, r))
    avg_actual_nodes = sum(actual_nodes) / len(actual_nodes)

    # 精确理论值（同一批区间）
    theo_nodes = [simulate_query_nodes(spec.n, l, r) for l, r in ranges[:500]]
    avg_theo_nodes = sum(theo_nodes) / len(theo_nodes)

    actual_ns = time_ns(lambda: [st.range_query(l, r) for l, r in ranges[:200]], 5)

    bound = compute(spec).range_query_nodes_bound
    return {
        "actual_ns_per_op": round(actual_ns / 200, 1),
        "avg_actual_nodes": round(avg_actual_nodes, 2),
        "avg_theory_nodes": round(avg_theo_nodes, 2),
        "nodes_match_ratio": round(avg_actual_nodes / avg_theo_nodes, 4),
        "classic_bound": bound,
        "bound_ratio": round(avg_theo_nodes / bound, 4),
        "predicted_ns_per_op": round(avg_theo_nodes * per_node_ns, 1),
        "iterations": 1000,
    }


def measure_lazy_vs_naive(spec: SegmentTreeSpec) -> dict:
    """实测懒标记区间更新 vs 朴素逐点更新。

    在 n=10000 上做 1000 次区间长度为 1000 的更新。
    """
    n = 10_000
    length = 1_000
    count = 200
    data = make_data(n)

    # 朴素：逐个点改
    st = SegmentTree(n)
    st.build(data)
    rng = random.Random(SEED + 3)
    # 必须保证 l < r：先取起点，再取落在 [l, n) 内的终点
    updates = []
    for _ in range(count):
        l = rng.randrange(0, n - length)
        updates.append((l, l + length))
    start = time.perf_counter_ns()
    for l, r in updates:
        st.naive_range_update(l, r, 1)
    naive_ns = (time.perf_counter_ns() - start) / count

    # 懒标记：一次区间更新
    lst = LazySegmentTree(n)
    lst.build(data)
    start = time.perf_counter_ns()
    for l, r in updates:
        lst.range_update(l, r, 1)
    lazy_ns = (time.perf_counter_ns() - start) / count

    # 正确性：两种方式的结果都与暴力真值一致
    # 暴力真值：原始数组逐个累加 delta
    rng2 = random.Random(SEED + 4)
    truth = list(data)
    for l, r in updates:
        for i in range(l, r):
            truth[i] += 1
    checked = 0
    mismatches = 0
    for _ in range(200):
        l = rng2.randrange(0, n)
        r = rng2.randrange(l + 1, n + 1)
        checked += 1
        expected = sum(truth[l:r])
        if st.range_query(l, r) != expected:
            mismatches += 1
        if lst.range_query(l, r) != expected:
            mismatches += 1

    m = compute(spec)
    height = m.height
    return {
        "n": n,
        "range_length": length,
        "count": count,
        "naive_ns": round(naive_ns, 1),
        "lazy_ns": round(lazy_ns, 1),
        "speedup": round(naive_ns / lazy_ns, 1),
        "naive_nodes_theory": length * height,
        "lazy_nodes_theory": m.lazy_range_update_nodes,
        "speedup_theory": round(length * height / m.lazy_range_update_nodes, 1),
        "correctness_checked": checked,
        "correctness_mismatches": mismatches,
    }


def main():
    print("=== 线段树实测验证 ===\n")

    rows = []
    for n in N_LIST:
        spec = SegmentTreeSpec(n=n)
        print(f"--- n={n:,} ---")
        build = measure_build(spec)
        point = measure_point_update(spec, build["per_node_ns"])
        rng_q = measure_range_query(spec, build["per_node_ns"])
        storage_actual = SegmentTree(n).storage_bytes()

        m = compute(spec)
        print(f"  建树: {build['build_ms']:.2f} ms  吞吐 {build['throughput_m_per_s']:.1f} M 元素/s  "
              f"每节点 {build['per_node_ns']:.2f} ns")
        print(f"  单点更新: 实测 {point['actual_ns_per_op']:.0f} ns  "
              f"理论 {point['nodes_theory']} 节点 × {build['per_node_ns']:.2f} ns = "
              f"{point['predicted_ns_per_op']:.0f} ns")
        print(f"  区间查询: 实测 {rng_q['actual_ns_per_op']:.0f} ns  访问节点 "
              f"实测 {rng_q['avg_actual_nodes']:.2f} / 理论 {rng_q['avg_theory_nodes']:.2f}  "
              f"（经典上界 {rng_q['classic_bound']}，占 {rng_q['bound_ratio']*100:.0f}%）")
        print(f"  存储: 实测 {storage_actual/1024/1024:.2f} MiB  理论 {m.memory_bytes/1024/1024:.2f} MiB")

        rows.append({
            "n": n,
            "tree_nodes": m.tree_nodes,
            "height": m.height,
            "build": build,
            "point_update": point,
            "range_query": rng_q,
            "storage_actual_bytes": storage_actual,
            "storage_theory_bytes": m.memory_bytes,
        })
        print()

    # 懒标记 vs 朴素
    print("--- 懒标记区间更新 vs 朴素逐点更新（n=10000，区间长 1000）---")
    lazy = measure_lazy_vs_naive(SegmentTreeSpec(n=1_000_000))
    print(f"  朴素逐点: {lazy['naive_ns']:.0f} ns/次  懒标记: {lazy['lazy_ns']:.0f} ns/次  "
          f"加速 {lazy['speedup']:.0f}x（理论 {lazy['speedup_theory']:.0f}x）")
    print(f"  结果一致: 抽查 {lazy['correctness_checked']} 个区间，不一致 {lazy['correctness_mismatches']} 个")

    result = {
        "meta": {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "python": sys.version.split()[0],
            "note": "自制迭代线段树（array 实现），实测建树/单点更新/区间查询/懒标记",
            "seed": SEED,
            "impl": "array('q') 4n，叶子在 n..2n-1，根在 1",
        },
        "per_n": rows,
        "lazy_vs_naive": lazy,
    }

    out = os.path.join(os.path.dirname(__file__), "..", "results",
                       f"segment_tree_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n结果写入 {out}")


if __name__ == "__main__":
    main()
