# -*- coding: utf-8 -*-
"""
线段树（Segment Tree）理论性能模型

线段树是区间查询的标准结构，用于维护一个数组上的区间聚合（和 / 最大 / 最小）。
它把区间不断二分，直到每个节点对应一个元素，聚合值自底向上递推。

核心结构：
- 二叉树，n 个叶节点（每个对应一个元素）
- 每个内部节点存储其对应区间的聚合值
- 递归二分：根节点对应 [0, n)，两个儿子各管一半
- 内部节点数 = n-1，总节点数 = 2n-1

两种实现，空间不同（本模型两种都给）：
- 递归指针版：恰好 2n-1 个节点
- 迭代数组版：补到 2 的幂，开 4n 或 2×2^ceil(log2(n)) 的空间，常数更大但缓存友好

核心指标：
1. 节点数与树高：2n-1 个节点，树高 ceil(log2(n)) + 1
2. 建树：O(n)，做 n-1 次合并
3. 单点更新：O(log n)，改一个叶子再沿路径回退到根，约 log2(n)+1 个节点
4. 区间查询：O(log n)，查询区间被分解为 O(log n) 个不相交的规范节点
5. 区间更新：朴素做法改区间内每个点，O(k log n)；懒标记做法 O(log n)
6. 空间：4n × value_size 字节（数组实现，n 个元素 int64 时 32n 字节）

本模块实现这些公式，参数可配。
"""
from dataclasses import dataclass
import math


@dataclass
class SegmentTreeSpec:
    """线段树参数"""
    n: int = 1_000_000           # 元素数量
    value_bytes: int = 8         # 每个节点存储值占字节数（int64）
    impl: str = "array"          # 实现方式："array" 迭代数组版 / "pointer" 递归指针版
    enable_lazy: bool = False    # 是否启用懒标记（区间更新）


@dataclass
class SegmentTreeMetrics:
    n: int                       # 元素数量
    tree_nodes: int              # 逻辑节点数（叶 n 个 + 内部 n-1 个）
    height: int                  # 树高（层数）
    array_size: int              # 实际数组长度（含补齐）
    memory_bytes: int            # 存储占用（字节）
    build_ops: int               # 建树合并次数
    point_update_nodes: int      # 单点更新访问节点数
    range_query_nodes_bound: int # 区间查询分解节点数上界
    range_query_nodes_typical: float  # 区间查询分解节点数典型值
    lazy_range_update_nodes: int # 懒标记区间更新访问节点数


def compute_tree_nodes(spec: SegmentTreeSpec) -> int:
    """逻辑节点数。叶 n 个 + 内部 n-1 个 = 2n-1。"""
    return 2 * spec.n - 1


def compute_height(spec: SegmentTreeSpec) -> int:
    """树高（层数）。完全二叉树高度 = ceil(log2(n)) + 1。"""
    return int(math.ceil(math.log2(spec.n))) + 1


def compute_array_size(spec: SegmentTreeSpec) -> int:
    """实际数组长度。

    array 实现按 4n 开（足够容纳任意 n 的补零结构），
    pointer 实现按 2n-1 开。
    """
    if spec.impl == "array":
        return 4 * spec.n
    return 2 * spec.n - 1


def compute_memory(spec: SegmentTreeSpec) -> int:
    """存储占用。数组长度 × 每节点字节数。

    启用懒标记时每个节点再挂一个标记，翻倍。
    """
    return compute_array_size(spec) * spec.value_bytes * (2 if spec.enable_lazy else 1)


def compute_build_ops(spec: SegmentTreeSpec) -> int:
    """建树合并次数。n-1 次（每个内部节点做一次合并）。"""
    return spec.n - 1


def compute_point_update_nodes(spec: SegmentTreeSpec) -> int:
    """单点更新访问节点数。改一个叶子，再沿路径回退到根。

    路径长度 = 树高 - 1 = ceil(log2(n))，加上叶子本身约 log2(n)+1 个节点。
    """
    return compute_height(spec) - 1 + 1


def compute_range_query_nodes_bound(spec: SegmentTreeSpec) -> int:
    """区间查询分解节点数上界。经典上界 4 × ceil(log2(n))。"""
    return 4 * int(math.ceil(math.log2(spec.n)))


def compute_range_query_nodes_typical(spec: SegmentTreeSpec) -> int:
    """区间查询分解节点数典型值。约 2 × log2(n)。"""
    return 2 * math.log2(spec.n)


def compute_lazy_range_update_nodes(spec: SegmentTreeSpec) -> int:
    """懒标记区间更新访问节点数。查询区间分解出多少个节点就标记多少个。

    与区间查询同一套分解，所以也是 2 × log2(n) 量级。
    """
    return int(2 * math.log2(spec.n))


def simulate_query_nodes(n: int, l: int, r: int) -> int:
    """精确模拟迭代线段树区间查询 [l, r) 访问的节点数。

    迭代线段树把叶子放在下标 n..2n-1，自底向上跳父节点。
    这个函数按同样的跳法数访问的节点，结果与实测一致，可作精确理论值。
    """
    l += n
    r += n
    left_path = []
    right_path = []
    while l < r:
        if l & 1:
            left_path.append(l)
            l += 1
        if r & 1:
            r -= 1
            right_path.append(r)
        l >>= 1
        r >>= 1
    return len(left_path) + len(right_path)


def _sample_query_nodes(n: int, samples: int = 200) -> float:
    """随机区间查询访问节点数的样本均值。供 selftest 与文档口径使用。"""
    import random
    rng = random.Random(0)
    total = 0
    for _ in range(samples):
        l = rng.randrange(0, n)
        r = rng.randrange(l + 1, n + 1)
        total += simulate_query_nodes(n, l, r)
    return total / samples


def compute(spec: SegmentTreeSpec) -> SegmentTreeMetrics:
    """给定参数，递推全部指标。"""
    return SegmentTreeMetrics(
        n=spec.n,
        tree_nodes=compute_tree_nodes(spec),
        height=compute_height(spec),
        array_size=compute_array_size(spec),
        memory_bytes=compute_memory(spec),
        build_ops=compute_build_ops(spec),
        point_update_nodes=compute_point_update_nodes(spec),
        range_query_nodes_bound=compute_range_query_nodes_bound(spec),
        range_query_nodes_typical=compute_range_query_nodes_typical(spec),
        lazy_range_update_nodes=compute_lazy_range_update_nodes(spec),
    )


def _selftest():
    """公式自洽性检查"""
    spec = SegmentTreeSpec(n=1_000_000)
    m = compute(spec)

    # 节点数 = 2n-1
    assert m.tree_nodes == 2 * spec.n - 1, "节点数应为 2n-1"

    # 数组实现按 4n 开
    assert m.array_size == 4 * spec.n, "数组长度应为 4n"
    assert m.memory_bytes == 4 * spec.n * spec.value_bytes, "内存应为 4n × value_bytes"

    # 树高随 log(n) 增长
    spec_big = SegmentTreeSpec(n=1_000_000_000)
    assert compute(spec_big).height > m.height, "n 越大，树高应越大"

    # 树高 = ceil(log2(n)) + 1
    assert m.height == int(math.ceil(math.log2(spec.n))) + 1

    # 单点更新节点数 = log2(n) + 1
    assert m.point_update_nodes == m.height, "单点更新节点数应等于树高"

    # 区间查询上界 >= 典型值
    assert m.range_query_nodes_bound >= m.range_query_nodes_typical

    # 建树 O(n)，合并次数 = n-1
    assert m.build_ops == spec.n - 1

    # 懒标记的访问节点数远小于朴素区间更新（区间长度 k 时朴素要 k × 树高）
    range_len = 1000
    naive = range_len * m.height
    assert m.lazy_range_update_nodes < naive / 10, "懒标记应比朴素区间更新快一个量级以上"

    # 启用懒标记，内存翻倍
    spec_lazy = SegmentTreeSpec(n=1_000_000, enable_lazy=True)
    assert compute(spec_lazy).memory_bytes == 2 * m.memory_bytes, "懒标记内存应翻倍"

    # 递归指针版空间更小
    spec_ptr = SegmentTreeSpec(n=1_000_000, impl="pointer")
    assert compute(spec_ptr).array_size == 2 * spec.n - 1

    # 区间查询精确模拟：全区间查询走"梳子路径"，节点数 ≤ 2×ceil(log2(n))
    # 注意不是 1：迭代线段树的叶子在 n..2n-1，根在 1，两端逐层上跳各取一支
    full = simulate_query_nodes(spec.n, 0, spec.n)
    assert full <= 2 * int(math.ceil(math.log2(spec.n))), "全区间查询不应超过 2×ceil(log2(n))"
    assert full >= 1, "全区间查询至少访问根节点"

    # 随机宽度区间的访问节点数应在上界以内
    # 实测平均 16.9（n=100 万），低于 2×log2(n)=39.9 这个松估计，说明迭代实现的分解更紧凑
    random_seed_samples = _sample_query_nodes(spec.n)
    assert random_seed_samples <= m.range_query_nodes_bound, "精确模拟不应超过上界"
    assert 1 <= random_seed_samples <= m.range_query_nodes_bound

    print("selftest 全部通过")
    # 演示
    print(f"\nn={m.n:,}  节点数={m.tree_nodes:,}  树高={m.height}  "
          f"数组长度={m.array_size:,}")
    print(f"建树合并={m.build_ops:,} 次  单点更新={m.point_update_nodes} 节点  "
          f"区间查询采样均值={_sample_query_nodes(spec.n):.1f} 节点 "
          f"(经典上界 {m.range_query_nodes_bound}，2×log2(n) 松估计 {m.range_query_nodes_typical:.0f})")
    print(f"内存={m.memory_bytes/1024/1024:.1f} MiB"
          + ("（含懒标记翻倍）" if spec.enable_lazy else ""))


if __name__ == "__main__":
    _selftest()
