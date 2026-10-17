# -*- coding: utf-8 -*-
"""
树状数组（Fenwick Tree / Binary Indexed Tree）理论性能模型

树状数组用一维数组实现前缀和的 O(log n) 更新与查询，是线段树在
"可逆聚合"（和、异或等）上的精简替代。它由 Fenwick 在 1994 年的论文中给出。

核心思想：
- tree[i] 不存单个元素，而存一段区间的聚合值：区间 (i - lowbit(i), i]
- lowbit(i) = i & (-i)，即 i 的二进制最低位的 1 代表的数值
- 例如 i=12 (1100b)，lowbit=4，tree[12] 存 a[9..12] 的和

为什么这样能凑出前缀和：
- lowbit 恰好把下标按"末尾连续零的个数"分层
- 累加前缀和 i 时，反复 i -= lowbit(i) 走到的下标段首尾相接、不重不漏
- 单点更新 i 时，反复 i += lowbit(i) 覆盖所有包含 i 的区间

核心指标：
1. 前缀和查询：O(log n)，迭代次数 = i 的二进制中 1 的位数（≤ log2(n)+1）
2. 单点更新：O(log n)，迭代次数同理，也是 popcount 量级
3. 区间和查询：prefix(r) - prefix(l-1)，两次前缀和，仍 O(log n)
4. 空间：n × value_size 字节，比线段树的 4n 少 4 倍
5. 不支持懒标记：区间更新要走差分数组技巧（两个树状数组，区间更新 + 单点/区间查询）
6. 只能做可逆聚合：加法、异或可以；取 max/min 不可以（信息在差分里丢掉了）

本模块实现这些公式，参数可配。
"""
from dataclasses import dataclass
import math


@dataclass
class FenwickTreeSpec:
    """树状数组参数"""
    n: int = 1_000_000           # 元素数量
    value_bytes: int = 8         # 每个元素占字节数（int64）
    variant: str = "basic"       # 变体："basic" 单点更新+前缀查询 /
                                 #        "range" 区间更新+单点查询（差分）/
                                 #        "range-range" 区间更新+区间查询（双树）


@dataclass
class FenwickTreeMetrics:
    n: int                       # 元素数量
    tree_count: int              # 需要几棵树（双树变体为 2）
    memory_bytes: int            # 存储占用（字节）
    update_iters: float          # 单点更新平均迭代次数
    prefix_iters: float          # 前缀和查询平均迭代次数
    range_query_iters: float     # 区间和查询平均迭代次数（= 2×前缀和）
    worst_iters: int             # 最坏情况迭代次数（= 二进制位数上界）
    supports_lazy: bool          # 是否原生支持区间更新


def lowbit(i: int) -> int:
    """lowbit(i) = i & (-i)。i 的二进制最低位的 1 代表的数值。"""
    return i & (-i)


def compute_update_iters(spec: FenwickTreeSpec) -> float:
    """单点更新平均迭代次数。

    更新位置 i 时反复 i += lowbit(i) 直到越界，迭代次数 = 路径上 1 的分布。
    平均而言约为 (1/2) × log2(n)，因为每次跳过的位数期望是 2。
    """
    return 0.5 * math.log2(spec.n)


def compute_prefix_iters(spec: FenwickTreeSpec) -> float:
    """前缀和查询平均迭代次数。与更新同阶，也是 (1/2) × log2(n)。"""
    return 0.5 * math.log2(spec.n)


def compute_range_query_iters(spec: FenwickTreeSpec) -> float:
    """区间和查询平均迭代次数。prefix(r) - prefix(l-1) 两次前缀和。"""
    return 2 * compute_prefix_iters(spec)


def compute_worst_iters(spec: FenwickTreeSpec) -> int:
    """最坏情况迭代次数。= 二进制的位数上界 = floor(log2(n)) + 1。

    当 i 的二进制全是 1 时（如 1023 = 0b1111111111），每次 lowbit 只消掉一位。
    """
    return int(math.floor(math.log2(spec.n))) + 1


def compute_tree_count(spec: FenwickTreeSpec) -> int:
    """需要几棵树。

    basic 变体 1 棵；range-range 变体（区间更新 + 区间查询）需要 2 棵。
    """
    return 2 if spec.variant == "range-range" else 1


def compute_supports_lazy(spec: FenwickTreeSpec) -> bool:
    """是否原生支持区间更新。

    basic 变体不支持；range 与 range-range 变体用差分技巧实现区间更新，
    但不是线段树那种"懒标记下传"，而是数学恒等变换。
    """
    return spec.variant in ("range", "range-range")


def compute_memory(spec: FenwickTreeSpec) -> int:
    """存储占用。树的数量 × n × 每元素字节数。"""
    return compute_tree_count(spec) * spec.n * spec.value_bytes


def simulate_prefix_iters(i: int) -> int:
    """精确模拟前缀和查询的迭代次数。i 反复减 lowbit 直到 0。"""
    steps = 0
    while i > 0:
        i -= lowbit(i)
        steps += 1
    return steps


def simulate_update_iters(i: int, n: int) -> int:
    """精确模拟单点更新的迭代次数。i 反复加 lowbit 直到超过 n。"""
    steps = 0
    while i <= n:
        i += lowbit(i)
        steps += 1
    return steps


def sample_iters(spec: FenwickTreeSpec, samples: int = 1000) -> dict:
    """在随机位置上采样，返回前缀和与更新的实测平均迭代次数。"""
    import random
    rng = random.Random(0)
    total_prefix = 0
    total_update = 0
    for _ in range(samples):
        i = rng.randrange(1, spec.n + 1)
        total_prefix += simulate_prefix_iters(i)
        total_update += simulate_update_iters(i, spec.n)
    return {
        "prefix_avg": total_prefix / samples,
        "update_avg": total_update / samples,
    }


def compute(spec: FenwickTreeSpec) -> FenwickTreeMetrics:
    """给定参数，递推全部指标。"""
    sampled = sample_iters(spec)
    return FenwickTreeMetrics(
        n=spec.n,
        tree_count=compute_tree_count(spec),
        memory_bytes=compute_memory(spec),
        update_iters=sampled["update_avg"],
        prefix_iters=sampled["prefix_avg"],
        range_query_iters=2 * sampled["prefix_avg"],
        worst_iters=compute_worst_iters(spec),
        supports_lazy=compute_supports_lazy(spec),
    )


def _selftest():
    """公式自洽性检查"""
    spec = FenwickTreeSpec(n=1_000_000)
    m = compute(spec)

    # 内存 = n × value_bytes，是线段树 4n 的 1/4
    assert m.memory_bytes == spec.n * spec.value_bytes
    assert m.memory_bytes == (4 * spec.n * spec.value_bytes) // 4

    # 最坏迭代次数 = floor(log2(n)) + 1
    assert m.worst_iters == int(math.floor(math.log2(spec.n))) + 1

    # 最坏情况确实可达：i = 2^k - 1 时前缀和迭代 k 次
    for k in range(1, 20):
        i = 2 ** k - 1
        assert simulate_prefix_iters(i) == k, f"i={i} 应迭代 {k} 次"

    # 平均迭代次数约为最坏的一半（每次 lowbit 期望消去 1 位，实际平均走 2 位）
    assert m.prefix_iters < m.worst_iters, "平均应小于最坏"
    assert m.prefix_iters >= m.worst_iters * 0.3, "平均不应低于最坏的三成"

    # 区间和查询 = 2 × 前缀和
    assert abs(m.range_query_iters - 2 * m.prefix_iters) < 1e-9

    # basic 变体不支持原生区间更新
    assert m.supports_lazy is False

    # range-range 变体需要 2 棵树，内存翻倍，支持区间更新
    spec_rr = FenwickTreeSpec(n=1_000_000, variant="range-range")
    m_rr = compute(spec_rr)
    assert m_rr.tree_count == 2
    assert m_rr.memory_bytes == 2 * m.memory_bytes
    assert m_rr.supports_lazy is True

    # 空间仍是线性的，且范围查询仍是 O(log n)
    spec_big = FenwickTreeSpec(n=1_000_000_000)
    assert compute(spec_big).memory_bytes == 8 * spec_big.n
    assert compute(spec_big).worst_iters == 30

    print("selftest 全部通过")
    # 演示
    print(f"\nn={m.n:,}  树数量={m.tree_count}  内存={m.memory_bytes/1024/1024:.1f} MiB")
    print(f"单点更新={m.update_iters:.1f} 次迭代  前缀和={m.prefix_iters:.1f} 次  "
          f"区间和={m.range_query_iters:.1f} 次  最坏={m.worst_iters} 次")
    print(f"原生支持区间更新: {'是' if m.supports_lazy else '否（需差分技巧）'}")


if __name__ == "__main__":
    _selftest()
