# -*- coding: utf-8 -*-
"""
ART（自适应基数树，Adaptive Radix Tree）理论性能模型

ART 是 Trie 的变体，核心改动是"节点大小跟着子节点数量走"。标准 Trie 用固定 σ 槽的
数组存子节点，σ 大时绝大多数槽位是空的；ART 按实际子节点数在四种节点形态间切换，
把每节点的固定开销压下来。

四种内部节点形态（子节点数上限 → 典型字节数）：
    Node4    ≤ 4   子节点   → 64 字节
    Node16   ≤ 16  子节点   → 128 字节
    Node48   ≤ 48  子节点   → 256 字节
    Node256  ≤ 256 子节点   → 2048 字节

核心指标：
1. 节点数：与同参数标准 Trie 相同（树形一致），懒展开能再压掉单链节点
2. 查找时间：O(k)，k 是键长。每层定位子节点的代价随节点形态变化
   - Node4  线性扫描，≤ 4 次比较
   - Node16 SIMD 单指令并行比较 16 个键字节
   - Node48 SIMD 查 256 字节索引表，再取指针
   - Node256 直接以字节值为下标，1 次访存
3. 空间：比固定 σ 槽的标准 Trie 少一个量级（σ 越大差距越大）

本模块全部用闭式公式给出这些指标。节点形态分布由"每个节点平均覆盖多少键"递推得到，
不依赖任何真实 ART 实现。
"""
from dataclasses import dataclass, field
import math



# 节点形态定义：(子节点数上限, 典型字节数)
NODE_TYPES = (
    ("Node4", 4, 64),
    ("Node16", 16, 128),
    ("Node48", 48, 256),
    ("Node256", 256, 2048),
)


@dataclass
class ARTSpec:
    """ART 参数"""
    n: int = 100_000               # 键数量
    avg_len: int = 12              # 平均键长 L
    sigma: int = 26                # 字符集大小
    len_at_least: tuple = ()       # 变长键模式：len_at_least[d] = 长度 ≥ d 的键数。
                                   # 留空则按定长键处理。
    # 每种节点形态的字节数，可按具体实现调整
    bytes_node4: int = 64
    bytes_node16: int = 128
    bytes_node48: int = 256
    bytes_node256: int = 2048
    leaf_bytes: int = 64           # 终止节点（持值）按 Node4 量级计


@dataclass
class ARTMetrics:
    n: int                          # 键数量
    avg_len: int                    # 键长
    sigma: int                      # 字符集大小
    node_count: int                 # 总节点数
    node_count_upper_bound: int     # 无共享上界 n × L
    node4_count: int                # 各形态节点数
    node16_count: int
    node48_count: int
    node256_count: int
    leaf_count: int                 # 终止节点数
    memory_bytes: int               # 总内存
    memory_per_key: float           # 每键字节数
    lookup_simd_ops: int            # 查找的 SIMD 级操作数 = 层数
    lookup_scalar_ops: float        # 等价的标量比较次数（把 SIMD 摊回标量）
    prefix_search_ops: int          # 前缀搜索最小代价 = L + k


def node_type_name(child_count: int) -> str:
    """按子节点数选节点形态名。"""
    for name, cap, _ in NODE_TYPES:
        if child_count <= cap:
            return name
    return "Node256"


def node_type_bytes(child_count: int, spec: ARTSpec) -> int:
    """按子节点数返回该节点的字节数。"""
    table = {
        "Node4": spec.bytes_node4,
        "Node16": spec.bytes_node16,
        "Node48": spec.bytes_node48,
        "Node256": spec.bytes_node256,
    }
    return table[node_type_name(child_count)]


def _distinct_prefix_count(m: int, d: int, sigma: int) -> float:
    """m 个独立均匀随机键在深度 d 上的期望互异前缀数。

    用 expm1/log1p 稳定形式：σ^d 一大，(1 - σ^{-d})^m 在 double 下会退化成 1.0，
    直接代入会让深层贡献整体归零。
    """
    if m <= 0 or sigma <= 1:
        return 0.0
    slots = float(sigma) ** d
    return slots * (-math.expm1(m * math.log1p(-1.0 / slots)))


def compute(spec: ARTSpec) -> ARTMetrics:
    """给定参数，逐深度递推节点形态分布与内存。"""
    if spec.len_at_least:
        # 变长键：深度 d 上只由长度够得着的键产生前缀
        cover = list(spec.len_at_least)
        top_depth = len(cover) - 1
        root_keys = cover[1] if len(cover) > 1 else spec.n
    else:
        # 定长键：每个深度上都是全部 n 个键
        cover = [spec.n] * (spec.avg_len + 1)
        top_depth = spec.avg_len
        root_keys = spec.n

    counts = {"Node4": 0, "Node16": 0, "Node48": 0, "Node256": 0}
    scalar_ops = 0.0
    simd_ops = 0
    node_count = 0  # 根节点在循环第一轮里计入，不重复加
    leaf_count = 0

    for d in range(0, top_depth):
        m = cover[d + 1]          # 深度 d 的节点由长度 ≥ d+1 的键支撑
        if d == 0:
            nodes_d = 1.0         # 根节点只有一个
            m = root_keys
        else:
            nodes_d = _distinct_prefix_count(cover[d], d, spec.sigma)

        if nodes_d <= 0:
            continue

        # 该层每个节点平均覆盖多少键，进而期望有多少个不同子字符
        keys_per_node = m / nodes_d
        expected_children = spec.sigma * (1.0 - (1.0 - 1.0 / spec.sigma) ** keys_per_node)

        counts[node_type_name(int(round(expected_children)))] += int(round(nodes_d))
        scalar_ops += nodes_d * expected_children
        simd_ops += 1
        node_count += int(round(nodes_d))

    # 最深层是终止节点，持值，按 Node4 量级计
    if top_depth >= 1:
        leaf_count = int(round(_distinct_prefix_count(cover[top_depth], top_depth, spec.sigma)))
        counts["Node4"] += leaf_count
        node_count += leaf_count
    leaf_count = max(leaf_count, 1)

    memory = (counts["Node4"] * spec.bytes_node4
              + counts["Node16"] * spec.bytes_node16
              + counts["Node48"] * spec.bytes_node48
              + counts["Node256"] * spec.bytes_node256)

    upper = spec.n * spec.avg_len

    return ARTMetrics(
        n=spec.n,
        avg_len=spec.avg_len,
        sigma=spec.sigma,
        node_count=node_count,
        node_count_upper_bound=upper,
        node4_count=counts["Node4"],
        node16_count=counts["Node16"],
        node48_count=counts["Node48"],
        node256_count=counts["Node256"],
        leaf_count=leaf_count,
        memory_bytes=memory,
        memory_per_key=memory / spec.n if spec.n else 0.0,
        lookup_simd_ops=simd_ops,
        lookup_scalar_ops=scalar_ops,
        prefix_search_ops=spec.avg_len + 1,
    )


def _selftest():
    """公式自洽性检查"""
    spec = ARTSpec(n=100_000, avg_len=12, sigma=26)
    m = compute(spec)

    # 1. 节点数落在 [n, n×L] 区间内
    assert spec.n <= m.node_count <= m.node_count_upper_bound, \
        f"节点数 {m.node_count} 越界"

    # 2. 各形态计数之和等于总节点数（含根）
    type_sum = (m.node4_count + m.node16_count + m.node48_count + m.node256_count)
    assert type_sum == m.node_count, \
        f"形态计数之和 {type_sum} != 总节点数 {m.node_count}"

    # 3. 查找的 SIMD 操作数 = 层数，与 n 无关
    spec_big = ARTSpec(n=100_000_000, avg_len=12, sigma=26)
    assert compute(spec_big).lookup_simd_ops == m.lookup_simd_ops, \
        "查找代价不应随 n 变化"

    # 4. σ 越大，互异前缀越多，节点数与每键内存都上升
    spec_sigma256 = ARTSpec(n=100_000, avg_len=12, sigma=256)
    m256 = compute(spec_sigma256)
    assert m256.node_count > m.node_count, "σ 越大，ART 节点应越多"
    assert m256.memory_per_key > m.memory_per_key, "σ 越大，每键内存应越高"
    assert m256.node256_count > m.node256_count, "σ=256 应有更多 Node256"

    # 5. 字符集越小，节点越少
    spec_small = ARTSpec(n=100_000, avg_len=12, sigma=4)
    assert compute(spec_small).node_count < m.node_count, "σ 越小节点应越少"

    # 6. 节点形态按子节点数单调升级
    assert node_type_bytes(3, spec) == spec.bytes_node4
    assert node_type_bytes(5, spec) == spec.bytes_node16
    assert node_type_bytes(17, spec) == spec.bytes_node48
    assert node_type_bytes(49, spec) == spec.bytes_node256

    # 7. 节点数随 n 单调增长
    spec_n2 = ARTSpec(n=200_000, avg_len=12, sigma=26)
    assert compute(spec_n2).node_count > m.node_count, "n 越大节点应越多"

    # 8. 变长键全量覆盖时退化回定长结果
    full = (spec.n,) * (spec.avg_len + 1)
    spec_v = ARTSpec(n=spec.n, avg_len=spec.avg_len, sigma=26, len_at_least=full)
    mv = compute(spec_v)
    assert mv.node_count == m.node_count, \
        f"变长键退化不一致：{mv.node_count} vs {m.node_count}"

    print("selftest 全部通过")
    print(f"\nn={m.n:,}  L={m.avg_len}  σ={m.sigma}")
    print(f"节点数={m.node_count:,}  （上界 n×L={m.node_count_upper_bound:,}）")
    print(f"形态分布: Node4={m.node4_count:,}  Node16={m.node16_count:,}  "
          f"Node48={m.node48_count:,}  Node256={m.node256_count:,}")
    print(f"内存={m.memory_bytes/1024/1024:.1f} MiB  （{m.memory_per_key:.0f} B/键）")
    print(f"查找={m.lookup_simd_ops} 层 SIMD 操作  "
          f"（等价标量比较 {m.lookup_scalar_ops:,.0f} 次）")


if __name__ == "__main__":
    _selftest()
