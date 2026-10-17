# -*- coding: utf-8 -*-
"""
跳表（Skip List）理论性能模型

跳表是 Redis Sorted Set、LevelDB/RocksDB MemTable 的核心结构。它用多层链表实现 O(log n) 的查找、插入、删除，比平衡树简单。

核心结构：
- 底层（Level 0）是完整的有序链表
- 每往上一层，节点数减半（概率 p，典型 0.5）
- 查找从最高层开始，向右走到下一个大于目标的位置，然后下降一层

核心指标：
1. 期望层数：log_{1/p}(n)，p=0.5 时约 log2(n)
2. 查找路径长度：O(log n)，期望比较次数
3. 存储占用：每个节点有 1~L 个指针（层数），期望指针数 = n/(1-p)
4. 插入时节点高度：随机决定，期望 1/(1-p) 个指针

本模块实现这些公式，参数可配。
"""
from dataclasses import dataclass
import math


@dataclass
class SkipListSpec:
    """跳表参数"""
    n: int = 1_000_000           # 元素数量
    p: float = 0.5               # 节点晋升概率（每层保留比例）
    key_bytes: int = 8           # 键占字节数
    value_bytes: int = 8         # 值占字节数
    pointer_bytes: int = 8       # 指针占字节数


@dataclass
class SkipListMetrics:
    n: int                       # 元素数量
    max_level: int               # 最大层数（期望）
    avg_levels_per_node: float   # 每个节点的平均层数
    total_pointers: int          # 总指针数
    storage_bytes: int           # 存储占用（字节）
    expected_comparisons: float  # 期望比较次数（查找路径）


def compute_max_level(spec: SkipListSpec) -> int:
    """期望最大层数。跳表最大层数约 log_{1/p}(n)。"""
    return int(math.ceil(math.log(spec.n) / math.log(1 / spec.p)))


def compute_avg_levels(spec: SkipListSpec) -> float:
    """每个节点的平均层数。几何分布，期望 1/(1-p)。"""
    return 1.0 / (1 - spec.p)


def compute_storage(spec: SkipListSpec) -> tuple:
    """存储占用。每个节点有随机层数，总指针数 = n/(1-p)。"""
    total_pointers = int(spec.n / (1 - spec.p))
    # 每个节点：key + value + 每层一个指针
    storage = spec.n * (spec.key_bytes + spec.value_bytes) + total_pointers * spec.pointer_bytes
    return total_pointers, storage


def compute_expected_comparisons(spec: SkipListSpec) -> float:
    """期望比较次数。跳表查找路径长度约 (1/p) × log_{1/p}(n)。"""
    max_level = compute_max_level(spec)
    # 每层期望走 1/(1-p) 步，下降 max_level 层
    return (1 / (1 - spec.p)) * max_level


def compute(spec: SkipListSpec) -> SkipListMetrics:
    """给定参数，递推全部指标。"""
    max_level = compute_max_level(spec)
    avg_levels = compute_avg_levels(spec)
    total_pointers, storage = compute_storage(spec)
    comparisons = compute_expected_comparisons(spec)

    return SkipListMetrics(
        n=spec.n,
        max_level=max_level,
        avg_levels_per_node=avg_levels,
        total_pointers=total_pointers,
        storage_bytes=storage,
        expected_comparisons=comparisons,
    )


def _selftest():
    """公式自洽性检查"""
    spec = SkipListSpec()

    # 最大层数应随 n 增长
    m = compute(spec)
    spec_big = SkipListSpec(n=1_000_000_000)
    assert compute(spec_big).max_level > m.max_level, "n 越大，层数应越多"

    # p 越大，层数越多（log_{1/p}(n)，1/p 越小，底数越小则层数越多）
    spec_high_p = SkipListSpec(p=0.75)
    assert compute(spec_high_p).max_level > m.max_level, "p 越大，层数应越多"

    # 平均层数 = 1/(1-p)
    assert abs(m.avg_levels_per_node - 1.0 / (1 - spec.p)) < 0.01

    # 存储应随 n 线性增长
    assert m.storage_bytes > 0

    # 期望比较次数应为 O(log n)
    assert m.expected_comparisons < spec.n, "比较次数应远小于 n"

    print("selftest 全部通过")
    # 演示
    print(f"\nn={m.n:,}  最大层数={m.max_level}  "
          f"平均层数/节点={m.avg_levels_per_node:.2f}  "
          f"期望比较次数={m.expected_comparisons:.1f}")
    print(f"总指针数={m.total_pointers:,}  "
          f"存储={m.storage_bytes/1024/1024:.1f} MiB")


if __name__ == "__main__":
    _selftest()
