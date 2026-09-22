# -*- coding: utf-8 -*-
"""
LRU 缓存算法理论性能模型

LRU（Least Recently Used）是数据库 Buffer Pool、操作系统页面置换、Web 缓存的标准算法。它淘汰最久未使用的页面。

核心结构：
- 双向链表：按访问时间排序，头部是最近使用，尾部是最久未使用
- 哈希表：O(1) 查找页面

核心指标：
1. 命中率：给定访问序列和缓存容量，命中次数 / 总访问次数
2. 期望命中率：与访问分布的 Zipf 系数有关，容量越大命中率越高
3. 操作延迟：查找 O(1)，插入/淘汰 O(1)

标准假设：
- 缓存容量 C（页面数）
- 访问序列长度 N
- 访问分布：Zipf 分布，参数 s（s 越大，访问越集中）

本模块实现 LRU 的命中率公式和操作延迟，参数可配。
"""
from dataclasses import dataclass
import math


@dataclass
class LRUSpec:
    """LRU 缓存参数"""
    capacity: int = 1000         # 缓存容量（页面数）
    access_sequence_length: int = 1_000_000  # 访问序列长度
    zipf_s: float = 1.0          # Zipf 分布参数（访问倾斜度）
    key_space: int = 10_000      # 键空间大小


@dataclass
class LRUMetrics:
    capacity: int                # 缓存容量
    access_sequence_length: int  # 访问序列长度
    expected_hit_rate: float     # 期望命中率
    expected_hits: int           # 期望命中次数
    expected_misses: int         # 期望未命中次数
    operation_latency_ns: int    # 操作延迟（纳秒）


def compute_expected_hit_rate(spec: LRUSpec) -> float:
    """
    期望命中率。LRU 的命中率与缓存容量和访问分布有关。
    简化模型：如果键空间均匀访问，命中率 ≈ C / key_space。
    Zipf 分布下，热门页面的命中率更高。
    """
    # 均匀分布近似：命中率 = C / key_space
    uniform_rate = min(1.0, spec.capacity / spec.key_space)

    # Zipf 分布修正：s 越大，访问越集中，命中率越高
    # 经验公式：命中率 ≈ uniform_rate × (1 + s)
    zipf_factor = 1 + spec.zipf_s
    hit_rate = min(1.0, uniform_rate * zipf_factor)

    return hit_rate


def compute_operation_latency(spec: LRUSpec) -> int:
    """操作延迟。哈希表查找 O(1)，约 50 ns。链表移动 O(1)，约 20 ns。"""
    return 70  # 哈希查找 + 链表移动


def compute(spec: LRUSpec) -> LRUMetrics:
    """给定参数，递推全部指标。"""
    hit_rate = compute_expected_hit_rate(spec)
    hits = int(spec.access_sequence_length * hit_rate)
    misses = spec.access_sequence_length - hits
    latency = compute_operation_latency(spec)

    return LRUMetrics(
        capacity=spec.capacity,
        access_sequence_length=spec.access_sequence_length,
        expected_hit_rate=hit_rate,
        expected_hits=hits,
        expected_misses=misses,
        operation_latency_ns=latency,
    )


def _selftest():
    """公式自洽性检查"""
    spec = LRUSpec()

    # 命中率应在 0-1 之间
    m = compute(spec)
    assert 0 <= m.expected_hit_rate <= 1, "命中率应在 0-1 之间"

    # 容量越大，命中率越高
    spec_big = LRUSpec(capacity=5000)
    assert compute(spec_big).expected_hit_rate > m.expected_hit_rate, "容量越大，命中率应越高"

    # Zipf 系数越大，命中率越高
    spec_zipf = LRUSpec(zipf_s=2.0)
    assert compute(spec_zipf).expected_hit_rate > m.expected_hit_rate, "Zipf 系数越大，命中率应越高"

    # 命中 + 未命中 = 总访问
    assert m.expected_hits + m.expected_misses == spec.access_sequence_length

    print("selftest 全部通过")
    # 演示
    print(f"\n容量={m.capacity}  访问次数={m.access_sequence_length:,}  "
          f"命中率={m.expected_hit_rate:.2%}")
    print(f"命中={m.expected_hits:,}  未命中={m.expected_misses:,}  "
          f"操作延迟={m.operation_latency_ns} ns")


if __name__ == "__main__":
    _selftest()
