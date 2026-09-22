# -*- coding: utf-8 -*-
"""
布隆过滤器（Bloom Filter）理论性能模型

布隆过滤器是概率型数据结构，用于快速判断"元素可能存在"或"一定不存在"。它是 LSM-tree、数据库、缓存系统的标准组件。

核心结构：
- m 比特的位数组
- k 个独立哈希函数
- 插入：对元素计算 k 个哈希值，将对应位设为 1
- 查询：检查 k 个位是否全为 1，全 1 则"可能存在"，有 0 则"一定不存在"

核心指标：
1. 假阳性率（False Positive Rate）：查询说"可能存在"但实际不在的概率
2. 最优哈希函数数：给定 m（比特数）和 n（元素数），最小化假阳性率的 k
3. 最优比特数：给定 n 和期望假阳性率 p，所需的 m

标准公式：
- 假阳性率：p = (1 - e^{-kn/m})^k
- 最优 k：k = (m/n) × ln2
- 最优 m：m = -n × ln(p) / (ln2)^2

本模块实现这些公式，参数可配。
"""
from dataclasses import dataclass
import math


@dataclass
class BloomFilterSpec:
    """布隆过滤器参数"""
    n: int = 1_000_000           # 预期元素数量
    m: int = 0                   # 比特数（0 表示自动计算）
    k: int = 0                   # 哈希函数数量（0 表示自动计算）
    bits_per_key: int = 10       # 每元素的比特数（m/n），典型 10


@dataclass
class BloomFilterMetrics:
    n: int                       # 元素数量
    m: int                       # 比特数
    k: int                       # 哈希函数数量
    fpr: float                   # 假阳性率
    memory_bytes: int            # 内存占用（字节）
    expected_fills: float        # 期望被设置的位数


def compute_optimal_k(m: int, n: int) -> int:
    """最优哈希函数数。k = (m/n) × ln2。"""
    return max(1, round((m / n) * math.log(2)))


def compute_optimal_m(n: int, fpr: float) -> int:
    """最优比特数。m = -n × ln(p) / (ln2)^2。"""
    return int(math.ceil(-n * math.log(fpr) / (math.log(2) ** 2)))


def compute_fpr(m: int, n: int, k: int) -> float:
    """假阳性率。p = (1 - e^{-kn/m})^k。"""
    return (1 - math.exp(-k * n / m)) ** k


def compute_memory(m: int) -> int:
    """内存占用。m 比特 = m/8 字节。"""
    return m // 8


def compute_expected_fills(m: int, n: int, k: int) -> float:
    """期望被设置的位数。m × (1 - e^{-kn/m})。"""
    return m * (1 - math.exp(-k * n / m))


def compute(spec: BloomFilterSpec) -> BloomFilterMetrics:
    """给定参数，递推全部指标。"""
    # 自动计算 m 和 k
    if spec.m == 0:
        m = spec.n * spec.bits_per_key
    else:
        m = spec.m

    if spec.k == 0:
        k = compute_optimal_k(m, spec.n)
    else:
        k = spec.k

    fpr = compute_fpr(m, spec.n, k)
    memory = compute_memory(m)
    fills = compute_expected_fills(m, spec.n, k)

    return BloomFilterMetrics(
        n=spec.n,
        m=m,
        k=k,
        fpr=fpr,
        memory_bytes=memory,
        expected_fills=fills,
    )


def _selftest():
    """公式自洽性检查"""
    spec = BloomFilterSpec(n=1_000_000, bits_per_key=10)

    # 假阳性率应在 0-1 之间
    m = compute(spec)
    assert 0 < m.fpr < 1, "假阳性率应在 0-1 之间"

    # bits_per_key 越大，假阳性率越低
    spec_more = BloomFilterSpec(n=1_000_000, bits_per_key=20)
    assert compute(spec_more).fpr < m.fpr, "比特数越多，假阳性率应越低"

    # k 太大或太小都会增加假阳性率
    spec_k1 = BloomFilterSpec(n=1_000_000, bits_per_key=10, k=1)
    spec_k10 = BloomFilterSpec(n=1_000_000, bits_per_key=10, k=10)
    assert compute(spec_k1).fpr > m.fpr, "k 太小，假阳性率应更高"
    assert compute(spec_k10).fpr > m.fpr, "k 太大，假阳性率应更高"

    # 最优 k 应使假阳性率最小
    optimal_k = compute_optimal_k(m.m, m.n)
    assert optimal_k == m.k, "自动计算的 k 应为最优"

    print("selftest 全部通过")
    # 演示
    print(f"\nn={m.n:,}  m={m.m:,} bits  k={m.k}  "
          f"假阳性率={m.fpr:.4f}  内存={m.memory_bytes/1024/1024:.2f} MiB")


if __name__ == "__main__":
    _selftest()
