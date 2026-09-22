# -*- coding: utf-8 -*-
"""
Space-Saving 理论性能模型

Space-Saving（Metwally, Agrawal & El Abbadi 2005）是确定性的流式 Heavy Hitters
算法，维护一个固定容量 m 的 Stream Summary，每个计数器是一个三元组
(element, count, error)。

核心结构：
- m 个三元组，count 为估计频次，error 为该元素进入摘要时继承的最小计数值
- 更新(i)：若 i 在摘要中则其 count +1；否则找到 count 最小的计数器，
  把它替换为 (i, 1, 原最小计数值)
- 查询(i)：返回摘要中 i 的 count；不在则返回 0

核心指标：
1. 确定性保证：频率 > n/m 的元素一定在摘要中
2. 误差界：估计值 <= 真实频率 <= 估计值 + n/m（双向有界）
3. 每次更新 O(1)（用最小堆管理 count，替换时弹堆顶）
4. 空间 O(m)，m = ceil(1/epsilon) 时误差界为 epsilon*n

关键性质：
- 所有计数器的 count 之和不超过 n。每次命中 +1，每次替换把总和从 S 降到
  S - min_count + 1（min_count 为被替换项的原计数），所以总和只减不增。
  上界 n 是误差界推导的抓手。
- 估计值永不超过真实值：count 每次只在实际命中时 +1，而替换时重置为 1，
  所以 count <= 该元素至今的真实出现次数。这一点与 Metwally 原文变体
  （替换为 (element, min_count + 1, min_count)）不同，后者总和恒等于 n，
  但替换瞬间的 count 会略高于真实值。
- 从未被替换过的计数器，其 count 就是真实频次（误差为 0）
- 相比 Misra-Gries，误差界从单侧变成双侧，上界同为 n/m

本模块实现这些公式，参数可配。
"""
from dataclasses import dataclass
import math


@dataclass
class SpaceSavingSpec:
    """Space-Saving 参数"""
    n: int = 1_000_000           # 流长度（总更新次数 N）
    m: int = 0                   # 摘要容量（0 表示按 epsilon 自动计算）
    epsilon: float = 0.01        # 相对误差 ε（误差界 = εN）
    universe_bits: int = 32      # 元素 ID 所需比特数（log2 字典大小 U）


@dataclass
class SpaceSavingMetrics:
    n: int                       # 流长度 N
    m: int                       # 摘要容量
    epsilon: float               # 实际相对误差 1/m
    error_bound: int             # 绝对误差界 n/m
    guarantee_threshold: int     # 保证被追踪的频率阈值 floor(n/m)+1
    counter_bits: int            # 单计数器所需比特数（计数上限 n）
    entry_bits: int              # 单个摘要项比特数（ID + count + error）
    space_bits: int              # 总比特数
    space_bytes: int             # 总字节数
    entries_per_kb: float        # 每 KiB 可容纳的摘要项数
    counter_sum_bound: int        # 计数总和上界（不超过 N）


def compute_m(epsilon: float) -> int:
    """摘要容量。m = ceil(1/epsilon)。"""
    return int(math.ceil(1.0 / epsilon))


def compute_epsilon(m: int) -> float:
    """由 m 反推实际误差。epsilon = 1/m。"""
    return 1.0 / m


def compute_error_bound(n: int, m: int) -> int:
    """绝对误差界。误差 <= n/m。"""
    return n // m


def compute_guarantee_threshold(n: int, m: int) -> int:
    """保证被追踪的频率阈值。频率 > n/m 即 floor(n/m)+1。"""
    return n // m + 1


def compute_counter_bits(n: int) -> int:
    """单计数器所需比特数。count 上限为 n，需 ceil(log2(n+1))。"""
    if n <= 1:
        return 1
    return int(math.ceil(math.log2(n + 1)))


def compute_entry_bits(n: int, universe_bits: int) -> int:
    """单个摘要项比特数。元素 ID + count + error 两个计数位宽。"""
    return universe_bits + 2 * compute_counter_bits(n)


def compute_space_bits(m: int, n: int, universe_bits: int) -> int:
    """总比特数。m × entry_bits。"""
    return m * compute_entry_bits(n, universe_bits)


def compute_space_bytes(m: int, n: int, universe_bits: int) -> int:
    """总字节数。向上取整到整字节。"""
    return math.ceil(compute_space_bits(m, n, universe_bits) / 8)


def compute_entries_per_kb(m: int, n: int, universe_bits: int) -> float:
    """每 KiB 可容纳的摘要项数。"""
    byts = compute_space_bytes(m, n, universe_bits)
    if byts == 0:
        return 0.0
    return m / (byts / 1024)


def compute_counter_sum_upper_bound(n: int) -> int:
    """计数总和的上界。替换使总和下降，故总和恒不超过 n。"""
    return n


def compute(spec: SpaceSavingSpec) -> SpaceSavingMetrics:
    """给定参数，递推全部指标。"""
    m = compute_m(spec.epsilon) if spec.m == 0 else spec.m
    epsilon = spec.epsilon if spec.m == 0 else compute_epsilon(m)

    err = compute_error_bound(spec.n, m)
    thr = compute_guarantee_threshold(spec.n, m)
    c_bits = compute_counter_bits(spec.n)
    e_bits = compute_entry_bits(spec.n, spec.universe_bits)
    s_bits = compute_space_bits(m, spec.n, spec.universe_bits)
    s_bytes = compute_space_bytes(m, spec.n, spec.universe_bits)
    epk = compute_entries_per_kb(m, spec.n, spec.universe_bits)

    return SpaceSavingMetrics(
        n=spec.n,
        m=m,
        epsilon=epsilon,
        error_bound=err,
        guarantee_threshold=thr,
        counter_bits=c_bits,
        entry_bits=e_bits,
        space_bits=s_bits,
        space_bytes=s_bytes,
        entries_per_kb=epk,
        counter_sum_bound=compute_counter_sum_upper_bound(spec.n),
    )


def _selftest():
    """公式自洽性检查"""
    spec = SpaceSavingSpec(n=1_000_000, epsilon=0.01)
    m = compute(spec)

    # m 与 epsilon 互为倒数
    assert m.m == 100, f"ε=0.01 时 m 应为 100，实际 {m.m}"
    assert abs(m.epsilon - 0.01) < 1e-12, "epsilon 应为 1/m"

    # 误差界 = n/m
    assert m.error_bound == 10_000, f"误差界应为 10000，实际 {m.error_bound}"

    # 保证阈值 = floor(n/m) + 1
    assert m.guarantee_threshold == 10_001, \
        f"保证阈值应为 10001，实际 {m.guarantee_threshold}"

    # epsilon 越小，m 越大
    m_loose = compute_m(0.1)
    m_tight = compute_m(0.001)
    assert m_tight > m_loose, "ε 越小，m 应越大"

    # 内存应随 m 线性增长
    m50 = compute(SpaceSavingSpec(n=1_000_000, m=50))
    m100 = compute(SpaceSavingSpec(n=1_000_000, m=100))
    assert m100.space_bits == 2 * m50.space_bits, "m 翻倍，内存应翻倍"

    # 摘要项比 Misra-Gries 多一个 error 位宽
    assert m.entry_bits == 32 + 2 * m.counter_bits, \
        "摘要项应为 ID + count + error"

    # 计数总和上界恒为 n（实际不超过 n）
    assert m.counter_sum_bound == m.n, "计数总和上界应为 n"

    # 误差界必须小于 n
    assert 0 < m.error_bound < m.n, "误差界应在 (0, n) 内"

    # 与 Misra-Gries 同 ε 同 N 下误差界一致
    from misra_gries_model import compute as mg_compute, MisraGriesSpec
    mg = mg_compute(MisraGriesSpec(n=1_000_000, epsilon=0.01))
    assert mg.error_bound == m.error_bound, "同参数下两算法误差界应一致"

    print("selftest 全部通过")
    print(f"\nN={m.n:,}  ε={m.epsilon}  m={m.m}")
    print(f"绝对误差界={m.error_bound:,}  保证阈值={m.guarantee_threshold:,}  "
          f"计数器位宽={m.counter_bits} bit")
    print(f"摘要项={m.entry_bits} bit  总空间={m.space_bits/8/1024:.2f} KiB  "
          f"({m.entries_per_kb:.1f} 摘要项/KiB)")
    print(f"计数总和上界={m.counter_sum_bound:,}（不超过 N）")


if __name__ == "__main__":
    _selftest()
