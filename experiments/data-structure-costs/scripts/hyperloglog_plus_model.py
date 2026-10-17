# -*- coding: utf-8 -*-
"""
HyperLogLog++（基数估计，工程化改进版）理论性能模型

HyperLogLog++ 是 Heule, Nunkesser & Hall (2013) 在原始 HyperLogLog 上做的
工程化改进，Google 在 PowerDrill 中使用，也是 Apache DataSketches 与 Redis 6.0+
的默认实现。相对原版的四处改动：

1. 64 位哈希替代 32 位，可估计上限从 2^32 提到 2^64
2. 小基数时用稀疏表示（sparse representation），只存非零寄存器，用 (idx, rank)
   键值对替代完整的 m 长度数组；基数增长触发密度阈值后切换到密集表示
3. 偏差校正：用实测数据建立的 (rawEstimate, bias) 对照表替代理论常数，
   在原始估计 <= 5m 的区间用 k 近邻插值修正系统性高估
4. 内存从"固定由 m 决定"变成"基数小时随基数增长"，这是小基数场景的核心收益

核心指标：
1. 标准误差：σ = 1.04/√m，与原版相同（寄存器算术没变）
2. 稀疏→密集切换点：元素数达到 m × k/32（k=6 时约 m/5）
3. 稀疏内存：n_sparse × (p + 6) 比特，与基数成正比
4. 稀疏估计：线性计数 E = m × ln(m/V)，V = m - 不同桶索引数

稀疏表示的精度来源：基数 n 远小于 m 时，(idx, rank) 列表里的不同桶索引数
t 几乎等于 n（碰撞极少），于是空桶数 V = m - t 是真实空桶数的上界估计，
线性计数给出的基数估计也就偏保守且方差小。这正是切换点取 m/5 的原因：
超过这个点之后 t 开始显著小于 n，V 被低估，估计偏差快速放大。

偏差校正的适用范围：原始 HLL 估计在 2.5m 到 5m 之间会系统性高估
（哈希碰撞与调和平均的非线性共同导致），HLL++ 用 k 近邻（k=6）在偏差表上
插值扣除。估计超过 5m 后偏差可忽略，直接返回原始值。

本模块实现这些公式，参数可配。
"""
from dataclasses import dataclass
import math


# 稀疏→密集切换系数。论文取 k=6，切换点为 m × k/32 ≈ m/5.33
SPARSE_K = 6

# 偏差校正的生效区间上界：原始估计 <= 5m 时才查表
BIAS_UPPER_R = 5.0

# 线性计数接管的原始估计上界：E < 2.5m 且有空桶时转线性计数
LINEAR_COUNTING_R = 2.5

# 偏差代理曲面的形状参数，仅用于验证插值与门控机制
BIAS_CENTER = 3.3      # 偏差峰值所在的 r = raw/m
BIAS_WIDTH = 1.0       # 曲面宽度
BIAS_AMPLITUDE = 0.15  # 峰值偏差占 m 的比例（约对应真实基数的几个百分点）

# k 近邻插值的邻居数，论文固定取 6
BIAS_NEIGHBORS = 6


@dataclass
class HyperLogLogPlusSpec:
    """HyperLogLog++ 参数"""
    p: int = 14                  # 桶数的指数，m = 2^p（典型 14）
    bits_per_register: int = 6   # 每个寄存器的比特数
    hash_bits: int = 64          # 哈希值位数（HLL++ 用 64 位）
    sparse_k: int = SPARSE_K     # 稀疏切换系数 k
    sparse_entries: int = 0      # 当前稀疏条目数（0 表示只算结构指标）


@dataclass
class HyperLogLogPlusMetrics:
    p: int                       # 桶数指数
    m: int                       # 桶数 2^p
    bits_per_register: int       # 每寄存器比特数
    hash_bits: int               # 哈希位数
    standard_error: float        # 标准误差 1.04/√m
    alpha: float                 # 修正常数 α_m
    dense_memory_bits: int       # 密集表示总比特数 m × bits_per_register
    dense_memory_bytes: int      # 密集表示总字节数
    sparse_threshold: int        # 稀疏→密集切换点（元素数）
    sparse_entry_bits: int       # 单个稀疏条目的比特数
    sparse_memory_bits: int      # 当前稀疏内存比特数
    sparse_memory_bytes: int     # 当前稀疏内存字节数
    memory_saving_ratio: float   # 稀疏相对密集的压缩比
    relative_error_99: float     # 99% 置信区间半宽（2.576σ）
    max_countable: int           # 可估计的最大基数 2^hash_bits


def compute_alpha(m: int) -> float:
    """修正常数 α_m，与原版 HyperLogLog 相同。"""
    if m == 16:
        return 0.673
    if m == 32:
        return 0.697
    if m == 64:
        return 0.709
    return 0.7213 / (1.0 + 1.079 / m)


def compute_standard_error(m: int) -> float:
    """标准误差。σ = 1.04/√m。"""
    return 1.04 / math.sqrt(m)


def compute_raw_estimate(registers: list) -> float:
    """原始估计值。E = α_m × m² / Σ(2^(-M[j]))。"""
    m = len(registers)
    alpha = compute_alpha(m)
    harmonic_sum = sum(2.0 ** (-r) for r in registers)
    if harmonic_sum == 0:
        return 0.0
    return alpha * (m ** 2) / harmonic_sum


def compute_linear_counting_estimate(m: int, num_empty: int) -> float:
    """线性计数。E* = m × ln(m/V)，V 为空桶数。"""
    if num_empty <= 0:
        return float(m)
    return m * math.log(m / num_empty)


def compute_large_range_correction(raw_estimate: float,
                                   hash_bits: int = 64) -> float:
    """大基数修正。E > 2^hash_bits/30 时哈希碰撞主导，原始估计偏低。"""
    threshold = (2 ** hash_bits) / 30.0
    if raw_estimate <= threshold:
        return raw_estimate
    ratio = raw_estimate / (2 ** hash_bits)
    if ratio >= 1.0:
        return raw_estimate
    return -(2 ** hash_bits) * math.log(1.0 - ratio)


def compute_sparse_threshold(m: int, k: int = SPARSE_K) -> int:
    """稀疏→密集切换点（元素数）。m × k/32。

    k=6 时约 m/5.33。论文取这个值是因为超过该点后不同桶索引数开始
    显著小于元素数，线性计数依赖的空桶估计随之失真。
    """
    return m * k // 32


def compute_sparse_entry_bits(p: int, bits_per_register: int) -> int:
    """单个稀疏条目的比特数：桶索引 p 位 + 寄存器值 bits_per_register 位。"""
    return p + bits_per_register


def compute_sparse_memory_bits(num_entries: int, p: int,
                               bits_per_register: int) -> int:
    """稀疏表示总比特数。与条目数成正比，条目数上界为 m。"""
    return num_entries * compute_sparse_entry_bits(p, bits_per_register)


def compute_sparse_memory_bytes(num_entries: int, p: int,
                                bits_per_register: int) -> int:
    """稀疏表示总字节数。向上取整到整字节。"""
    return math.ceil(compute_sparse_memory_bits(num_entries, p,
                                                bits_per_register) / 8)


def compute_sparse_estimate(m: int, distinct_indices: int) -> float:
    """稀疏模式下的基数估计：线性计数。

    只存非零寄存器时拿不到真实空桶数，用不同桶索引数 t 代替占用桶数，
    得 V = m - t 作为真实空桶数的下界，因此估计值偏保守（偏高）。
    t 接近元素数时（基数远小于 m）这个偏差可忽略。
    """
    if distinct_indices <= 0:
        return 0.0
    if distinct_indices >= m:
        # 所有桶都被占满，线性计数失效，退化为密集估计的下界
        return float(m)
    return compute_linear_counting_estimate(m, m - distinct_indices)


def compute_bias_ratio(raw_over_m: float) -> float:
    """偏差占 m 的比例，参数化代理曲面。

    论文的偏差表由约 1.4 万次模拟标定，逐点数值无法用闭式给出。
    本函数用一条高斯形曲面代替，只保留三个论文明确的性质：
    原始估计低于 2.5m 时不修正（该区间由线性计数接管）、
    偏差在 r≈3.3 附近达到峰值、超过 5m 后衰减到可忽略。
    具体偏差数值以实测为准，本函数只服务于插值与门控机制的自测。
    """
    if raw_over_m < LINEAR_COUNTING_R or raw_over_m > BIAS_UPPER_R:
        return 0.0
    x = (raw_over_m - BIAS_CENTER) / BIAS_WIDTH
    return BIAS_AMPLITUDE * math.exp(-0.5 * x * x)


def build_bias_table(m: int, num_anchors: int = 32) -> list:
    """建立 (rawEstimate, bias) 锚点表。

    锚点按 r = raw/m 均匀分布在 1.0 到 6.0，覆盖线性计数接管的边界
    到偏差完全消失的区间。实际工程实现里这张表由模拟数据填。
    """
    table = []
    for i in range(num_anchors):
        r = 1.0 + (6.0 - 1.0) * i / (num_anchors - 1)
        table.append((r * m, compute_bias_ratio(r) * m))
    return table


def compute_bias(raw_estimate: float, m: int, table: list,
                 k: int = BIAS_NEIGHBORS) -> float:
    """k 近邻插值查偏差表，对应论文的 EstimateBias。

    取 rawEstimate 最接近的 k 个锚点，用它们偏差的算术平均作为修正量。
    只在 raw <= 5m 时生效，超过后偏差表不再有信息。
    """
    if raw_estimate > BIAS_UPPER_R * m:
        return 0.0
    if not table:
        return 0.0
    ranked = sorted(table, key=lambda a: abs(a[0] - raw_estimate))
    neighbors = ranked[:min(k, len(ranked))]
    return sum(b for _, b in neighbors) / len(neighbors)


def compute_bias_corrected_estimate(raw_estimate: float, m: int,
                                    table: list) -> float:
    """偏差校正后的基数估计。E* = raw - bias(raw)。"""
    return raw_estimate - compute_bias(raw_estimate, m, table)


def compute_cardinality_plus(m: int, distinct_indices: int,
                             registers: list = None, k: int = SPARSE_K,
                             table: list = None, hash_bits: int = 64,
                             use_bias: bool = True) -> float:
    """HyperLogLog++ 完整基数估计。

    先按稀疏/密集分流：条目数未过切换点时走稀疏线性计数，
    过了切换点走密集路径，密集路径内再分线性计数、偏差校正、大基数修正三层。
    """
    threshold = compute_sparse_threshold(m, k)

    # 稀疏模式
    if distinct_indices < threshold:
        return compute_sparse_estimate(m, distinct_indices)

    # 密集模式
    if registers is None:
        return 0.0
    raw = compute_raw_estimate(registers)
    num_empty = sum(1 for r in registers if r == 0)

    if num_empty > 0 and raw < LINEAR_COUNTING_R * m:
        return compute_linear_counting_estimate(m, num_empty)

    if use_bias:
        est = compute_bias_corrected_estimate(raw, m, table or [])
    else:
        est = raw

    return compute_large_range_correction(est, hash_bits)


def compute_memory_saving_ratio(num_entries: int, m: int, p: int,
                                bits_per_register: int) -> float:
    """稀疏相对密集的空间压缩比。>1 表示稀疏更省。"""
    dense = m * bits_per_register
    sparse = compute_sparse_memory_bits(num_entries, p, bits_per_register)
    if sparse == 0:
        return float("inf")
    return dense / sparse


def compute(spec: HyperLogLogPlusSpec) -> HyperLogLogPlusMetrics:
    """给定参数，递推全部指标。"""
    m = 2 ** spec.p
    alpha = compute_alpha(m)
    sigma = compute_standard_error(m)
    dense_bits = compute_memory_bits(m, spec.bits_per_register)
    threshold = compute_sparse_threshold(m, spec.sparse_k)
    entry_bits = compute_sparse_entry_bits(spec.p, spec.bits_per_register)
    sparse_bits = compute_sparse_memory_bits(spec.sparse_entries, spec.p,
                                             spec.bits_per_register)

    return HyperLogLogPlusMetrics(
        p=spec.p,
        m=m,
        bits_per_register=spec.bits_per_register,
        hash_bits=spec.hash_bits,
        standard_error=sigma,
        alpha=alpha,
        dense_memory_bits=dense_bits,
        dense_memory_bytes=compute_memory_bytes(m, spec.bits_per_register),
        sparse_threshold=threshold,
        sparse_entry_bits=entry_bits,
        sparse_memory_bits=sparse_bits,
        sparse_memory_bytes=compute_sparse_memory_bytes(
            spec.sparse_entries, spec.p, spec.bits_per_register),
        memory_saving_ratio=compute_memory_saving_ratio(
            spec.sparse_entries, m, spec.p, spec.bits_per_register),
        relative_error_99=2.576 * sigma,
        max_countable=2 ** spec.hash_bits,
    )


def compute_memory_bits(m: int, bits_per_register: int) -> int:
    """密集表示总比特数。m × bits_per_register。"""
    return m * bits_per_register


def compute_memory_bytes(m: int, bits_per_register: int) -> int:
    """密集表示总字节数。向上取整到整字节。"""
    return math.ceil(m * bits_per_register / 8)


def _selftest():
    """公式自洽性检查"""
    spec = HyperLogLogPlusSpec(p=14, sparse_entries=1000)
    m = compute(spec)

    # m 应为 2^p
    assert m.m == 2 ** 14, "m 应为 2^14"

    # 标准误差与原版一致
    assert abs(m.standard_error - 1.04 / math.sqrt(m.m)) < 1e-12, \
        "标准误差应为 1.04/√m"
    assert 0 < m.standard_error < 0.05, "标准误差应在 0-5% 之间"

    # 64 位哈希把可估计上限提到 2^64
    assert m.max_countable == 2 ** 64, "64 位哈希上限应为 2^64"

    # 切换点应为 m × k/32
    assert m.sparse_threshold == m.m * SPARSE_K // 32, \
        "切换点应为 m × k/32"

    # 稀疏内存随条目数线性增长，且在条目少时远小于密集内存
    small = compute(HyperLogLogPlusSpec(p=14, sparse_entries=100))
    large = compute(HyperLogLogPlusSpec(p=14, sparse_entries=3000))
    assert small.sparse_memory_bits * 30 == large.sparse_memory_bits, \
        "稀疏内存应随条目数线性增长"
    assert small.sparse_memory_bytes < small.dense_memory_bytes, \
        "条目少时稀疏内存应远小于密集内存"

    # 切换点处稀疏内存仍应小于密集内存
    assert large.sparse_memory_bytes < large.dense_memory_bytes, \
        "切换点处稀疏内存应仍小于密集内存"

    # 单个条目比特数 = p + 寄存器位数
    assert m.sparse_entry_bits == m.p + m.bits_per_register, \
        "条目比特数应为 p + bits_per_register"

    # 稀疏估计：不同桶索引越多，估计越大
    assert compute_sparse_estimate(16384, 1000) > compute_sparse_estimate(16384, 100), \
        "不同桶索引越多，稀疏估计应越大"

    # 稀疏估计在小基数下应接近真实基数
    est = compute_sparse_estimate(16384, 1000)
    assert abs(est - 1000) / 1000 < 0.05, \
        f"基数 1000 时稀疏估计应接近真实值，实际 {est:.1f}"

    # 偏差校正：只在 raw <= 5m 时生效
    table = build_bias_table(16384)
    assert compute_bias(3.3 * 16384, 16384, table) > 0, \
        "偏差峰值区间应有正偏差"
    assert compute_bias(6.0 * 16384, 16384, table) == 0, \
        "超过 5m 不应修正"
    corrected = compute_bias_corrected_estimate(3.3 * 16384, 16384, table)
    assert corrected < 3.3 * 16384, "偏差校正应压低估计"

    # k 近邻插值结果应落在邻居偏差的最小与最大之间
    biases = sorted(b for _, b in table)[:6]
    lo, hi = min(biases), max(biases)
    assert lo <= compute_bias(1.05 * 16384, 16384, table) <= hi, \
        "插值结果应落在邻居取值范围内"

    # 大基数修正：超过阈值后应放大估计
    raw = 2 ** 64 / 20.0
    assert compute_large_range_correction(raw) > raw, "大基数修正应放大估计"

    # 稀疏/密集分流的边界行为
    thr = compute_sparse_threshold(16384)
    assert compute_cardinality_plus(16384, thr - 1, [0] * 16384,
                                    table=table) > 0, "稀疏路径应返回估计"
    dense_est = compute_cardinality_plus(16384, thr, [0] * 16384, table=table)
    assert dense_est == 0.0, "全零寄存器的密集估计应为 0"

    print("selftest 全部通过")
    print(f"\np={m.p}  m={m.m:,}  α={m.alpha:.6f}  哈希={m.hash_bits} 位")
    print(f"标准误差={m.standard_error:.4%}  99% 置信区间半宽={m.relative_error_99:.4%}")
    print(f"密集内存={m.dense_memory_bits:,} bits = {m.dense_memory_bytes:,} B")
    print(f"稀疏→密集切换点={m.sparse_threshold:,} 个元素 "
          f"(m × {SPARSE_K}/32)")
    print(f"稀疏条目={spec.sparse_entries:,} 时内存="
          f"{m.sparse_memory_bits:,} bits = {m.sparse_memory_bytes:,} B，"
          f"压缩比 {m.memory_saving_ratio:.2f}x")


if __name__ == "__main__":
    _selftest()
