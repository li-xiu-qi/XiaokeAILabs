# -*- coding: utf-8 -*-
"""
HyperLogLog（基数估计）理论性能模型

HyperLogLog 是 Flajolet et al. (2007) 提出的概率型基数估计结构，
用固定且极小的内存（典型 12 KB）统计海量数据中不重复元素的数量。

核心思想（源自 Flajolet-Martin 的"抛硬币"直觉）：
- 一个元素哈希后二进制表示末尾连续 0 的个数，相当于抛硬币连续抛出正面的次数
- 见到 k 个连续 0 的概率是 1/2^k，反推集合大小约为 2^k
- 单次观察方差极大，用 m 个独立桶并行观察，取调和平均（而非算术平均）抑制离群值

核心结构：
- m 个寄存器（bucket），m = 2^p（p 典型 14，m=16384）
- 元素哈希后，前 p 位选桶索引，剩余位中前导零个数 +1 写入该桶（取历史最大值）
- 基数估计：E = α_m × m² / Σ(2^(-M[j]))，α_m 是修正常数

核心指标：
1. 标准误差：σ = 1.04/√m，m=16384 时约 0.81%
2. 空间成本：m × 5 bits（寄存器值 0-32 需 5 bit，实际用 6 bit 更安全）
3. 小基数修正：E < 2.5m 且有空桶时用线性计数（Linear Counting）
4. 大基数修正：E > 2^32/30 时修正 32 位哈希饱和

标准误差来源：取倒数求和的调和平均，其倒数近似服从指数分布，
调和平均的标准误约为 1.04/√m，这个 1.04 是实测常数。

本模块实现这些公式，参数可配。
"""
from dataclasses import dataclass
import math


@dataclass
class HyperLogLogSpec:
    """HyperLogLog 参数"""
    p: int = 14                  # 桶数的指数，m = 2^p（典型 14）
    bits_per_register: int = 6   # 每个寄存器的比特数
    hash_bits: int = 32          # 哈希值位数（用于大基数修正）


@dataclass
class HyperLogLogMetrics:
    p: int                       # 桶数指数
    m: int                       # 桶数 2^p
    bits_per_register: int       # 每寄存器比特数
    standard_error: float        # 标准误差 1.04/√m
    alpha: float                 # 修正常数 α_m
    memory_bits: int             # 总比特数 m × bits_per_register
    memory_bytes: int            # 总字节数
    relative_error_99: float     # 99% 置信区间半宽（2.576σ）


def compute_alpha(m: int) -> float:
    """修正常数 α_m。

    m=16 → 0.673, m=32 → 0.697, m=64 → 0.709,
    m >= 128 → 0.7213/(1 + 1.079/m)
    来源：Flajolet et al. (2007) 通过模拟标定，用于修正调和平均的系统偏差。
    """
    if m == 16:
        return 0.673
    if m == 32:
        return 0.697
    if m == 64:
        return 0.709
    if m >= 128:
        return 0.7213 / (1.0 + 1.079 / m)
    # m < 16 时论文未给标定值，按渐近式外推（仅用于完整性）
    return 0.7213 / (1.0 + 1.079 / m)


def compute_standard_error(m: int) -> float:
    """标准误差。σ = 1.04/√m。"""
    return 1.04 / math.sqrt(m)


def compute_raw_estimate(registers: list) -> float:
    """原始估计值。E = α_m × m² / Σ(2^(-M[j]))。

    全零寄存器时调和和为 m，返回 α×m，此时应由 compute_cardinality
    转交线性计数接管，本函数不做截断。
    """
    m = len(registers)
    alpha = compute_alpha(m)
    harmonic_sum = sum(2.0 ** (-r) for r in registers)
    if harmonic_sum == 0:
        return 0.0
    return alpha * (m ** 2) / harmonic_sum


def compute_linear_counting_estimate(m: int, num_empty: int) -> float:
    """小基数修正：线性计数。

    当原始估计值 < 2.5m 且有空桶时，用空桶比例反推：
    E* = m × ln(m/V)，V 为空桶数。
    依据：泊松过程下空桶期望比例 e^(-n/m)，反解 n。
    """
    if num_empty == 0:
        return float(m)
    return m * math.log(m / num_empty)


def compute_large_range_correction(raw_estimate: float,
                                   hash_bits: int = 32) -> float:
    """大基数修正。当 E > 2^hash_bits/30 时，原始估计因哈希碰撞开始偏低。

    E* = -2^32 × ln(1 - E/2^32)
    """
    threshold = (2 ** hash_bits) / 30.0
    if raw_estimate <= threshold:
        return raw_estimate
    ratio = raw_estimate / (2 ** hash_bits)
    if ratio >= 1.0:
        return raw_estimate
    return -(2 ** hash_bits) * math.log(1.0 - ratio)


def compute_cardinality(registers: list, hash_bits: int = 32) -> float:
    """完整基数估计（含小基数与大基数修正）。"""
    m = len(registers)
    raw = compute_raw_estimate(registers)
    num_empty = sum(1 for r in registers if r == 0)

    # 小基数修正
    if num_empty > 0 and raw < 2.5 * m:
        return compute_linear_counting_estimate(m, num_empty)

    # 大基数修正
    return compute_large_range_correction(raw, hash_bits)


def compute_memory_bits(m: int, bits_per_register: int) -> int:
    """总比特数。m × bits_per_register。"""
    return m * bits_per_register


def compute_memory_bytes(m: int, bits_per_register: int) -> int:
    """总字节数。向上取整到整字节。"""
    return math.ceil(m * bits_per_register / 8)


def compute_confidence_interval(m: int, z: float = 1.96) -> tuple:
    """置信区间半宽。z=1.96 对应 95%，z=2.576 对应 99%。"""
    sigma = compute_standard_error(m)
    return (-z * sigma, z * sigma)


def compute_max_countable(hash_bits: int = 32) -> int:
    """可估计的最大基数。受哈希位数限制，超过后碰撞主导，估计失效。"""
    return 2 ** hash_bits


def compute(spec: HyperLogLogSpec) -> HyperLogLogMetrics:
    """给定参数，递推全部指标。"""
    m = 2 ** spec.p
    alpha = compute_alpha(m)
    sigma = compute_standard_error(m)
    memory_bits = compute_memory_bits(m, spec.bits_per_register)

    return HyperLogLogMetrics(
        p=spec.p,
        m=m,
        bits_per_register=spec.bits_per_register,
        standard_error=sigma,
        alpha=alpha,
        memory_bits=memory_bits,
        memory_bytes=compute_memory_bytes(m, spec.bits_per_register),
        relative_error_99=2.576 * sigma,
    )


def _selftest():
    """公式自洽性检查"""
    spec = HyperLogLogSpec(p=14, bits_per_register=6)
    m = compute(spec)

    # m 应为 2^p
    assert m.m == 2 ** 14, "m 应为 2^14"

    # 标准误差应在合理范围
    assert 0 < m.standard_error < 0.05, "标准误差应在 0-5% 之间"

    # p 越大，m 越大，标准误差越小
    m_small = compute(HyperLogLogSpec(p=10))
    m_large = compute(HyperLogLogSpec(p=16))
    assert m_small.m < m.m < m_large.m, "p 越大，m 应越大"
    assert m_small.standard_error > m.standard_error > m_large.standard_error, \
        "p 越大，标准误差应越小"

    # 内存应随 p 指数增长：p 从 10 到 16 增加 6，m 增 2^6 = 64 倍
    assert m_small.memory_bits * 64 == m_large.memory_bits, \
        "p 增加 6，内存应增 64 倍"

    # α 应随 m 增大趋近 0.7213
    assert compute_alpha(16) < compute_alpha(64) < compute_alpha(16384), \
        "α 应随 m 增大趋近渐近值"

    # 线性计数：空桶越多，估计越小
    est_full = compute_linear_counting_estimate(1024, 1)
    est_empty = compute_linear_counting_estimate(1024, 1023)
    assert est_full > est_empty, "空桶越多，线性计数估计应越小"

    # 大基数修正：超过阈值后应放大估计
    raw = 2 ** 32 / 20.0
    assert compute_large_range_correction(raw) > raw, "大基数修正应放大估计"
    raw_small = 1000.0
    assert compute_large_range_correction(raw_small) == raw_small, \
        "低于阈值不应修正"

    # 调和平均：全零寄存器时调和和为 m，返回 α×m（小基数边界值）
    assert abs(compute_raw_estimate([0] * 10) - compute_alpha(10) * 10) < 1e-9, \
        "全零寄存器应返回 α×m"

    print("selftest 全部通过")
    print(f"\np={m.p}  m={m.m:,}  α={m.alpha:.6f}")
    print(f"标准误差={m.standard_error:.4%}  99% 置信区间半宽={m.relative_error_99:.4%}")
    print(f"内存={m.memory_bits:,} bits = {m.memory_bytes:,} bytes = "
          f"{m.memory_bytes/1024:.1f} KiB")


if __name__ == "__main__":
    _selftest()
