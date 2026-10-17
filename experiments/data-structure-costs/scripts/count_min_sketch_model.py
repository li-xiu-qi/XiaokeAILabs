# -*- coding: utf-8 -*-
"""
Count-Min Sketch（计数最小草图）理论性能模型

Count-Min Sketch 是 Cormode & Muthukrishnan (2005) 提出的概率型频次统计结构。
它用远小于真实数据量的空间，给出每个元素的出现频率的上界估计，保证永不高估真实值。

核心结构：
- d 行 w 列的二维计数数组（共 d×w 个计数器）
- d 个两两独立（pairwise independent）的哈希函数，每行一个
- 更新(i, c)：对每行 r，把计数器 C[r][h_r(i)] 加上 c
- 查询(i)：取 d 行对应位置的最小值 min_r C[r][h_r(i)] 作为估计值

核心指标：
1. 误差界：估计值 <= 真实值 + εN，成立概率 >= 1-δ（N 为总更新次数）
2. 误差只朝一个方向：sketch 只会高估，不会低估真实频次（point query 永不出错下限）
3. 参数：w = ceil(e/ε)，d = ceil(ln(1/δ))
4. 空间 = d × w × counter_bits 比特

关键性质：
- 取最小值而非平均值：哈希冲突只会把计数器推高，取 min 能滤掉被污染最严重的行
- 支持加性合并（可分布式聚合）：两个 sketch 逐格相加即得合并流上的 sketch
- 单调性：更新只增不减，适合"只增"的频次统计

本模块实现这些公式，参数可配。
"""
from dataclasses import dataclass
import math


@dataclass
class CountMinSketchSpec:
    """Count-Min Sketch 参数"""
    n: int = 1_000_000           # 预期总更新次数（流长度 N）
    epsilon: float = 0.01        # 相对误差 ε（估计偏差不超过 εN）
    delta: float = 0.001         # 失败概率 δ（误差界以 >= 1-δ 概率成立）
    counter_bits: int = 32       # 每个计数器比特数
    width: int = 0               # 宽度 w（0 表示按 ε 自动计算）
    depth: int = 0               # 深度 d（0 表示按 δ 自动计算）


@dataclass
class CountMinSketchMetrics:
    n: int                       # 总更新次数 N
    width: int                   # 每行计数器数 w
    depth: int                   # 行数 d
    epsilon: float               # 相对误差
    delta: float                 # 失败概率
    counter_bits: int            # 每计数器比特数
    absolute_error: float        # 绝对误差界 εN
    num_counters: int            # 计数器总数 d×w
    memory_bits: int             # 总比特数
    memory_bytes: int            # 总字节数
    counter_max: int             # 单个计数器最大值


def compute_width(epsilon: float) -> int:
    """宽度。w = ceil(e/ε)。e 为自然常数。"""
    return int(math.ceil(math.e / epsilon))


def compute_depth(delta: float) -> float:
    """深度。d = ceil(ln(1/δ))。"""
    return int(math.ceil(math.log(1.0 / delta)))


def compute_absolute_error(epsilon: float, n: int) -> float:
    """绝对误差界。误差 <= εN。"""
    return epsilon * n


def compute_num_counters(width: int, depth: int) -> int:
    """计数器总数。d × w。"""
    return width * depth


def compute_memory_bits(width: int, depth: int, counter_bits: int) -> int:
    """总比特数。d × w × counter_bits。"""
    return width * depth * counter_bits


def compute_memory_bytes(width: int, depth: int, counter_bits: int) -> int:
    """总字节数。向上取整到整字节。"""
    return math.ceil(width * depth * counter_bits / 8)


def compute_relative_error_bound(width: int, n: int) -> float:
    """给定 w 与 N 反推相对误差。ε = e/w，误差 <= εN = eN/w。"""
    return math.e / width


def compute_counter_max(counter_bits: int) -> int:
    """单个计数器可表示的最大值。2^bits - 1。"""
    return (1 << counter_bits) - 1


def compute_overflow_probability(n: int, width: int, depth: int) -> float:
    """计数器溢出概率的上界估计。

    某行某个计数器溢出要求该格收到的更新数超过 2^bits。
    上界：d × w × (eN/w)^bits / bits! 量级，这里给出简化上界
    d * (eN/w)^bits / bits!。
    """
    mean = n / width
    if mean < 1:
        return 0.0
    # 泊松近似下，单格收到 >= K 次的概率上界
    k = compute_counter_max(32)
    try:
        log_p = k * math.log(mean) - mean - sum(math.log(i) for i in range(1, k + 1))
    except (ValueError, OverflowError):
        return 1.0
    return min(1.0, depth * width * math.exp(log_p))


def compute(spec: CountMinSketchSpec) -> CountMinSketchMetrics:
    """给定参数，递推全部指标。"""
    # 自动计算 w 与 d
    w = compute_width(spec.epsilon) if spec.width == 0 else spec.width
    d = compute_depth(spec.delta) if spec.depth == 0 else spec.depth

    # 若指定了 w，反推实际 ε
    epsilon = spec.epsilon if spec.width == 0 else compute_relative_error_bound(w, spec.n)

    abs_err = compute_absolute_error(epsilon, spec.n)
    num_counters = compute_num_counters(w, d)
    mem_bits = compute_memory_bits(w, d, spec.counter_bits)
    mem_bytes = compute_memory_bytes(w, d, spec.counter_bits)
    c_max = compute_counter_max(spec.counter_bits)

    return CountMinSketchMetrics(
        n=spec.n,
        width=w,
        depth=d,
        epsilon=epsilon,
        delta=spec.delta,
        counter_bits=spec.counter_bits,
        absolute_error=abs_err,
        num_counters=num_counters,
        memory_bits=mem_bits,
        memory_bytes=mem_bytes,
        counter_max=c_max,
    )


def _selftest():
    """公式自洽性检查"""
    spec = CountMinSketchSpec(n=1_000_000, epsilon=0.01, delta=0.001)
    m = compute(spec)

    # w 与 d 应满足参数公式
    assert m.width == 272, f"w 应为 272，实际 {m.width}"
    assert m.depth == 7, f"d 应为 7，实际 {m.depth}"

    # ε 越小，w 越大
    w_loose = compute_width(0.1)
    w_tight = compute_width(0.001)
    assert w_tight > w_loose, "ε 越小，宽度应越大"

    # δ 越小，d 越大
    d_loose = compute_depth(0.1)
    d_tight = compute_depth(1e-6)
    assert d_tight > d_loose, "δ 越小，深度应越大"

    # 内存应随计数器位数线性增长
    m16 = compute(CountMinSketchSpec(epsilon=0.01, delta=0.001, counter_bits=16))
    m32 = compute(CountMinSketchSpec(epsilon=0.01, delta=0.001, counter_bits=32))
    assert m32.memory_bits == 2 * m16.memory_bits, "位数翻倍，内存应翻倍"

    # 绝对误差应等于 εN
    assert abs(m.absolute_error - 0.01 * m.n) < 1e-9, "绝对误差应为 εN"

    # 计数器最大值应为 2^bits - 1
    assert m.counter_max == 2 ** 32 - 1, "32 位计数器最大值应为 2^32-1"

    # 内存应为正且合理
    assert 0 < m.memory_bytes < 1024 * 1024, "内存在合理范围"

    print("selftest 全部通过")
    print(f"\nN={m.n:,}  ε={m.epsilon}  δ={m.delta}")
    print(f"w={m.width}  d={m.depth}  计数器={m.num_counters:,}  "
          f"绝对误差={m.absolute_error:,.0f}")
    print(f"内存={m.memory_bits/8/1024:.2f} KiB  "
          f"(每元素 {m.memory_bits/m.n:.2f} bits)")


if __name__ == "__main__":
    _selftest()
