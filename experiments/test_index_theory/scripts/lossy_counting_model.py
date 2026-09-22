# -*- coding: utf-8 -*-
"""
Lossy Counting 理论性能模型

Lossy Counting（Manku & Motwani 2002）是确定性的流式 Heavy Hitters 算法。
把流切成一连串固定大小的桶，桶结束时按"年龄"淘汰低频项。

核心结构：
- 桶宽 w = ceil(1/epsilon)，流被切成 N/w 个桶
- 一张 (元素, 频次, 最大可能误差) 的表
- 更新(i)：表中有 i 则 +1，否则插入 (i, 1, 当前桶号 - 1)
- 桶边界处理：删掉所有 频次 + delta <= 当前桶号 的项（delta 为该项的最大误差）

核心指标：
1. 确定性保证：频率 > epsilon x N 的元素一定留在表里
2. 误差界：估计值 <= 真实频率 <= 估计值 + epsilon x N（双侧有界）
3. 空间 O((1/epsilon) x log(epsilon x N))，比 O(1/epsilon) 多一个对数因子
4. 无概率参数，同一输入流结果唯一

关键性质：
- delta 单调不降：元素每活过一个桶，delta 增加 1，故 频次 + delta 也单调不降
- 桶边界才做删除，桶内只增不减，所以两次淘汰之间表只增长
- 与 Space-Saving 的区别在时间结构：Space-Saving 用容量上限驱动淘汰，
  Lossy Counting 用桶边界驱动淘汰，后者更容易做成"定期批量清扫"
- 空间多出的 log(epsilon x N) 因子来自"任意时刻被追踪元素个数"的界，
  即频次 > (当前桶号 - 1) 的元素数，上界为 1/epsilon x log(epsilon x N)

本模块实现这些公式，参数可配。
"""
from dataclasses import dataclass
import math


@dataclass
class LossyCountingSpec:
    """Lossy Counting 参数"""
    n: int = 1_000_000           # 流长度（已处理元素数 N）
    epsilon: float = 0.01        # 相对误差 ε（误差界 = εN）
    bucket_size: int = 0         # 桶宽 w（0 表示按 epsilon 自动计算）
    universe_bits: int = 32      # 元素 ID 所需比特数（log2 字典大小 U）


@dataclass
class LossyCountingMetrics:
    n: int                       # 流长度 N
    epsilon: float               # 实际相对误差 1/w
    bucket_size: int             # 桶宽 w
    num_buckets: int             # 桶总数 ceil(N/w)
    error_bound: int             # 绝对误差界 εN
    guarantee_threshold: int     # 保证被追踪的频率阈值 floor(εN)+1
    counter_bits: int            # 单计数器所需比特数（计数上限 N）
    entry_bits: int              # 单个表项比特数（ID + 频次 + delta）
    space_bits: int              # 总比特数（按理论界估算）
    space_bytes: int             # 总字节数
    expected_entries: float      # 理论界下的期望表项数 (1/ε)×log(εN)


def compute_bucket_size(epsilon: float) -> int:
    """桶宽。w = ceil(1/epsilon)。"""
    return int(math.ceil(1.0 / epsilon))


def compute_epsilon(bucket_size: int) -> float:
    """由桶宽反推实际误差。epsilon = 1/w。"""
    return 1.0 / bucket_size


def compute_num_buckets(n: int, bucket_size: int) -> int:
    """桶总数。ceil(N/w)。"""
    return int(math.ceil(n / bucket_size))


def compute_error_bound(epsilon: float, n: int) -> int:
    """绝对误差界。误差 <= εN。"""
    return int(epsilon * n)


def compute_guarantee_threshold(epsilon: float, n: int) -> int:
    """保证被追踪的频率阈值。频率 > εN 即 floor(εN)+1。"""
    return int(epsilon * n) + 1


def compute_counter_bits(n: int) -> int:
    """单计数器所需比特数。计数上限为 N，需 ceil(log2(N+1))。"""
    if n <= 1:
        return 1
    return int(math.ceil(math.log2(n + 1)))


def compute_expected_entries(epsilon: float, n: int) -> float:
    """理论界下的期望表项数。(1/ε) × log2(εN)。"""
    if epsilon * n <= 1:
        return 0.0
    return (1.0 / epsilon) * math.log2(epsilon * n)


def compute_entry_bits(n: int, universe_bits: int) -> int:
    """单个表项比特数。元素 ID + 频次 + delta 两个计数位宽。"""
    return universe_bits + 2 * compute_counter_bits(n)


def compute_space_bits(epsilon: float, n: int, universe_bits: int) -> int:
    """总比特数（按理论界估算）。期望表项数 × entry_bits。"""
    return int(math.ceil(compute_expected_entries(epsilon, n) *
                         compute_entry_bits(n, universe_bits)))


def compute_space_bytes(epsilon: float, n: int, universe_bits: int) -> int:
    """总字节数。向上取整到整字节。"""
    return math.ceil(compute_space_bits(epsilon, n, universe_bits) / 8)


def compute(spec: LossyCountingSpec) -> LossyCountingMetrics:
    """给定参数，递推全部指标。"""
    w = compute_bucket_size(spec.epsilon) if spec.bucket_size == 0 else spec.bucket_size
    epsilon = spec.epsilon if spec.bucket_size == 0 else compute_epsilon(w)

    err = compute_error_bound(epsilon, spec.n)
    thr = compute_guarantee_threshold(epsilon, spec.n)
    nb = compute_num_buckets(spec.n, w)
    c_bits = compute_counter_bits(spec.n)
    e_bits = compute_entry_bits(spec.n, spec.universe_bits)
    s_bits = compute_space_bits(epsilon, spec.n, spec.universe_bits)
    s_bytes = compute_space_bytes(epsilon, spec.n, spec.universe_bits)
    ee = compute_expected_entries(epsilon, spec.n)

    return LossyCountingMetrics(
        n=spec.n,
        epsilon=epsilon,
        bucket_size=w,
        num_buckets=nb,
        error_bound=err,
        guarantee_threshold=thr,
        counter_bits=c_bits,
        entry_bits=e_bits,
        space_bits=s_bits,
        space_bytes=s_bytes,
        expected_entries=ee,
    )


def _selftest():
    """公式自洽性检查"""
    spec = LossyCountingSpec(n=1_000_000, epsilon=0.01)
    m = compute(spec)

    # 桶宽 = ceil(1/epsilon)
    assert m.bucket_size == 100, f"ε=0.01 时 w 应为 100，实际 {m.bucket_size}"
    assert abs(m.epsilon - 0.01) < 1e-12, "epsilon 应为 1/w"

    # 误差界 = εN
    assert m.error_bound == 10_000, f"误差界应为 10000，实际 {m.error_bound}"

    # 保证阈值 = floor(εN) + 1
    assert m.guarantee_threshold == 10_001, \
        f"保证阈值应为 10001，实际 {m.guarantee_threshold}"

    # 桶总数 = ceil(N/w)
    assert m.num_buckets == 10_000, f"桶总数应为 10000，实际 {m.num_buckets}"

    # ε 越小，桶宽越大（固定 N 下桶数 = εN，反而越少）
    w_loose = compute_bucket_size(0.1)
    w_tight = compute_bucket_size(0.001)
    assert w_tight > w_loose, "ε 越小，桶宽应越大"
    assert compute_num_buckets(1_000_000, w_tight) < \
        compute_num_buckets(1_000_000, w_loose), "ε 越小，桶数应越少"

    # 桶数应恰好等于 ε×N 向上取整
    assert m.num_buckets == int(math.ceil(m.epsilon * m.n)), \
        "桶数应为 ceil(εN)"

    # ε 越小，期望表项数越多（(1/ε)×log(εN) 单调降）
    m_eps_loose = compute(LossyCountingSpec(n=1_000_000, epsilon=0.1))
    m_eps_tight = compute(LossyCountingSpec(n=1_000_000, epsilon=0.001))
    assert m_eps_tight.bucket_size > m_eps_loose.bucket_size, "ε 越小桶宽越大"
    assert m_eps_tight.expected_entries > m_eps_loose.expected_entries, \
        "ε 越小，期望表项数应越多"
    assert m_eps_tight.space_bytes > m_eps_loose.space_bytes, \
        "ε 越小，理论空间应越大"

    # 表项位宽 = ID + 频次 + delta
    assert m.entry_bits == 32 + 2 * m.counter_bits, \
        "表项应为 ID + 频次 + delta"

    # 期望表项数应满足 (1/ε) × log(εN) 量级
    expected = (1.0 / 0.01) * math.log2(0.01 * 1_000_000)
    assert abs(m.expected_entries - expected) < 1e-6, \
        f"期望表项数应为 {expected}，实际 {m.expected_entries}"

    # 误差界必须小于 n
    assert 0 < m.error_bound < m.n, "误差界应在 (0, n) 内"

    # 与 Space-Saving 同 ε 同 N 下误差界一致
    from space_saving_model import compute as ss_compute, SpaceSavingSpec
    ss = ss_compute(SpaceSavingSpec(n=1_000_000, epsilon=0.01))
    assert ss.error_bound == m.error_bound, "同参数下两算法误差界应一致"

    print("selftest 全部通过")
    print(f"\nN={m.n:,}  ε={m.epsilon}  w={m.bucket_size}  "
          f"桶数={m.num_buckets:,}")
    print(f"绝对误差界={m.error_bound:,}  保证阈值={m.guarantee_threshold:,}  "
          f"计数器位宽={m.counter_bits} bit")
    print(f"表项={m.entry_bits} bit  理论空间={m.space_bits/8/1024:.2f} KiB  "
          f"(期望表项 {m.expected_entries:.0f})")


if __name__ == "__main__":
    _selftest()
