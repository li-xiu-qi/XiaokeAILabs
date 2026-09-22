# -*- coding: utf-8 -*-
"""
Misra-Gries 理论性能模型

Misra-Gries（Misra & Gries 1982）是最早的确定性流式 Heavy Hitters 算法。
用 k 个计数器维护一张候选表，任何出现次数超过 n/k 的元素都保证留在表里。

核心结构：
- 一张最多容纳 k 个 (元素, 计数) 的表 T
- 更新(i)：若 i 在 T 中则对应计数 +1；否则若 T 未满则插入并置 1；
  否则 T 中所有计数 -1，并把归零的元素删掉
- 查询(i)：返回 T 中 i 的计数；若 i 不在 T 中则返回 0

核心指标：
1. 确定性保证：频率 > n/k 的元素一定在最终的表 T 中
2. 误差界：估计值 <= 真实频率 + n/k（单侧，只低估不高估）
3. 空间：O(k)，k = ceil(1/epsilon) 时误差界为 epsilon*n
4. 无概率参数，同一输入流结果唯一

关键性质：
- 每条流的更新都让计数器总和减少（插入新元素时 T 满则全体 -1），
  所以 sum(T) 单调不增且 <= n，这是误差界推导的抓手
- 被淘汰的元素是真的"计入过"：每次全体减 1 相当于 k 个元素各付 1 的账，
  所以低估的部分上界就是减去的总量，即 n - sum(T)
- 相比 Space-Saving 与 Lossy Counting，Misra-Gries 实现最简单，
  但每次淘汰要遍历 k 个计数器，最坏 O(k)

本模块实现这些公式，参数可配。
"""
from dataclasses import dataclass
import math


@dataclass
class MisraGriesSpec:
    """Misra-Gries 参数"""
    n: int = 1_000_000           # 流长度（总更新次数 N）
    k: int = 0                   # 计数器数（0 表示按 epsilon 自动计算）
    epsilon: float = 0.01        # 相对误差 ε（误差界 = εN）
    universe_bits: int = 32      # 元素 ID 所需比特数（log2 字典大小 U）


@dataclass
class MisraGriesMetrics:
    n: int                       # 流长度 N
    k: int                       # 计数器数
    epsilon: float               # 实际相对误差 1/k
    error_bound: int             # 绝对误差界 n/k
    guarantee_threshold: int     # 保证被追踪的频率阈值 floor(n/k)+1
    max_candidates: int          # 表的最大容量 k
    counter_bits: int            # 单计数器所需比特数（计数到 n）
    entry_bits: int              # 单个表项比特数（ID + 计数）
    space_bits: int              # 总比特数 k × entry_bits
    space_bytes: int             # 总字节数
    entries_per_kb: float        # 每 KiB 可容纳的表项数


def compute_k(epsilon: float) -> int:
    """计数器数。k = ceil(1/epsilon)。"""
    return int(math.ceil(1.0 / epsilon))


def compute_epsilon(k: int) -> float:
    """由 k 反推实际误差。epsilon = 1/k。"""
    return 1.0 / k


def compute_error_bound(n: int, k: int) -> int:
    """绝对误差界。误差 <= n/k。"""
    return n // k


def compute_guarantee_threshold(n: int, k: int) -> int:
    """保证被追踪的频率阈值。频率 > n/k 即 floor(n/k)+1。"""
    return n // k + 1


def compute_counter_bits(n: int) -> int:
    """单计数器所需比特数。计数上限为 n，故需 ceil(log2(n+1))。"""
    if n <= 1:
        return 1
    return int(math.ceil(math.log2(n + 1)))


def compute_entry_bits(n: int, universe_bits: int) -> int:
    """单个表项比特数。元素 ID 位宽 + 计数器位宽。"""
    return universe_bits + compute_counter_bits(n)


def compute_space_bits(k: int, n: int, universe_bits: int) -> int:
    """总比特数。k × (universe_bits + counter_bits)。"""
    return k * compute_entry_bits(n, universe_bits)


def compute_space_bytes(k: int, n: int, universe_bits: int) -> int:
    """总字节数。向上取整到整字节。"""
    return math.ceil(compute_space_bits(k, n, universe_bits) / 8)


def compute_entries_per_kb(k: int, n: int, universe_bits: int) -> float:
    """每 KiB 可容纳的表项数（用于评估空间效率）。"""
    byts = compute_space_bytes(k, n, universe_bits)
    if byts == 0:
        return 0.0
    return k / (byts / 1024)


def compute_max_tracked_error(n: int, k: int) -> int:
    """最坏情况下的累计低估量上界。n - sum(T) <= n/k 对每个被追踪元素成立，
    这里返回全体计数器被减去的总量上界 n/k × 淘汰次数，简化为 n/k。"""
    return n // k


def compute(spec: MisraGriesSpec) -> MisraGriesMetrics:
    """给定参数，递推全部指标。"""
    k = compute_k(spec.epsilon) if spec.k == 0 else spec.k
    epsilon = spec.epsilon if spec.k == 0 else compute_epsilon(k)

    err = compute_error_bound(spec.n, k)
    thr = compute_guarantee_threshold(spec.n, k)
    c_bits = compute_counter_bits(spec.n)
    e_bits = compute_entry_bits(spec.n, spec.universe_bits)
    s_bits = compute_space_bits(k, spec.n, spec.universe_bits)
    s_bytes = compute_space_bytes(k, spec.n, spec.universe_bits)
    epk = compute_entries_per_kb(k, spec.n, spec.universe_bits)

    return MisraGriesMetrics(
        n=spec.n,
        k=k,
        epsilon=epsilon,
        error_bound=err,
        guarantee_threshold=thr,
        max_candidates=k,
        counter_bits=c_bits,
        entry_bits=e_bits,
        space_bits=s_bits,
        space_bytes=s_bytes,
        entries_per_kb=epk,
    )


def _selftest():
    """公式自洽性检查"""
    spec = MisraGriesSpec(n=1_000_000, epsilon=0.01)
    m = compute(spec)

    # k 与 epsilon 互为倒数（向上取整）
    assert m.k == 100, f"ε=0.01 时 k 应为 100，实际 {m.k}"
    assert abs(m.epsilon - 0.01) < 1e-12, "epsilon 应为 1/k"

    # 误差界 = n/k
    assert m.error_bound == 10_000, f"误差界应为 10000，实际 {m.error_bound}"

    # 保证阈值 = floor(n/k) + 1，即 > n/k 的最小整数
    assert m.guarantee_threshold == 10_001, \
        f"保证阈值应为 10001，实际 {m.guarantee_threshold}"

    # epsilon 越小，k 越大
    k_loose = compute_k(0.1)
    k_tight = compute_k(0.001)
    assert k_tight > k_loose, "ε 越小，k 应越大"

    # 内存应随 k 线性增长
    m50 = compute(MisraGriesSpec(n=1_000_000, k=50))
    m100 = compute(MisraGriesSpec(n=1_000_000, k=100))
    assert m100.space_bits == 2 * m50.space_bits, "k 翻倍，内存应翻倍"

    # 计数器位宽应足够表示 n
    assert m.counter_bits >= math.ceil(math.log2(m.n + 1)) - 1, \
        "计数器位宽应能表示 n"

    # 32 位 ID + 20 位计数 = 52 位/表项
    assert m.entry_bits == 32 + m.counter_bits, "表项位宽应为 ID + 计数"

    # 内存应为正且合理（k=100 时应在 KiB 量级）
    assert 0 < m.space_bytes < 1024 * 1024, "内存在合理范围"

    # 误差界必须小于 n（否则保证无意义）
    assert 0 < m.error_bound < m.n, "误差界应在 (0, n) 内"

    print("selftest 全部通过")
    print(f"\nN={m.n:,}  ε={m.epsilon}  k={m.k}")
    print(f"绝对误差界={m.error_bound:,}  保证阈值={m.guarantee_threshold:,}  "
          f"计数器位宽={m.counter_bits} bit")
    print(f"表项={m.entry_bits} bit  总空间={m.space_bits/8/1024:.2f} KiB  "
          f"({m.entries_per_kb:.1f} 表项/KiB)")


if __name__ == "__main__":
    _selftest()
