# -*- coding: utf-8 -*-
"""
哈希索引理论性能模型（核心计算器）

纯闭式解。哈希索引的核心：O(1) 点查，但要处理哈希冲突。
两种冲突解决：
- 拉链法（chaining）：每个桶一个链表，冲突追加到链表
- 开放寻址（open addressing）：冲突后探测下一个空位（线性/二次/双重哈希）

负载因子 α = n / 桶数，是关键参数。
- 拉链法：平均探测次数 ≈ 1 + α/2（查链表平均长度）
- 开放寻址（线性探测）：平均探测次数 ≈ (1 + 1/(1-α)^2) / 2
- 开放寻址（双重哈希）：平均探测次数 ≈ -ln(1-α) / α

存储：桶数 × 桶大小（拉链法）或 桶数 × 条目大小（开放寻址）
扩容阈值：α > 0.75 时性能急剧恶化，需扩容（重新哈希）
"""
from dataclasses import dataclass
import math


@dataclass
class HashSpec:
    n: int                       # 条目数
    bucket_size: int = 8         # 桶大小（字节），存指针或条目
    entry_size: int = 100        # 条目大小（字节）
    load_factor: float = 0.75    # 负载因子 α = n / 桶数
    method: str = "chaining"     # "chaining" 或 "open_addressing"
    probe: str = "linear"        # 开放寻址的探测方式："linear" 或 "double"


@dataclass
class HashMetrics:
    n: int
    num_buckets: int             # 桶数
    load_factor: float           # 实际负载因子
    avg_probes: float            # 平均探测次数
    storage_bytes: int           # 存储字节
    needs_resize: bool           # 是否需要扩容


def num_buckets(n: int, load_factor: float) -> int:
    """桶数 = ceil(n / load_factor)，向上取整到 2 的幂（哈希表常用）。"""
    return math.ceil(n / load_factor)


def avg_probes_chaining(load_factor: float) -> float:
    """
    拉链法平均探测次数。
    查成功 ≈ 1 + α/2 - α/(2n)（链表中找，平均走一半）
    查失败 ≈ 1 + α（走到链表尾）
    综合 ≈ 1 + α/2
    """
    return 1 + load_factor / 2


def avg_probes_linear(load_factor: float) -> float:
    """
    开放寻址（线性探测）平均探测次数。
    查成功 ≈ (1 + 1/(1-α)) / 2
    查失败 ≈ (1 + 1/(1-α)^2) / 2
    综合 ≈ (1 + 1/(1-α)^2) / 2（失败代价高）
    """
    if load_factor >= 1.0:
        return float("inf")
    return (1 + 1 / (1 - load_factor) ** 2) / 2


def avg_probes_double(load_factor: float) -> float:
    """
    开放寻址（双重哈希）平均探测次数。
    双重哈希避免线性探测的聚集问题。
    查成功 ≈ -ln(1-α) / α
    查失败 ≈ 1 / (1-α)
    """
    if load_factor >= 1.0:
        return float("inf")
    return -math.log(1 - load_factor) / load_factor


def avg_probes(spec: HashSpec) -> float:
    """根据方法计算平均探测次数。"""
    if spec.method == "chaining":
        return avg_probes_chaining(spec.load_factor)
    elif spec.probe == "linear":
        return avg_probes_linear(spec.load_factor)
    else:
        return avg_probes_double(spec.load_factor)


def storage_bytes(spec: HashSpec) -> int:
    """
    存储字节。
    拉链法：桶数组（num_buckets × bucket_size）+ 条目（n × entry_size）
    开放寻址：桶数组（num_buckets × entry_size，条目存在桶里）
    """
    nb = num_buckets(spec.n, spec.load_factor)
    if spec.method == "chaining":
        return nb * spec.bucket_size + spec.n * spec.entry_size
    else:
        return nb * spec.entry_size


def compute(spec: HashSpec) -> HashMetrics:
    """给定参数，递推全部指标。"""
    nb = num_buckets(spec.n, spec.load_factor)
    probes = avg_probes(spec)
    storage = storage_bytes(spec)
    # 扩容阈值：α > 0.75 时开放寻址性能恶化，拉链法可容忍到 1.0
    threshold = 0.75 if spec.method == "open_addressing" else 1.0
    needs_resize = spec.load_factor > threshold
    return HashMetrics(
        n=spec.n,
        num_buckets=nb,
        load_factor=spec.load_factor,
        avg_probes=probes,
        storage_bytes=storage,
        needs_resize=needs_resize,
    )


def _selftest():
    """公式自洽性检查"""
    # 拉链法：负载因子越高探测越多
    spec_low = HashSpec(n=1000, load_factor=0.5, method="chaining")
    spec_high = HashSpec(n=1000, load_factor=0.9, method="chaining")
    assert compute(spec_high).avg_probes > compute(spec_low).avg_probes

    # 拉链法探测次数 ≈ 1 + α/2
    m = compute(HashSpec(n=1000, load_factor=0.8, method="chaining"))
    assert abs(m.avg_probes - 1.4) < 0.01, f"拉链法 α=0.8 应探测 1.4 次, 实际 {m.avg_probes}"

    # 开放寻址线性探测：α=0.5 时应探测约 2.5 次
    m_oa = compute(HashSpec(n=1000, load_factor=0.5, method="open_addressing", probe="linear"))
    assert 2.0 < m_oa.avg_probes < 3.0, f"线性探测 α=0.5 应探测约 2.5 次, 实际 {m_oa.avg_probes}"

    # 双重哈希比线性探测少（无聚集）
    m_linear = compute(HashSpec(n=1000, load_factor=0.8, method="open_addressing", probe="linear"))
    m_double = compute(HashSpec(n=1000, load_factor=0.8, method="open_addressing", probe="double"))
    assert m_double.avg_probes < m_linear.avg_probes, "双重哈希应比线性探测少"

    # 拉链法存储 = 桶 + 条目
    m_chain = compute(HashSpec(n=1000, bucket_size=8, entry_size=100, load_factor=0.75, method="chaining"))
    expected = m_chain.num_buckets * 8 + 1000 * 100
    assert m_chain.storage_bytes == expected

    # 开放寻址 α 接近 1 时探测次数爆炸
    m_critical = compute(HashSpec(n=1000, load_factor=0.99, method="open_addressing", probe="linear"))
    assert m_critical.avg_probes > 50, "α=0.99 时线性探测应极慢"

    print("selftest 全部通过")
    # 演示
    print("\n拉链法 vs 开放寻址（线性/双重）探测次数对比：")
    print(f"{'α':>6} {'拉链':>8} {'线性':>8} {'双重':>8}")
    for lf in [0.5, 0.75, 0.9, 0.95]:
        chain = compute(HashSpec(n=1000, load_factor=lf, method="chaining")).avg_probes
        linear = compute(HashSpec(n=1000, load_factor=lf, method="open_addressing", probe="linear")).avg_probes
        double = compute(HashSpec(n=1000, load_factor=lf, method="open_addressing", probe="double")).avg_probes
        print(f"{lf:>6.2f} {chain:>8.2f} {linear:>8.2f} {double:>8.2f}")


if __name__ == "__main__":
    _selftest()
