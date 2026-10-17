# -*- coding: utf-8 -*-
"""
罗宾汉哈希（Robin Hood Hashing）理论性能模型

开放寻址 + 线性探测的一个变体。每个槽记录元素相对「理想位置」的探测距离 PSL
（Probe Sequence Length，即当前位置 - home(key)）。

核心机制「劫富济贫」：插入时若当前位置已被占，比较双方的 PSL。
- 新元素 PSL 更大（更穷）→ 抢占该位置，把更富（PSL 更小）的旧元素往后挤
- 新元素 PSL 更小（更富）→ 不抢占，继续向后探测

这个「从富者处取回」的策略使所有元素的 PSL 分布被压平，方差大幅缩小。
查找时可以提前终止：当前位置元素的 PSL 比自己还小，说明后面不可能有自己。

核心公式（均匀哈希假设下的理想值，Robin Hood 逼近它）：
- 插入 / 不成功查找：E[probes] = 1 / (1 - α)
- 成功查找：E[probes] = (1/α) · ln(1 / (1 - α))   （对插入时的 α 从 0 积分到 α）
- 普通线性探测成功查找：E[probes] = (1 + 1/(1-α)) / 2   ← 对照组
- 普通线性探测不成功查找：E[probes] = (1 + 1/(1-α)²) / 2

注意口径差异：α=0.9 时成功查找理想值 2.56，而普通线性探测成功查找均值 5.50。
两者量的是同一件事（成功查找），前者是均匀哈希的理论下限，Robin Hood 让
实测分布逼近它，但实测均值仍略高于理想值。罗宾汉真正的收益在 PSL 方差与
尾部（p99 / 最坏），而非均值本身——见 verify_robin_hood.py 的实测。

参考：Celis P, Larson P, Munro J. "Robin Hood Hashing", FOCS 1985；
      Viola A. "Distributional analysis of Robin Hood linear probing
      hashing with buckets", Algorithmica 2005。
"""
from dataclasses import dataclass
import math


@dataclass
class RobinHoodSpec:
    n: int = 1_000_000          # 元素数
    entry_size: int = 16        # 单条目字节数（key + value）
    load_factor: float = 0.9    # 目标负载因子 α = n / 容量
    max_load: float = 0.93      # 扩容阈值（超过则重建表）
    simd_width: int = 16        # SIMD 并行比较的槽位数（0 表示不做分组扫描）


@dataclass
class RobinHoodMetrics:
    n: int
    capacity: int               # 槽位数
    load_factor: float          # 实际负载因子 n / capacity
    avg_probe_success: float    # 成功查找期望探测（理想值）
    avg_probe_unsuccess: float  # 插入 / 不成功查找期望探测
    linear_success: float       # 普通线性探测成功查找期望探测（同口径对照）
    linear_unsuccess: float     # 普通线性探测不成功查找期望探测（对照）
    probe_reduction: float      # 相对线性探测的成功查找均值降幅（倍数）
    psl_reference_std: float    # 未均衡参照系下 PSL 标准差（几何分布）
    storage_bytes: int          # 槽位存储字节（条目 + PSL 字段）
    needs_resize: bool


def capacity_for(n: int, load_factor: float) -> int:
    """槽位数 = ceil(n / α)。"""
    return math.ceil(n / load_factor)


def avg_probe_success(load_factor: float) -> float:
    """
    成功查找的期望探测次数（均匀哈希理想值，Robin Hood 逼近它）。

    插入成本取决于插入时刻的 α（从 0 增长到当前 α），故对 α 积分：
        (1/α) · ∫₀^α dx/(1-x) = (1/α) · ln(1/(1-α))
    """
    if load_factor <= 0.0:
        return 1.0
    if load_factor >= 1.0:
        return float("inf")
    return math.log(1.0 / (1.0 - load_factor)) / load_factor


def avg_probe_unsuccess(load_factor: float) -> float:
    """插入 / 不成功查找的期望探测次数。每步遇到空槽的概率 (1-α)，几何分布均值。"""
    if load_factor >= 1.0:
        return float("inf")
    return 1.0 / (1.0 - load_factor)


def linear_probe_success(load_factor: float) -> float:
    """
    普通线性探测（无再平衡）的**成功查找**期望探测。

    含聚集惩罚：Knuth 给出的成功查找公式为 (1 + 1/(1-α)) / 2。
    α=0.9 时 5.50 次，是罗宾汉同口径理想值 2.56 的两倍多。
    注意不要与不成功查找公式 (1 + 1/(1-α)²) / 2（α=0.9 时 50.5）混用。
    """
    if load_factor >= 1.0:
        return float("inf")
    return (1.0 + 1.0 / (1.0 - load_factor)) / 2.0


def linear_probe_unsuccess(load_factor: float) -> float:
    """普通线性探测的不成功查找 / 插入期望探测。(1 + 1/(1-α)²) / 2。"""
    if load_factor >= 1.0:
        return float("inf")
    return (1.0 + 1.0 / (1.0 - load_factor) ** 2) / 2.0


def psl_reference_std(load_factor: float) -> float:
    """
    参照系：若 PSL 服从成功概率 (1-α) 的几何分布，标准差 = sqrt(α)/(1-α)。

    这不是声称 Robin Hood 满足该分布，而是给出「探测距离完全未均衡」时的
    离散度量级。Robin Hood 把方差压到该值以下，压平的幅度由实测给出。
    """
    if load_factor >= 1.0:
        return float("inf")
    return math.sqrt(load_factor) / (1.0 - load_factor)


def storage_bytes(spec: RobinHoodSpec, capacity: int) -> int:
    """
    存储字节。开放寻址条目连续存放，每槽额外存 PSL 字段。

    PSL 上界为 O(log n) 量级，工程实现常打包进 4-8 位，这里按 4 字节计（保守）。
    SIMD 分组时额外加 metadata 数组（每槽 1 字节控制字），否则不额外开销。
    """
    psl_field = 4
    base = capacity * (spec.entry_size + psl_field)
    if spec.simd_width > 0:
        base += capacity  # 每槽 1 字节控制字
    return base


def compute(spec: RobinHoodSpec) -> RobinHoodMetrics:
    """给定参数，递推全部指标。"""
    cap = capacity_for(spec.n, spec.load_factor)
    actual_alpha = spec.n / cap
    succ = avg_probe_success(actual_alpha)
    unsucc = avg_probe_unsuccess(actual_alpha)
    lin_s = linear_probe_success(actual_alpha)
    lin_u = linear_probe_unsuccess(actual_alpha)
    return RobinHoodMetrics(
        n=spec.n,
        capacity=cap,
        load_factor=actual_alpha,
        avg_probe_success=succ,
        avg_probe_unsuccess=unsucc,
        linear_success=lin_s,
        linear_unsuccess=lin_u,
        probe_reduction=lin_s / succ,
        psl_reference_std=psl_reference_std(actual_alpha),
        storage_bytes=storage_bytes(spec, cap),
        needs_resize=actual_alpha > spec.max_load,
    )


def _selftest():
    """公式自洽性检查"""
    # 不成功查找 = 1/(1-α)：α=0.9 时应为 10
    assert abs(avg_probe_unsuccess(0.9) - 10.0) < 1e-9, \
        f"α=0.9 不成功查找应为 10, 实际 {avg_probe_unsuccess(0.9)}"

    # 成功查找理想值：(1/0.9)·ln(10) ≈ 2.558
    assert abs(avg_probe_success(0.9) - math.log(10) / 0.9) < 1e-9

    # compute 的实际 α 因容量向上取整而略低于目标，须与闭式一致
    m = compute(RobinHoodSpec(n=1000, load_factor=0.9))
    assert abs(m.avg_probe_unsuccess - 1.0 / (1.0 - m.load_factor)) < 1e-9

    # 罗宾汉理想值显著低于普通线性探测成功查找（α=0.9 时 5.5 vs 2.56）
    assert abs(linear_probe_success(0.9) - 5.5) < 1e-9, \
        f"线性探测 α=0.9 成功查找应为 5.5, 实际 {linear_probe_success(0.9)}"
    assert linear_probe_success(0.9) / avg_probe_success(0.9) > 2.0
    assert abs(linear_probe_unsuccess(0.9) - 50.5) < 1e-9
    assert m.probe_reduction > 2.0, \
        f"α≈0.9 时罗宾汉应比线性探测快一倍以上, 实际比值 {m.probe_reduction}"

    # 负载越高，探测次数越多
    m_low = compute(RobinHoodSpec(n=1000, load_factor=0.5))
    m_high = compute(RobinHoodSpec(n=1000, load_factor=0.9))
    assert m_high.avg_probe_success > m_low.avg_probe_success
    assert m_high.avg_probe_unsuccess > m_low.avg_probe_unsuccess

    # 容量公式：ceil(n/α)
    m2 = compute(RobinHoodSpec(n=1000, load_factor=0.9))
    assert m2.capacity == math.ceil(1000 / 0.9), f"容量应为 1112, 实际 {m2.capacity}"

    # 存储 = cap × (entry + 4 字节 PSL)
    spec = RobinHoodSpec(n=1000, load_factor=0.9, entry_size=16, simd_width=0)
    m3 = compute(spec)
    assert m3.storage_bytes == m3.capacity * 20

    # SIMD 分组时每槽多 1 字节控制字
    spec_simd = RobinHoodSpec(n=1000, load_factor=0.9, entry_size=16, simd_width=16)
    m4 = compute(spec_simd)
    assert m4.storage_bytes == m4.capacity * 21

    # 扩容阈值：α > max_load 判定需重建
    assert compute(RobinHoodSpec(n=1000, load_factor=0.95, max_load=0.93)).needs_resize
    assert not compute(RobinHoodSpec(n=1000, load_factor=0.9, max_load=0.93)).needs_resize

    # α 趋近 1 时探测次数爆炸
    m_crit = compute(RobinHoodSpec(n=1000, load_factor=0.99))
    assert m_crit.avg_probe_unsuccess > 50

    print("selftest 全部通过")
    print(f"罗宾汉哈希 vs 普通线性探测（成功查找期望探测）：")
    print(f"{'α':>6} {'罗宾汉':>8} {'线性成功':>10} {'线性失败':>10} {'降幅':>8} {'未均衡PSLσ':>12}")
    for lf in [0.5, 0.7, 0.9, 0.95]:
        mm = compute(RobinHoodSpec(n=1_000_000, load_factor=lf))
        print(f"{mm.load_factor:>6.3f} {mm.avg_probe_success:>8.2f} "
              f"{mm.linear_success:>10.2f} {mm.linear_unsuccess:>10.2f} "
              f"{mm.probe_reduction:>7.2f}x "
              f"{mm.psl_reference_std:>12.2f}")


if __name__ == "__main__":
    _selftest()
