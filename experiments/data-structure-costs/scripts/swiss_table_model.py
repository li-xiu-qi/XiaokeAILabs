# -*- coding: utf-8 -*-
"""
瑞士表（Swiss Table）理论性能模型

Google Abseil 的开放寻址哈希表实现，被 Rust hashbrown、Go 1.24+ 标准库 map 采用。

核心结构：槽位数组 + 独立的元数据数组（metadata，每槽 1 字节控制字）。
哈希值被切成两段：
  - h1（高位）：决定起始 group 的位置
  - h2（低位 7 位）：作为「签名」存进控制字，用于快速过滤

查找时不逐个比较 key，而是用 SIMD 一条指令同时比较 16 个控制字，
先定位候选槽位，再对少数候选做完整 key 比较。

核心公式：
- group 大小 G = 8（SSE2，16 字节 / 每控制字 1 字节）或 16（AVX2，32 字节）
- 负载因子上限 7/8 = 0.875
- 一次组内扫描固定比较 G 个槽，代价与 G 成正比、与 α 近似无关
- 组数 = ceil(容量 / G)，探测序列在组间推进

本模块实现这些公式，参数可配。
"""
from dataclasses import dataclass
import math


@dataclass
class SwissTableSpec:
    n: int = 1_000_000          # 元素数
    entry_size: int = 16        # 单条目字节数（key + value）
    target_load: float = 0.875  # 目标负载因子（上限 7/8）
    group_size: int = 8         # 组内槽位数（8=SSE2，16=AVX2）
    max_load: float = 0.875     # 扩容阈值


@dataclass
class SwissTableMetrics:
    n: int
    num_slots: int              # 槽位数（向上取整到 group 的整数倍）
    num_groups: int             # 组数
    group_size: int
    load_factor: float          # 实际负载因子
    storage_bytes: int          # 总存储（metadata + 槽位数组）
    metadata_bytes: int         # 元数据数组字节
    slot_array_bytes: int       # 槽位数组字节
    metadata_ratio: float       # 元数据占总存储比例
    avg_groups_probed: float    # 平均需要扫描的组数
    needs_resize: bool


def num_slots_for(n: int, target_load: float, group_size: int) -> int:
    """
    槽位数 = ceil(n / target_load)，再向上取整到 group 的整数倍。

    分组是硬约束：每组必须完整，否则最后一组不足 G 个槽，
    SIMD 一次比较 G 个控制字就会读到数组越界的位置。
    """
    raw = math.ceil(n / target_load)
    return math.ceil(raw / group_size) * group_size


def compute_storage(spec: SwissTableSpec, num_slots: int) -> tuple:
    """
    存储字节。(metadata 字节, 槽位字节)。

    metadata：每槽 1 字节控制字。控制字编码三态（空 / 墓碑 / 已占用）+ 7 位 h2 签名。
    槽位数组：连续存放条目，这是缓存友好的来源。
    """
    metadata = num_slots  # 每槽 1 字节
    slots = num_slots * spec.entry_size
    return metadata, slots


def avg_groups_probed(load_factor: float, group_size: int) -> float:
    """
    平均需要扫描的组数（**独立占用假设下的近似**）。

    探测在「找到一个含空槽的组」时停止。若各组占用相互独立，组内无空槽的
    概率 ≈ α^G（G 个槽全被占），故平均扫描组数 ≈ 1 / (1 - α^G)。

    α=0.875、G=8 时 α^8 ≈ 0.344，平均扫描约 1.52 组。

    **该假设在高负载下会低估。** 实测（verify_swiss_table.py）显示 α=0.5
    时吻合（1.00 vs 1.06），α=0.85 时实测 3.36 而模型给 1.36。原因是探测序列
    沿组线性推进，连续组之间占用相关（一组满了，相邻组往往也接近满），
    独立性假设不成立。α ≤ 0.7 时该近似可用，α > 0.8 时需用实测标定。
    """
    if load_factor <= 0.0:
        return 1.0
    if load_factor >= 1.0:
        return float("inf")
    p_full = load_factor ** group_size
    return 1.0 / (1.0 - p_full)


def compute(spec: SwissTableSpec) -> SwissTableMetrics:
    """给定参数，递推全部指标。"""
    slots = num_slots_for(spec.n, spec.target_load, spec.group_size)
    groups = slots // spec.group_size
    alpha = spec.n / slots
    meta, slot_bytes = compute_storage(spec, slots)
    total = meta + slot_bytes
    return SwissTableMetrics(
        n=spec.n,
        num_slots=slots,
        num_groups=groups,
        group_size=spec.group_size,
        load_factor=alpha,
        storage_bytes=total,
        metadata_bytes=meta,
        slot_array_bytes=slot_bytes,
        metadata_ratio=meta / total,
        avg_groups_probed=avg_groups_probed(alpha, spec.group_size),
        needs_resize=alpha > spec.max_load,
    )


def _selftest():
    """公式自洽性检查"""
    # 负载因子上限 7/8：α=0.875 时不应触发扩容
    m = compute(SwissTableSpec(n=1_000_000, target_load=0.875))
    assert m.load_factor <= 0.875 + 1e-9, \
        f"目标 0.875 时实际负载不应超过上限, 实际 {m.load_factor}"
    assert not m.needs_resize

    # 超过上限应触发扩容
    m_over = compute(SwissTableSpec(n=1000, target_load=0.9))
    assert m_over.needs_resize, "目标 0.9 超过 7/8 应触发扩容"

    # 槽位数必须是 group 的整数倍
    for gs in [8, 16]:
        mm = compute(SwissTableSpec(n=1000, group_size=gs))
        assert mm.num_slots % gs == 0, f"槽位数必须是 {gs} 的整数倍"

    # 组数 = 槽位 / 组大小
    assert m.num_groups == m.num_slots // m.group_size

    # 存储 = metadata + 槽位数组
    spec = SwissTableSpec(n=1000, entry_size=16)
    mm = compute(spec)
    assert mm.storage_bytes == mm.metadata_bytes + mm.slot_array_bytes
    assert mm.metadata_bytes == mm.num_slots  # 每槽 1 字节
    assert mm.slot_array_bytes == mm.num_slots * 16

    # 负载越高，平均扫描组数越多
    m_low = compute(SwissTableSpec(n=1_000_000, target_load=0.5))
    m_high = compute(SwissTableSpec(n=1_000_000, target_load=0.875))
    assert m_high.avg_groups_probed > m_low.avg_groups_probed

    # α=0.875、G=8 时平均扫描组数应小于 2
    assert 1.0 < m.avg_groups_probed < 2.0, \
        f"α=0.875 平均扫描应在 1-2 组之间, 实际 {m.avg_groups_probed}"

    # 元数据占比：条目 16 字节时约为 1/17
    spec16 = SwissTableSpec(n=1_000_000, entry_size=16)
    m16 = compute(spec16)
    assert abs(m16.metadata_ratio - 1 / 17) < 0.01

    print("selftest 全部通过")
    print(f"\nn={m.n:,}  槽位={m.num_slots:,}  组数={m.num_groups:,}  负载={m.load_factor:.4f}")
    print(f"元数据={m.metadata_bytes/1024/1024:.2f} MiB  槽位数组={m.slot_array_bytes/1024/1024:.2f} MiB")
    print(f"平均扫描组数={m.avg_groups_probed:.2f}")
    print(f"\n不同负载与组大小下的平均扫描组数：")
    print(f"{'α':>6} {'G=8':>8} {'G=16':>8} {'元数据占比(16B)':>16}")
    for lf in [0.5, 0.7, 0.85]:
        a = compute(SwissTableSpec(n=1_000_000, target_load=lf, group_size=8))
        b = compute(SwissTableSpec(n=1_000_000, target_load=lf, group_size=16))
        print(f"{a.load_factor:>6.3f} {a.avg_groups_probed:>8.2f} "
              f"{b.avg_groups_probed:>8.2f} {a.metadata_ratio:>16.2%}")


if __name__ == "__main__":
    _selftest()
