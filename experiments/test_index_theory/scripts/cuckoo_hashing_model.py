# -*- coding: utf-8 -*-
"""
布谷鸟哈希（Cuckoo Hashing）理论性能模型

布谷鸟哈希是一种开放寻址哈希表，用两个哈希函数 h1、h2 和两个表 T1、T2。
每个 key 只能放在 T1[h1(key)] 或 T2[h2(key)] 两个候选位置之一。
插入冲突时把占据位置的旧 key「踢出」，让旧 key 去它的另一个候选位置，递归下去。

核心指标：
1. 查找：O(1) 最坏（只查两个位置），对比拉链法 O(1+α)
2. 删除：O(1)（直接清空对应位置，不回填）
3. 负载因子阈值：理论最大 0.5（两表合计容量 / 元素数），实际工程用 0.4-0.45
4. 空间：2 × 表容量 × (key + value) 字节
5. 踢出链长度：高负载下递归踢出可能成环，需设最大踢出次数并触发 rehash

核心公式：
- 单表容量 M = ceil(n / (2 × 目标负载因子))
- 空间 = 2 × M × (key_size + value_size)
- 负载因子 α = n / (2 × M)
- 两表合计满负载阈值 α_max = 0.5（n = 2M 时）

本模块实现这些公式，参数可配。
"""
from dataclasses import dataclass
import math


@dataclass
class CuckooSpec:
    """布谷鸟哈希参数"""
    n: int = 1_000_000           # 预期元素数量
    key_size: int = 8            # 每个 key 的字节数
    value_size: int = 8          # 每个 value 的字节数
    target_load: float = 0.45    # 目标负载因子（工程实践 0.4-0.5）
    num_tables: int = 2          # 哈希表数量（标准为 2）
    num_hash: int = 2            # 每个 key 的候选位置数（标准为 2）


@dataclass
class CuckooMetrics:
    n: int                       # 元素数量
    table_capacity: int          # 单表容量（槽位数）
    total_slots: int             # 两表合计槽位数
    load_factor: float           # 实际负载因子 α = n / 合计槽位
    storage_bytes: int           # 存储字节（两表合计）
    max_load_theory: float       # 理论最大负载因子（阈值）
    insert_feasible: bool        # 目标负载是否低于阈值
    avg_lookup_positions: float  # 平均查找探测位置数（查两个位置）


def compute_table_capacity(n: int, target_load: float, num_tables: int = 2) -> int:
    """单表容量。合计槽位 = n / target_load，单表 = 合计 / num_tables，向上取整。

    说明：理论容量取精确值，使实际负载因子贴近目标。
    工程实现常把容量向上取整到 2 的幂（便于位运算索引），会带来额外空槽，
    实际负载因子因此低于目标，这部分开销在验证脚本中单独演示。
    """
    total_needed = math.ceil(n / target_load)
    per_table = math.ceil(total_needed / num_tables)
    return per_table


def compute_load_factor(n: int, total_slots: int) -> float:
    """负载因子。α = n / 合计槽位数。"""
    return n / total_slots if total_slots > 0 else 0.0


def compute_storage(table_capacity: int, num_tables: int,
                    key_size: int, value_size: int) -> int:
    """存储字节。2 × 表容量 × (key + value)。"""
    return num_tables * table_capacity * (key_size + value_size)


def compute(spec: CuckooSpec) -> CuckooMetrics:
    """给定参数，递推全部指标。"""
    cap = compute_table_capacity(spec.n, spec.target_load, spec.num_tables)
    total_slots = spec.num_tables * cap
    load = compute_load_factor(spec.n, total_slots)
    storage = compute_storage(cap, spec.num_tables, spec.key_size, spec.value_size)

    # 理论最大负载阈值：n 个元素占满一半槽位时 α=0.5
    # 工程上超过 0.5 必然出现无法插入（需 rehash），故阈值 0.5
    max_load_theory = 1.0 / spec.num_hash if spec.num_hash >= 2 else 0.5

    return CuckooMetrics(
        n=spec.n,
        table_capacity=cap,
        total_slots=total_slots,
        load_factor=load,
        storage_bytes=storage,
        max_load_theory=max_load_theory,
        insert_feasible=spec.target_load < max_load_theory,
        avg_lookup_positions=float(spec.num_hash),
    )


def _selftest():
    """公式自洽性检查"""
    # 目标负载 0.45 时插入应可行（< 阈值 0.5）
    m = compute(CuckooSpec(n=1_000_000, target_load=0.45))
    assert m.insert_feasible, "目标负载 0.45 应可行"
    assert m.load_factor <= 0.5 + 1e-9, "实际负载因子不应超过 0.5"

    # 目标负载 0.6 时超过阈值，不可行
    m_bad = compute(CuckooSpec(n=1_000_000, target_load=0.6))
    assert not m_bad.insert_feasible, "目标负载 0.6 超过阈值应不可行"

    # 负载因子越低，所需容量越大，存储越多
    m_high = compute(CuckooSpec(n=1_000_000, target_load=0.40))
    m_low = compute(CuckooSpec(n=1_000_000, target_load=0.49))
    assert m_high.table_capacity >= m_low.table_capacity, "低负载需要更大容量"
    assert m_high.storage_bytes >= m_low.storage_bytes, "低负载需要更多存储"

    # 理论容量使实际负载因子贴近目标（向上取整带来的误差 < 1e-6 相对）
    m2 = compute(CuckooSpec(n=1_000_000, target_load=0.45))
    assert abs(m2.load_factor - 0.45) < 0.01, \
        f"理论容量应使实际负载贴近目标 0.45, 实际 {m2.load_factor}"

    # 存储公式：2 × cap × (key + value)
    spec = CuckooSpec(n=1000, key_size=8, value_size=8, target_load=0.45)
    mm = compute(spec)
    assert mm.storage_bytes == 2 * mm.table_capacity * 16

    # 查找位置数 = 哈希函数数
    assert m.avg_lookup_positions == 2.0, "标准布谷鸟查找查 2 个位置"

    print("selftest 全部通过")
    # 演示
    print(f"\nn={m.n:,}  单表容量={m.table_capacity:,}  合计槽位={m.total_slots:,}")
    print(f"负载因子={m.load_factor:.3f}  存储={m.storage_bytes/1024/1024:.2f} MiB  "
          f"理论阈值={m.max_load_theory}")


if __name__ == "__main__":
    _selftest()
