# -*- coding: utf-8 -*-
"""
LSM-tree 理论性能模型（核心计算器）

纯闭式解，不依赖真实数据库。LSM-tree 的核心权衡：
- 写放大 WA：每写 1 字节用户数据，实际写盘多少字节（含 compaction 重写）
- 点查 IO：要查多少层（MemTable + 每层 SSTable，可能每层一个布隆过滤器）
- 空间放大 SA：磁盘上存的数据量 / 用户数据量（多层版本共存）
- 范围扫描 IO：要合并多少层的 SSTable

标准 tiering compaction（每层 size_ratio 倍于上层，满了触发合并）：
- L 层（不含 MemTable），size_ratio = T
- 写放大 WA ≈ T × L / 2（每层数据被重写约 L/2 次，每次写 T 倍）
- 更精确：WA = (T^L - 1) / (T - 1) × (T-1)/T ... 用教科书近似 T×L/2
- 点查最坏 IO = L（每层一个 SSTable）+ 1（MemTable）
- 点查期望 IO（布隆过滤器命中）= log2(每层条目数) 的查找代价，近似 1-2 次磁盘读/层

leveling compaction（每层 size_ratio 倍，但每层只保留一份有序数据）：
- 写放大 WA ≈ T × L / 2（和 tiering 类似，但每层内是全局有序归并）
- 空间放大 SA ≈ 1.1（leveling 每层只有一份，重叠少）
- tiering 空间放大 SA ≈ T/2（每层内多份 run 共存）

本模块实现两种 compaction 的公式，参数可配。
"""
from dataclasses import dataclass
import math


@dataclass
class LSMSpec:
    """LSM-tree 参数"""
    memtable_size: int = 64 * 1024 * 1024   # MemTable 大小（字节），默认 64 MiB
    sst_size: int = 2 * 1024 * 1024 * 1024  # 每层 SSTable 目标大小（字节），默认 2 GiB
    size_ratio: int = 10                     # 每层 / 上层 的倍数，典型 4-10
    num_levels: int = 6                      # 磁盘层数（不含 MemTable）
    entry_size: int = 100                    # 每条 KV 的字节数（key + value）
    compaction: str = "tiering"              # "tiering" 或 "leveling"


@dataclass
class LSMMetrics:
    n: int                       # 用户数据条目数
    total_data_bytes: int        # 用户数据总字节
    write_amplification: float   # 写放大（每写 1 字节用户数据，实际写盘字节）
    space_amplification: float   # 空间放大（磁盘 / 用户数据）
    point_lookup_worst_io: int   # 点查最坏 IO（每层一个 SSTable）
    point_lookup_expected_io: float  # 点查期望 IO（布隆过滤器过滤后）
    range_scan_io: int           # 范围扫描要合并的层数相关 IO
    num_sstables: int            # 磁盘上 SSTable 总数


def write_amplification_tiering(spec: LSMSpec) -> float:
    """
    Tiering compaction 写放大。
    每层满了后，选该层一个 run 与下层所有 run 归并。
    近似 WA ≈ size_ratio × num_levels / 2
    """
    return spec.size_ratio * spec.num_levels / 2


def write_amplification_leveling(spec: LSMSpec) -> float:
    """
    Leveling compaction 写放大。
    每层满了后，将该层数据与下层全部数据归并（下层只有一份有序 run）。
    近似 WA ≈ size_ratio × num_levels / 2
    实际上 leveling 的 WA 略高于 tiering，因为每次归并涉及更多数据。
    教科书近似两者都是 O(T×L)，leveling 常数稍大。
    """
    return spec.size_ratio * spec.num_levels / 2 * 1.2


def space_amplification(spec: LSMSpec) -> float:
    """
    空间放大。Tiering 每层内存多份 run，重叠多，SA ≈ size_ratio / 2。
    Leveling 每层只有一份有序 run，SA ≈ 1 + 1/size_ratio（约 1.1）。
    """
    if spec.compaction == "tiering":
        return spec.size_ratio / 2
    else:
        return 1 + 1 / spec.size_ratio


def num_sstables(spec: LSMSpec, n: int) -> int:
    """
    磁盘上 SSTable 总数。Tiering 每层可有多个 run，leveling 每层一个 run。
    简化：按每层 SSTable 目标大小估算磁盘占用条目数，除以每 SSTable 容量。
    """
    total_bytes = n * spec.entry_size
    if total_bytes <= spec.memtable_size:
        return 0
    # 磁盘上的字节数（含空间放大）
    disk_bytes = total_bytes * space_amplification(spec)
    # 每 SSTable 容量（字节）
    return math.ceil(disk_bytes / spec.sst_size)


def point_lookup_io(spec: LSMSpec, n: int) -> tuple:
    """
    点查 IO。
    最坏情况：MemTable + 每层一个 SSTable = num_levels + 1 次磁盘读。
    期望情况：每层有一个布隆过滤器，可以秒判该层有没有这个 key。
    过滤器说"没有"则跳过该层（0 次磁盘读），说"可能有"则读 1 个 SSTable。
    假阳性率 p（典型 1%）时，期望磁盘读 ≈ num_levels × p（大部分层被过滤掉）。
    """
    worst = spec.num_levels + 1  # MemTable + 每层一个 SSTable
    bloom_fpr = 0.01  # 布隆过滤器假阳性率，典型 1%
    expected = worst * bloom_fpr  # 大部分层被布隆过滤，只读假阳性层
    return worst, expected


def range_scan_io(spec: LSMSpec, n: int) -> int:
    """
    范围扫描 IO。要合并所有层的 SSTable（每层一份），用最小堆归并。
    IO 次数 ≈ num_levels × 2（每层读一个 SSTable 的索引块 + 数据块）。
    实际范围扫描是 LSM-tree 的弱项，因为要查多层。
    """
    return spec.num_levels * 2


def compute(spec: LSMSpec, n: int) -> LSMMetrics:
    """给定条目数 n，递推全部指标。"""
    total_bytes = n * spec.entry_size
    if spec.compaction == "tiering":
        wa = write_amplification_tiering(spec)
    else:
        wa = write_amplification_leveling(spec)
    sa = space_amplification(spec)
    worst_io, exp_io = point_lookup_io(spec, n)
    n_sst = num_sstables(spec, n)
    return LSMMetrics(
        n=n,
        total_data_bytes=total_bytes,
        write_amplification=wa,
        space_amplification=sa,
        point_lookup_worst_io=worst_io,
        point_lookup_expected_io=exp_io,
        range_scan_io=range_scan_io(spec, n),
        num_sstables=n_sst,
    )


def _selftest():
    """公式自洽性检查"""
    spec = LSMSpec()

    # 写放大应大于 1（LSM 的核心代价）
    m = compute(spec, 1_000_000)
    assert m.write_amplification > 1, "写放大应大于 1"

    # 层数越多写放大越大
    spec_more = LSMSpec(num_levels=10)
    assert compute(spec_more, 1_000_000).write_amplification > m.write_amplification

    # size_ratio 越大写放大越大
    spec_big = LSMSpec(size_ratio=20)
    assert compute(spec_big, 1_000_000).write_amplification > m.write_amplification

    # tiering 空间放大 > leveling（tiering 多层版本共存）
    m_tier = compute(LSMSpec(compaction="tiering"), 1_000_000)
    m_level = compute(LSMSpec(compaction="leveling"), 1_000_000)
    assert m_tier.space_amplification > m_level.space_amplification, \
        "tiering 空间放大应大于 leveling"

    # 点查最坏 IO = 层数 + 1
    assert m.point_lookup_worst_io == spec.num_levels + 1

    # 期望 IO 远小于最坏 IO（布隆过滤器大部分过滤掉）
    assert m.point_lookup_expected_io < m.point_lookup_worst_io

    print("selftest 全部通过")
    # 演示
    for comp in ["tiering", "leveling"]:
        s = LSMSpec(compaction=comp)
        mm = compute(s, 10_000_000)
        print(f"[{comp:8s}] n=10M  WA={mm.write_amplification:.1f}x  "
              f"SA={mm.space_amplification:.2f}x  "
              f"point_io_worst={mm.point_lookup_worst_io}  "
              f"point_io_exp={mm.point_lookup_expected_io:.2f}  "
              f"range_io={mm.range_scan_io}")


if __name__ == "__main__":
    _selftest()
