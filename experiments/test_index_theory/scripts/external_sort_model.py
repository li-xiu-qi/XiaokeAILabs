# -*- coding: utf-8 -*-
"""
外部排序（External Sort）理论性能模型

外部排序是数据库 ORDER BY、大规模数据排序的基础。当数据量超过内存时，需要分块排序 + 多路归并。

核心流程：
1. 分块（Run Generation）：将大文件分成多个块，每块加载到内存排序，写回磁盘
2. 归并（Merge）：多路归并所有有序块，生成最终排序结果

核心指标：
1. 归并轮数：ceil(log_{fan_in}(num_runs))，fan_in 是归并路数
2. I/O 次数：读数据 num_runs + 1 次（每轮读一次，最后一轮读一次）
3. 内存占用：fan_in × buffer_size（归并缓冲区）
4. 排序时间：分块排序时间 + 归并时间

标准假设：
- 总数据量 N 字节
- 内存容量 M 字节
- 页大小 P 字节
- 归并路数 fan_in = M / (2 × P)（每路两个缓冲区：读 + 写）

本模块实现这些公式，参数可配。
"""
from dataclasses import dataclass
import math


@dataclass
class ExternalSortSpec:
    """外部排序参数"""
    total_bytes: int = 10 * 1024 * 1024 * 1024  # 总数据量（10 GB）
    memory_bytes: int = 1 * 1024 * 1024 * 1024   # 内存容量（1 GB）
    page_size: int = 4096                          # 页大小（字节）
    fan_in: int = 0                                # 归并路数（0 表示自动计算）


@dataclass
class ExternalSortMetrics:
    total_bytes: int               # 总数据量
    memory_bytes: int              # 内存容量
    num_runs: int                  # 初始有序块数
    fan_in: int                    # 归并路数
    merge_rounds: int              # 归并轮数
    total_io_reads: int            # 总读次数（页数）
    total_io_writes: int           # 总写次数（页数）
    memory_for_merge: int          # 归并内存占用（字节）


def compute_num_runs(spec: ExternalSortSpec) -> int:
    """初始有序块数。N / M，向上取整。"""
    return math.ceil(spec.total_bytes / spec.memory_bytes)


def compute_fan_in(spec: ExternalSortSpec) -> int:
    """归并路数。M / (2 × P)，每路两个缓冲区。"""
    if spec.fan_in > 0:
        return spec.fan_in
    return max(2, spec.memory_bytes // (2 * spec.page_size))


def compute_merge_rounds(num_runs: int, fan_in: int) -> int:
    """归并轮数。ceil(log_{fan_in}(num_runs))。"""
    if num_runs <= 1:
        return 0
    return math.ceil(math.log(num_runs) / math.log(fan_in))


def compute_io(spec: ExternalSortSpec, num_runs: int, fan_in: int, merge_rounds: int) -> tuple:
    """
    I/O 次数。
    分块阶段：读 N 一次，写 N 一次（有序块写回）。
    归并阶段：每轮读一次，最后一轮写一次。
    总读 = N + merge_rounds × N
    总写 = N + N（最后一轮）
    """
    pages_total = spec.total_bytes // spec.page_size
    # 分块：读一次 + 写一次
    read_pages = pages_total
    write_pages = pages_total
    # 归并：每轮读一次
    read_pages += merge_rounds * pages_total
    # 最后一轮写一次
    write_pages += pages_total
    return read_pages, write_pages


def compute(spec: ExternalSortSpec) -> ExternalSortMetrics:
    """给定参数，递推全部指标。"""
    num_runs = compute_num_runs(spec)
    fan_in = compute_fan_in(spec)
    merge_rounds = compute_merge_rounds(num_runs, fan_in)
    read_pages, write_pages = compute_io(spec, num_runs, fan_in, merge_rounds)
    merge_memory = fan_in * 2 * spec.page_size  # 每路两个缓冲区

    return ExternalSortMetrics(
        total_bytes=spec.total_bytes,
        memory_bytes=spec.memory_bytes,
        num_runs=num_runs,
        fan_in=fan_in,
        merge_rounds=merge_rounds,
        total_io_reads=read_pages,
        total_io_writes=write_pages,
        memory_for_merge=merge_memory,
    )


def _selftest():
    """公式自洽性检查"""
    spec = ExternalSortSpec()

    # 归并轮数应为非负整数
    m = compute(spec)
    assert m.merge_rounds >= 0

    # fan_in 越大，归并轮数越少
    spec_big = ExternalSortSpec(fan_in=64)
    assert compute(spec_big).merge_rounds <= m.merge_rounds, "fan_in 越大，轮数应越少"

    # 内存越大，初始有序块数越少
    spec_big_mem = ExternalSortSpec(memory_bytes=2 * 1024 * 1024 * 1024)
    assert compute(spec_big_mem).num_runs < m.num_runs, "内存越大，块数应越少"

    # I/O 应为正
    assert m.total_io_reads > 0
    assert m.total_io_writes > 0

    print("selftest 全部通过")
    # 演示
    print(f"\n总数据={m.total_bytes/1024/1024/1024:.1f} GiB  "
          f"内存={m.memory_bytes/1024/1024/1024:.1f} GiB")
    print(f"初始有序块={m.num_runs}  归并路数={m.fan_in}  "
          f"归并轮数={m.merge_rounds}")
    print(f"I/O: 读 {m.total_io_reads:,} 页, 写 {m.total_io_writes:,} 页  "
          f"归并内存={m.memory_for_merge/1024:.1f} KiB")


if __name__ == "__main__":
    _selftest()
