# -*- coding: utf-8 -*-
"""
Join 算法理论性能模型（核心计算器）

纯闭式解，不依赖真实数据库。Join 是数据库查询的核心操作，三种基础算法：
1. Nested Loop Join（NLJ）：双重循环，最基础但最慢
2. Hash Join（HJ）：构建哈希表 + 探测，等值 join 的最优解
3. Sort-Merge Join（SMJ）：排序 + 归并，适合大数据集、有序输入

核心指标：
1. CPU 时间：比较次数、哈希计算、排序比较
2. I/O 次数：读取表的数据块数（假设无缓存，全表扫描）
3. 内存占用：哈希表大小、排序缓冲区

标准假设：
- 表 R 有 N_R 行，表 S 有 N_S 行
- 结果集有 N_J 行（join 结果数量）
- 页大小 4 KB，每行 100 字节
- 内存可容纳 M 页（buffer pool 大小）

NLJ（Nested Loop Join）：
- 外层循环 R，内层循环 S
- CPU 比较次数：N_R × N_S（最坏情况，无索引）
- I/O：读 R 一次（N_R × row_size / page_size）+ 读 S N_R 次（每行 R 都要全表扫 S）
- 最坏 I/O：N_R × (N_S × row_size / page_size)

Hash Join：
- 构建阶段：读 S 一次，构建哈希表（S 的 join 键）
- 探测阶段：读 R 一次，探测哈希表
- CPU：哈希计算 N_R + N_S 次，比较 N_R 次（平均）
- I/O：读 R 一次 + 读 S 一次 = (N_R + N_S) × row_size / page_size
- 内存：哈希表大小 = N_S × row_size

Sort-Merge Join：
- 排序阶段：R 和 S 各自排序（外部排序，多路归并）
- 归并阶段：线性归并，每表只读一次
- CPU：排序比较 N_R × log(N_R) + N_S × log(N_S)，归并比较 N_J
- I/O：排序阶段读写各一次（2 × (N_R + N_S)），归并阶段再读一次
- 总 I/O：3 × (N_R + N_S) × row_size / page_size（读 3 次，写 1 次）

本模块实现这些公式，参数可配。
"""
from dataclasses import dataclass
import math


@dataclass
class JoinSpec:
    """Join 参数"""
    n_r: int = 1_000_000          # 表 R 的行数
    n_s: int = 100_000            # 表 S 的行数
    row_size: int = 100           # 每行字节数
    page_size: int = 4096         # 页大小（字节）
    memory_pages: int = 1000      # 内存可容纳的页数（buffer pool）
    join_result_size: int = 100_000  # join 结果行数（估算）


@dataclass
class JoinMetrics:
    n_r: int
    n_s: int
    # NLJ
    nlj_cpu_comparisons: int
    nlj_io_reads: int
    # Hash Join
    hj_cpu_hashes: int
    hj_io_reads: int
    hj_memory_bytes: int
    # Sort-Merge Join
    smj_cpu_comparisons: int
    smj_io_reads: int
    smj_io_writes: int


def nlj_metrics(spec: JoinSpec) -> dict:
    """
    Nested Loop Join。
    外层 R，内层 S。假设无索引，内层全表扫描。
    """
    # CPU 比较次数：R 的每行都要和 S 的每行比较
    cpu_comparisons = spec.n_r * spec.n_s

    # I/O：读 R 一次 + 对 R 的每行都读 S 一次
    r_pages = math.ceil(spec.n_r * spec.row_size / spec.page_size)
    s_pages = math.ceil(spec.n_s * spec.row_size / spec.page_size)
    io_reads = r_pages + spec.n_r * s_pages

    return {
        "cpu_comparisons": cpu_comparisons,
        "io_reads": io_reads,
    }


def hash_join_metrics(spec: JoinSpec) -> dict:
    """
    Hash Join。
    构建阶段：读 S 一次，构建哈希表。
    探测阶段：读 R 一次，探测哈希表。
    """
    # CPU：哈希计算 N_S（构建）+ N_R（探测），比较 N_R 次（平均）
    cpu_hashes = spec.n_r + spec.n_s

    # I/O：读 R 一次 + 读 S 一次
    r_pages = math.ceil(spec.n_r * spec.row_size / spec.page_size)
    s_pages = math.ceil(spec.n_s * spec.row_size / spec.page_size)
    io_reads = r_pages + s_pages

    # 内存：哈希表大小 = N_S × row_size（存储 S 的 join 键 + 行指针）
    memory_bytes = spec.n_s * (spec.row_size + 8)  # +8 字节指针

    return {
        "cpu_hashes": cpu_hashes,
        "io_reads": io_reads,
        "memory_bytes": memory_bytes,
    }


def sort_merge_join_metrics(spec: JoinSpec) -> dict:
    """
    Sort-Merge Join。
    排序阶段：外部排序（假设内存足够，用内存排序）。
    归并阶段：线性归并。
    """
    # CPU：排序比较 N log N，归并比较 N_J
    sort_comparisons = spec.n_r * math.log2(spec.n_r) + spec.n_s * math.log2(spec.n_s)
    merge_comparisons = spec.join_result_size
    cpu_comparisons = int(sort_comparisons + merge_comparisons)

    # I/O：排序阶段读一次 + 写一次（外部排序），归并阶段再读一次
    # 简化：假设内存排序，不落盘，所以只读一次
    r_pages = math.ceil(spec.n_r * spec.row_size / spec.page_size)
    s_pages = math.ceil(spec.n_s * spec.row_size / spec.page_size)
    io_reads = r_pages + s_pages
    io_writes = 0  # 内存排序，不落盘

    # 如果内存不够，需要外部排序，增加 I/O
    total_pages = r_pages + s_pages
    if total_pages > spec.memory_pages:
        # 外部排序：读写各一次（多路归并）
        io_writes = r_pages + s_pages
        io_reads += io_writes  # 归并时再读一次

    return {
        "cpu_comparisons": cpu_comparisons,
        "io_reads": io_reads,
        "io_writes": io_writes,
    }


def compute(spec: JoinSpec) -> JoinMetrics:
    """给定参数，递推全部指标。"""
    nlj = nlj_metrics(spec)
    hj = hash_join_metrics(spec)
    smj = sort_merge_join_metrics(spec)

    return JoinMetrics(
        n_r=spec.n_r,
        n_s=spec.n_s,
        nlj_cpu_comparisons=nlj["cpu_comparisons"],
        nlj_io_reads=nlj["io_reads"],
        hj_cpu_hashes=hj["cpu_hashes"],
        hj_io_reads=hj["io_reads"],
        hj_memory_bytes=hj["memory_bytes"],
        smj_cpu_comparisons=smj["cpu_comparisons"],
        smj_io_reads=smj["io_reads"],
        smj_io_writes=smj["io_writes"],
    )


def _selftest():
    """公式自洽性检查"""
    spec = JoinSpec()

    # NLJ I/O 应远大于 Hash Join
    m = compute(spec)
    assert m.nlj_io_reads > m.hj_io_reads * 10, "NLJ I/O 应远大于 Hash Join"

    # Hash Join I/O 最小（读一次）
    r_pages = math.ceil(spec.n_r * spec.row_size / spec.page_size)
    s_pages = math.ceil(spec.n_s * spec.row_size / spec.page_size)
    assert m.hj_io_reads == r_pages + s_pages

    # Sort-Merge CPU 应小于 NLJ（比较次数）
    assert m.smj_cpu_comparisons < m.nlj_cpu_comparisons

    # Hash Join 内存应大于 0
    assert m.hj_memory_bytes > 0

    # 内存越大，外部排序 I/O 越少
    spec_small_mem = JoinSpec(memory_pages=10)
    spec_big_mem = JoinSpec(memory_pages=100000)
    smj_small = compute(spec_small_mem)
    smj_big = compute(spec_big_mem)
    assert smj_small.smj_io_writes > smj_big.smj_io_writes, "内存越大，外部排序 I/O 应越少"

    print("selftest 全部通过")
    # 演示
    print(f"\nR={m.n_r:,} 行, S={m.n_s:,} 行")
    print(f"NLJ:      CPU {m.nlj_cpu_comparisons:,} 次比较, I/O {m.nlj_io_reads:,} 次读")
    print(f"Hash Join: CPU {m.hj_cpu_hashes:,} 次哈希, I/O {m.hj_io_reads:,} 次读, "
          f"内存 {m.hj_memory_bytes/1024/1024:.1f} MiB")
    print(f"Sort-Merge: CPU {m.smj_cpu_comparisons:,} 次比较, I/O {m.smj_io_reads:,} 次读, "
          f"{m.smj_io_writes:,} 次写")


if __name__ == "__main__":
    _selftest()
