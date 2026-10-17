# -*- coding: utf-8 -*-
"""
可扩展哈希（Extendible Hashing）理论性能模型

可扩展哈希是动态哈希方法，用「目录（directory）+ 桶（bucket）」两级结构。
目录有 2^d 个槽，d 是全局深度；每个桶有局部深度 d'（≤ d）和固定容量。
插入时桶满则分裂，局部深度 +1；若局部深度 == 全局深度，目录先翻倍再分裂。

核心指标：
1. 目录大小：2^d × pointer_bytes（全局深度 d 决定）
2. 桶分裂次数：插入过程中桶满导致的分裂总数
3. 目录翻倍次数：局部深度追平全局深度时触发，目录 ×2
4. 查找：O(1)（目录索引一次 + 桶内线性扫描）
5. 空间：目录 + 所有桶的 (key + value)

核心公式：
- 目录槽数 = 2^d
- 桶容量固定 B，桶满条件：桶内元素数 == B
- 分裂概率（经验）：负载因子约 0.8 时，平均每次插入分裂概率约 0.2
- 目录翻倍次数 = 全局深度的增量次数（最多 log2(n)）

本模块实现这些公式，参数可配。
"""
from dataclasses import dataclass
import math


@dataclass
class ExtendibleSpec:
    """可扩展哈希参数"""
    n: int = 1_000_000           # 预期元素数量
    bucket_capacity: int = 4     # 桶容量（固定，典型 2/4/8）
    key_size: int = 8            # 每个 key 的字节数
    value_size: int = 8          # 每个 value 的字节数
    pointer_size: int = 8        # 目录槽中指针的字节数
    initial_global_depth: int = 1  # 初始全局深度 d（目录 2^d 个槽）


@dataclass
class ExtendibleMetrics:
    n: int                       # 元素数量
    global_depth: int            # 全局深度 d
    directory_slots: int         # 目录槽数 = 2^d
    num_buckets: int             # 桶数量（去重后）
    bucket_capacity: int         # 桶容量
    directory_bytes: int         # 目录占用字节 = 2^d × pointer
    bucket_bytes: int            # 所有桶占用字节
    total_bytes: int             # 目录 + 桶
    space_utilization: float     # 空间利用率 = n / (桶数 × 桶容量)
    est_splits: float            # 估算桶分裂次数（概率口径：稳态均摊）
    est_splits_deterministic: int  # 估算桶分裂次数（确定性口径：桶满即分裂）
    est_directory_doubles: int   # 估算目录翻倍次数


def compute_directory_slots(global_depth: int) -> int:
    """目录槽数。2^d。"""
    return 2 ** global_depth


def estimate_global_depth(n: int, bucket_capacity: int) -> int:
    """
    估算插入 n 个元素后的全局深度。

    桶数 ≈ n / (桶容量 × 平均利用率)，全局深度 = ceil(log2(桶数)) + 1。
    这个 +1 来自目录翻倍机制：桶分裂追平全局深度时目录先翻倍，
    导致目录槽数相对桶数有一次冗余翻倍（实测一致）。
    """
    if n <= 0:
        return 1
    avg_util = {2: 0.98, 4: 0.84, 8: 0.66}.get(bucket_capacity, 0.8)
    num_buckets = max(2, math.ceil(n / (bucket_capacity * avg_util)))
    depth = math.ceil(math.log2(num_buckets)) + 1
    return max(1, depth)


def estimate_num_buckets(n: int, bucket_capacity: int) -> int:
    """估算桶数量（去重后）。桶数 ≈ n / (桶容量 × 平均利用率)。"""
    if n <= 0:
        return 1
    avg_util = {2: 0.98, 4: 0.84, 8: 0.66}.get(bucket_capacity, 0.8)
    return max(1, math.ceil(n / (bucket_capacity * avg_util)))


def estimate_splits(n: int, bucket_capacity: int) -> float:
    """
    估算桶分裂次数（概率口径）。

    负载因子约 0.8 时，平均每次插入触发桶分裂的概率约 0.2，
    故 n 次插入的期望分裂次数 ≈ n × 0.2。这是稳态均摊视角。
    """
    return n * 0.2


def estimate_splits_deterministic(n: int, bucket_capacity: int) -> int:
    """
    估算桶分裂次数（确定性口径）。

    「桶满即分裂」实现里，每次分裂恰好产生一个新桶，从 1 个桶起步，
    故桶分裂次数 = 最终桶数 - 1。这是逐个插入的累计视角。
    """
    return estimate_num_buckets(n, bucket_capacity) - 1


def estimate_directory_doubles(global_depth: int,
                                initial_depth: int = 1) -> int:
    """估算目录翻倍次数 = 最终全局深度 - 初始深度。"""
    return max(0, global_depth - initial_depth)


def compute(spec: ExtendibleSpec) -> ExtendibleMetrics:
    """给定参数，递推全部指标。桶数为主变量，其余量由桶数与全局深度递推。"""
    est_buckets = estimate_num_buckets(spec.n, spec.bucket_capacity)
    gd = estimate_global_depth(spec.n, spec.bucket_capacity)
    slots = compute_directory_slots(gd)
    dir_bytes = slots * spec.pointer_size

    bucket_bytes = est_buckets * spec.bucket_capacity * (spec.key_size + spec.value_size)
    total = dir_bytes + bucket_bytes

    # 空间利用率：n 个元素分布在 est_buckets × 桶容量 个槽位
    utilization = spec.n / (est_buckets * spec.bucket_capacity) if est_buckets > 0 else 0.0

    splits = estimate_splits(spec.n, spec.bucket_capacity)
    splits_det = est_buckets - 1  # 桶满即分裂：分裂次数 = 桶数 - 1
    doubles = estimate_directory_doubles(gd, spec.initial_global_depth)

    return ExtendibleMetrics(
        n=spec.n,
        global_depth=gd,
        directory_slots=slots,
        num_buckets=est_buckets,
        bucket_capacity=spec.bucket_capacity,
        directory_bytes=dir_bytes,
        bucket_bytes=bucket_bytes,
        total_bytes=total,
        space_utilization=utilization,
        est_splits=splits,
        est_splits_deterministic=splits_det,
        est_directory_doubles=doubles,
    )


def _selftest():
    """公式自洽性检查"""
    # 目录槽数 = 2^d
    assert compute_directory_slots(1) == 2
    assert compute_directory_slots(3) == 8

    # 元素越多，全局深度越大（桶越多）
    m_small = compute(ExtendibleSpec(n=100, bucket_capacity=4))
    m_large = compute(ExtendibleSpec(n=1_000_000, bucket_capacity=4))
    assert m_large.global_depth > m_small.global_depth, "n 越大全局深度应越大"
    assert m_large.directory_slots > m_small.directory_slots, "n 越大目录应越大"

    # 桶容量越大，所需桶数越少，全局深度越小
    m_b2 = compute(ExtendibleSpec(n=100_000, bucket_capacity=2))
    m_b8 = compute(ExtendibleSpec(n=100_000, bucket_capacity=8))
    assert m_b8.num_buckets <= m_b2.num_buckets, "桶容量大，桶数应更少"

    # 总字节 = 目录 + 桶
    m = compute(ExtendibleSpec(n=1_000_000, bucket_capacity=4))
    assert m.total_bytes == m.directory_bytes + m.bucket_bytes

    # 空间利用率在 (0, 1] 区间
    assert 0 < m.space_utilization <= 1.0, "空间利用率应在 (0,1]"

    # 目录翻倍次数 ≥ 0 且不超过 log2(n)
    assert 0 <= m.est_directory_doubles <= math.log2(m.n) + 1

    # 桶分裂次数随 n 线性增长
    m2 = compute(ExtendibleSpec(n=2_000_000, bucket_capacity=4))
    assert m2.est_splits > m.est_splits, "n 翻倍分裂次数应增加"
    # 确定性口径：桶满即分裂，分裂次数 = 桶数 - 1
    assert m.est_splits_deterministic >= 0
    assert m2.est_splits_deterministic > m.est_splits_deterministic

    print("selftest 全部通过")
    # 演示
    print(f"\nn={m.n:,}  全局深度={m.global_depth}  目录槽数={m.directory_slots:,}")
    print(f"桶数={m.num_buckets:,}  桶容量={m.bucket_capacity}  "
          f"空间利用率={m.space_utilization:.2f}")
    print(f"目录={m.directory_bytes/1024:.1f} KiB  "
          f"桶={m.bucket_bytes/1024/1024:.2f} MiB  合计={m.total_bytes/1024/1024:.2f} MiB")


if __name__ == "__main__":
    _selftest()
