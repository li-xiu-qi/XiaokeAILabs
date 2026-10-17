# -*- coding: utf-8 -*-
"""
布谷鸟过滤器（Cuckoo Filter）理论性能模型

布谷鸟过滤器是 Bloom Filter 的增强型替代，用布谷鸟哈希（cuckoo hashing）存储元素指纹。
它在保持低假阳性率的同时首次完整支持删除操作，且具备更好的空间效率与缓存局部性。

核心结构（Fan et al., CoNEXT 2014）：
- m 个桶（bucket），每桶 b 个槽位（典型 b=4）
- 每个元素计算一个 f 位指纹（fingerprint），只存指纹不存完整元素
- 每个元素有两个候选桶：h1(x) 与 h2(x) = h1(x) XOR hash(fingerprint(x)) mod m
  （partial-key cuckoo hashing，用指纹反推第二个位置，省掉第二个哈希）
- 插入：两个桶有空槽则放入；都满则随机踢出一个元素，把它重插到它的另一个候选桶，递归
- 查询：只看两个桶，命中指纹则"可能存在"
- 删除：从两个桶中精确移除该指纹，不影响其他元素

核心指标：
1. 假阳性率：ε = 1 - (1 - 1/2^f)^(2b)，上界近似 2b/2^f
2. 最小指纹位数：f >= ceil(log2(2b/ε))
3. 空间成本：C = f/α bits per item，α 为负载因子（b=2→84%, b=4→95%, b=8→98%）

与 Bloom Filter 的空间交叉点：
- Bloom：1.44·log2(1/ε) bits/item
- Cuckoo：(log2(1/ε) + log2(2b))/α bits/item
- 令两者相等得 log2(1/ε) ≈ 8.15，即 ε ≈ 0.0028
- ε 比 0.28% 更低时布谷鸟更省；更高时布隆更省（但布隆不支持删除）

本模块实现这些公式，参数可配。
"""
from dataclasses import dataclass
import math


@dataclass
class CuckooFilterSpec:
    """布谷鸟过滤器参数"""
    n: int = 1_000_000           # 预期元素数量
    bucket_size: int = 4         # 每桶槽数 b（1/2/4/8）
    fingerprint_bits: int = 0    # 指纹位数 f（0 表示按 target_fpr 自动计算）
    target_fpr: float = 0.01     # 目标假阳性率（f 自动计算时使用）
    load_factor: float = 0.95    # 目标负载因子 α
    semi_sort: bool = False      # 是否启用半排序桶优化


@dataclass
class CuckooFilterMetrics:
    n: int                       # 元素数量
    bucket_size: int             # 每桶槽数 b
    fingerprint_bits: int        # 指纹位数 f
    load_factor: float           # 负载因子 α
    num_buckets: int             # 桶数 m
    num_slots: int               # 总槽数 m·b
    fpr: float                   # 假阳性率
    memory_bits: int             # 指纹区占用比特数
    memory_bytes: int            # 指纹区占用字节数
    bits_per_item: float         # 每元素比特数 C = f/α
    capacity: int                # 可容纳元素数（floor(α·m·b)）


# 论文实测的理论负载因子上界（partial-key cuckoo hashing, k=2 哈希函数）
# b=1→50%, b=2→84%, b=4→95%, b=8→98%（Fan et al. 2014, Section 5.1）
THEORETICAL_LOAD_FACTOR = {1: 0.50, 2: 0.84, 4: 0.95, 8: 0.98}


def compute_fpr(fingerprint_bits: int, bucket_size: int,
                load_factor: float = 1.0) -> float:
    """假阳性率。ε = 1 - (1 - 1/2^f)^(2b)。

    load_factor < 1 时按"每个槽位以 α 概率被占用"细化：
    ε = 1 - (1 - α/2^f)^(2b)。load_factor = 1 时退化为论文上界 2b/2^f。
    """
    p_slot_hit = load_factor / (2 ** fingerprint_bits)
    return 1 - (1 - p_slot_hit) ** (2 * bucket_size)


def compute_fpr_upper_bound(fingerprint_bits: int, bucket_size: int) -> float:
    """假阳性率上界（论文式 5）。2b/2^f，对应所有槽位都被占用的最坏情况。"""
    return (2 * bucket_size) / (2 ** fingerprint_bits)


def compute_min_fingerprint_bits(target_fpr: float, bucket_size: int) -> int:
    """最小指纹位数（论文式 6）。f >= ceil(log2(2b/ε))。"""
    return int(math.ceil(math.log2(2 * bucket_size / target_fpr)))


def compute_num_buckets(n: int, bucket_size: int, load_factor: float) -> int:
    """桶数。m = ceil(n / (b·α))。"""
    return int(math.ceil(n / (bucket_size * load_factor)))


def compute_bits_per_item(fingerprint_bits: int, load_factor: float) -> float:
    """每元素比特数（论文式 4）。C = f/α。"""
    return fingerprint_bits / load_factor


def compute_bits_per_item_semi_sort(target_fpr: float, bucket_size: int,
                                    load_factor: float) -> float:
    """半排序桶优化的每元素比特数（论文 Table 2）。(log2(1/ε) + 2)/α。"""
    return (math.log2(1.0 / target_fpr) + 2) / load_factor


def compute_bloom_bits_per_item(target_fpr: float) -> float:
    """空间最优布隆过滤器的每元素比特数。1.44·log2(1/ε)。"""
    return 1.44 * math.log2(1.0 / target_fpr)


def compute_memory(num_buckets: int, bucket_size: int,
                   fingerprint_bits: int) -> int:
    """指纹区占用比特数。m × b × f。"""
    return num_buckets * bucket_size * fingerprint_bits


def compute_capacity(num_buckets: int, bucket_size: int,
                     load_factor: float) -> int:
    """可容纳元素数。floor(α × m × b)。"""
    return int(num_buckets * bucket_size * load_factor)


def compute(spec: CuckooFilterSpec) -> CuckooFilterMetrics:
    """给定参数，递推全部指标。"""
    # 自动计算指纹位数
    if spec.fingerprint_bits == 0:
        f = compute_min_fingerprint_bits(spec.target_fpr, spec.bucket_size)
    else:
        f = spec.fingerprint_bits

    # 理论负载因子上界：实际目标不得超过该值，否则插入必然失败
    alpha_cap = THEORETICAL_LOAD_FACTOR.get(spec.bucket_size, 0.95)
    alpha = min(spec.load_factor, alpha_cap)

    num_buckets = compute_num_buckets(spec.n, spec.bucket_size, alpha)
    num_slots = num_buckets * spec.bucket_size

    fpr = compute_fpr(f, spec.bucket_size, alpha)
    memory_bits = compute_memory(num_buckets, spec.bucket_size, f)
    bits_per_item = compute_bits_per_item(f, alpha)
    capacity = compute_capacity(num_buckets, spec.bucket_size, alpha)

    return CuckooFilterMetrics(
        n=spec.n,
        bucket_size=spec.bucket_size,
        fingerprint_bits=f,
        load_factor=alpha,
        num_buckets=num_buckets,
        num_slots=num_slots,
        fpr=fpr,
        memory_bits=memory_bits,
        memory_bytes=memory_bits // 8,
        bits_per_item=bits_per_item,
        capacity=capacity,
    )


def _selftest():
    """公式自洽性检查"""
    spec = CuckooFilterSpec(n=1_000_000, bucket_size=4,
                            fingerprint_bits=0, target_fpr=0.01)
    m = compute(spec)

    # 假阳性率应在 0-1 之间
    assert 0 < m.fpr < 1, "假阳性率应在 0-1 之间"

    # 指纹位数越多，假阳性率越低
    f8 = compute(CuckooFilterSpec(n=1000, bucket_size=4, fingerprint_bits=8))
    f16 = compute(CuckooFilterSpec(n=1000, bucket_size=4, fingerprint_bits=16))
    assert f16.fpr < f8.fpr, "指纹位数越多，假阳性率应越低"

    # 目标假阳性率越低，所需指纹位数越多
    f_loose = compute_min_fingerprint_bits(0.1, 4)
    f_tight = compute_min_fingerprint_bits(0.001, 4)
    assert f_tight > f_loose, "目标假阳性率越低，指纹位数应越多"

    # 负载因子越高，假阳性率越高
    low = compute(CuckooFilterSpec(n=1000, bucket_size=4,
                                   fingerprint_bits=12, load_factor=0.5))
    high = compute(CuckooFilterSpec(n=1000, bucket_size=4,
                                    fingerprint_bits=12, load_factor=0.95))
    assert high.fpr > low.fpr, "负载因子越高，假阳性率应越高"

    # 桶越大，负载因子上界越高（b=8 > b=4 > b=2 > b=1）
    assert (THEORETICAL_LOAD_FACTOR[8] > THEORETICAL_LOAD_FACTOR[4]
            > THEORETICAL_LOAD_FACTOR[2] > THEORETICAL_LOAD_FACTOR[1]), \
        "桶越大，理论负载因子应越高"

    # 自动计算的指纹位数应满足目标假阳性率（实测值不超过目标）
    assert m.fpr <= spec.target_fpr, "自动指纹位数应使假阳性率达到目标"

    # 空间交叉点：ε 极低时布谷鸟比布隆省
    assert (compute_bits_per_item(13, 0.95)
            < compute_bloom_bits_per_item(0.001)), "低 ε 时布谷鸟应更省空间"

    # 桶数应能容纳 n 个元素
    assert m.capacity >= spec.n, "容量应不小于 n"

    print("selftest 全部通过")
    print(f"\nn={m.n:,}  b={m.bucket_size}  f={m.fingerprint_bits} bits  "
          f"负载因子={m.load_factor:.0%}")
    print(f"桶数={m.num_buckets:,}  槽数={m.num_slots:,}  "
          f"假阳性率={m.fpr:.4f}  每元素={m.bits_per_item:.2f} bits  "
          f"内存={m.memory_bytes/1024/1024:.2f} MiB")


if __name__ == "__main__":
    _selftest()
