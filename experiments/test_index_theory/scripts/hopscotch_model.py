# -*- coding: utf-8 -*-
"""
跳房子哈希（Hopscotch Hashing）理论性能模型

Herlihy, Shavit, Tzafrir (2008) 提出。核心是把「邻域」变成一等概念：
每个桶的 home 位置周围 H 个连续槽位构成它的邻域，桶用一个 H 位位图
（hop information）记录邻域内哪些位置存了自己的元素。

不变式：任何元素都存放在它 home 的 H 个邻域槽位之内。
由此查找是常数上界：先查 home，再按位图查最多 H 个位置，O(H)。

核心公式：
- 桶数 = ceil(n / 目标负载因子)，负载因子可达 0.9 以上
- 查找代价：O(H)，H 通常取 32（一个缓存行的槽位数）
- 插入：目标桶满时，从邻域内找空位，用线性探测 + 交换把空位移进邻域
- 位图大小：每桶 H 位 = H/8 字节（H=32 时 4 字节）

本模块实现这些公式，参数可配。
"""
from dataclasses import dataclass
import math


@dataclass
class HopscotchSpec:
    n: int = 1_000_000          # 元素数
    entry_size: int = 16        # 单条目字节数
    target_load: float = 0.9    # 目标负载因子（可达 0.9+）
    neighborhood: int = 32      # 邻域大小 H（位图位数）
    max_displacement: int = 512 # 插入时线性探测的最大位移，超限触发扩容
    max_load: float = 0.9375    # 扩容阈值（15/16）


@dataclass
class HopscotchMetrics:
    n: int
    num_buckets: int            # 桶数
    load_factor: float          # 实际负载因子
    neighborhood: int           # 邻域大小 H
    lookup_bound: int           # 查找上界（最多检查的位置数）= H
    bitmap_bytes: int           # 位图总字节（每桶 H 位）
    storage_bytes: int          # 总存储（位图 + 槽位数组）
    bitmap_ratio: float         # 位图占总存储比例
    avg_lookup_positions: float # 平均查找位置数（邻域内元素个数的期望）
    needs_resize: bool


def num_buckets_for(n: int, target_load: float) -> int:
    """桶数 = ceil(n / 目标负载因子)。"""
    return math.ceil(n / target_load)


def compute_bitmap_bytes(num_buckets: int, neighborhood: int) -> int:
    """
    位图总字节。每桶 H 位 = ceil(H/8) 字节。

    注意这里是每桶一个位图，不是每桶一位。这是跳房子哈希的主要额外开销：
    H=32 时每桶 4 字节，相比普通线性探测每桶 0 字节。
    """
    bytes_per_bucket = (neighborhood + 7) // 8
    return num_buckets * bytes_per_bucket


def avg_lookup_positions(load_factor: float, neighborhood: int) -> float:
    """
    平均查找需要检查的位置数。

    查找先查 home 桶（1 次），home 空则结束；home 有元素则按位图检查邻域内
    属于本桶的元素。属于本桶的元素数在邻域内服从二项分布：
        k ~ Binomial(H, α / H)，期望 = α

    故平均检查位置数 ≈ 1（home）+ α（邻域内属于本桶的元素期望）。
    邻域边界截断会带来小修正，此处忽略。

    α=0.9、H=32 时约 1.90。
    """
    return 1.0 + load_factor


def compute(spec: HopscotchSpec) -> HopscotchMetrics:
    """给定参数，递推全部指标。"""
    buckets = num_buckets_for(spec.n, spec.target_load)
    alpha = spec.n / buckets
    bitmap = compute_bitmap_bytes(buckets, spec.neighborhood)
    slots = buckets * spec.entry_size
    total = bitmap + slots
    return HopscotchMetrics(
        n=spec.n,
        num_buckets=buckets,
        load_factor=alpha,
        neighborhood=spec.neighborhood,
        lookup_bound=spec.neighborhood,
        bitmap_bytes=bitmap,
        storage_bytes=total,
        bitmap_ratio=bitmap / total,
        avg_lookup_positions=avg_lookup_positions(alpha, spec.neighborhood),
        needs_resize=alpha > spec.max_load,
    )


def _selftest():
    """公式自洽性检查"""
    # 负载 0.9 可行（< 阈值 15/16）
    m = compute(HopscotchSpec(n=1_000_000, target_load=0.9))
    assert m.load_factor <= 0.9375 + 1e-9
    assert not m.needs_resize

    # 超过阈值应扩容
    m_over = compute(HopscotchSpec(n=1000, target_load=0.95))
    assert m_over.needs_resize, "负载 0.95 超过 15/16 应触发扩容"

    # 查找上界 = H
    assert m.lookup_bound == m.neighborhood == 32

    # 位图字节：每桶 ceil(H/8)
    spec = HopscotchSpec(n=1000, neighborhood=32)
    mm = compute(spec)
    assert mm.bitmap_bytes == mm.num_buckets * 4

    # H=16 时每桶 2 字节
    m16 = compute(HopscotchSpec(n=1000, neighborhood=16))
    assert m16.bitmap_bytes == m16.num_buckets * 2

    # 存储 = 位图 + 槽位
    assert mm.storage_bytes == mm.bitmap_bytes + mm.num_buckets * 16

    # 平均查找位置 ≈ 1 + α
    assert abs(m.avg_lookup_positions - 1.9) < 0.01, \
        f"α=0.9 平均查找应约 1.9, 实际 {m.avg_lookup_positions}"

    # 邻域越大，位图开销越大，查找上界也越大
    m_small = compute(HopscotchSpec(n=1_000_000, neighborhood=8))
    m_big = compute(HopscotchSpec(n=1_000_000, neighborhood=64))
    assert m_big.bitmap_bytes > m_small.bitmap_bytes
    assert m_big.lookup_bound > m_small.lookup_bound

    # 负载越高，平均查找位置越多
    m_low = compute(HopscotchSpec(n=1000, target_load=0.5))
    m_high = compute(HopscotchSpec(n=1000, target_load=0.9))
    assert m_high.avg_lookup_positions > m_low.avg_lookup_positions

    print("selftest 全部通过")
    print(f"\nn={m.n:,}  桶数={m.num_buckets:,}  负载={m.load_factor:.4f}  H={m.neighborhood}")
    print(f"位图={m.bitmap_bytes/1024/1024:.2f} MiB  查找上界={m.lookup_bound}  "
          f"平均查找位置={m.avg_lookup_positions:.2f}")
    print(f"\n不同 H 与负载下的指标（条目 16 字节）：")
    print(f"{'α':>6} {'H':>4} {'位图占比':>10} {'查找上界':>10} {'平均位置':>10}")
    for lf in [0.5, 0.7, 0.9]:
        for h in [8, 16, 32]:
            mm = compute(HopscotchSpec(n=1_000_000, target_load=lf, neighborhood=h))
            print(f"{mm.load_factor:>6.3f} {h:>4} {mm.bitmap_ratio:>10.2%} "
                  f"{mm.lookup_bound:>10} {mm.avg_lookup_positions:>10.2f}")


if __name__ == "__main__":
    _selftest()
