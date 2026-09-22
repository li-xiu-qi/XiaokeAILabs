# -*- coding: utf-8 -*-
"""
B+树理论性能模型（核心计算器）

纯闭式解，不依赖任何真实数据库。给定数据规模和页参数，递推出：
- 扇出 fanout、叶子容量 leaf_cap
- 树高 height（点查 IO 次数）
- 存储页数 storage_pages、存储字节 storage_bytes
- 范围扫描 m 个键的 IO 次数 range_scan_io

公式来源：标准 B+树分析（Database System Concepts, Ramakrishnan/Gehrke）。
常数（page_overhead）由 sqlite 实测标定，见 verify_btree_storage.py。

用法：
    from btree_model import BTreeSpec, compute
    spec = BTreeSpec(key_size=8, value_size=100)
    m = compute(spec, n=1_000_000)
    print(m.height, m.storage_bytes, m.point_io)
"""

from dataclasses import dataclass
import math


@dataclass
class BTreeSpec:
    """B+树的物理与逻辑参数"""
    page_size: int = 4096        # 页字节数，sqlite/InnoDB 常用 4096
    key_size: int = 8            # 键字节数（BIGINT=8，UUID=16）
    value_size: int = 100        # 叶子记录值字节数（聚簇）或指向数据的指针（二级索引）
    ptr_size: int = 8            # 子节点指针字节数
    page_overhead: int = 100     # 每页保留字节（页头 + cell 指针数组等）
    fill_factor: float = 0.70    # 填充因子，随机插入的 B+树典型 0.6~0.7


@dataclass
class BTreeMetrics:
    """compute() 的返回结果"""
    n: int                       # 记录数
    fanout: int                  # 内部节点扇出（最大子节点数）
    leaf_cap: int                # 每叶子最多记录数
    height: int                  # 树高（层数，等于点查 IO 次数）
    leaf_pages: int              # 叶子页数
    internal_pages: int          # 内部节点页数
    storage_pages: int           # 总页数
    storage_bytes: int           # 存储字节（= storage_pages * page_size）
    point_io: int                # 点查 IO 次数 = height
    range_scan_io: int           # 范围扫描 IO 次数（对 m 个键）


def _fanout(spec: BTreeSpec) -> int:
    """内部节点扇出。每项 = key + child_ptr，加页开销，乘填充因子。"""
    usable = spec.page_size - spec.page_overhead
    per_entry = spec.key_size + spec.ptr_size
    f = int(usable * spec.fill_factor / per_entry)
    return max(f, 2)


def _leaf_cap(spec: BTreeSpec) -> int:
    """每叶子最多记录数。每项 = key + value，加页开销，乘填充因子。"""
    usable = spec.page_size - spec.page_overhead
    per_entry = spec.key_size + spec.value_size
    c = int(usable * spec.fill_factor / per_entry)
    return max(c, 1)


def _height(num_leaf: int, fanout: int) -> int:
    """
    树高（层数）。根在 layer 0，叶在 layer h-1。
    layer k 最多 fanout^k 个节点，需要 fanout^(h-1) >= num_leaf。
    """
    if num_leaf <= 1:
        return 1
    return math.ceil(math.log(num_leaf, fanout)) + 1


def _internal_pages(num_leaf: int, fanout: int, height: int) -> int:
    """
    内部节点总页数。逐层累加：layer 1 有 ceil(num_leaf/fanout) 个节点，
    layer 2 有 ceil(上一层/fanout)，直到根（1 个）。
    """
    if height <= 1:
        return 0
    total = 0
    nodes_at_level = num_leaf
    for _ in range(height - 1):  # 从叶子上一层累加到根的下一层
        nodes_at_level = math.ceil(nodes_at_level / fanout)
        total += nodes_at_level
    return total


def range_scan_io(spec: BTreeSpec, n: int, m: int) -> int:
    """
    范围扫描 m 个连续键的 IO 次数。
    1 次点查定位起始叶，之后顺序读跨越的叶子页。
    叶子可能部分填充，扫描 m 条跨越 ceil(m / leaf_cap) 个叶页。
    """
    if m <= 0:
        return 0
    leaf_cap = _leaf_cap(spec)
    h = _height(math.ceil(n / leaf_cap), _fanout(spec))
    leaf_pages_spanned = math.ceil(m / leaf_cap)
    return h - 1 + leaf_pages_spanned  # 根到叶路径上最后一段由扫描接管


def compute(spec: BTreeSpec, n: int) -> BTreeMetrics:
    """给定记录数 n，递推出全部指标。"""
    fanout = _fanout(spec)
    leaf_cap = _leaf_cap(spec)
    num_leaf = math.ceil(n / leaf_cap)
    height = _height(num_leaf, fanout)
    internal_pages = _internal_pages(num_leaf, fanout, height)
    leaf_pages = num_leaf
    storage_pages = leaf_pages + internal_pages
    storage_bytes = storage_pages * spec.page_size
    return BTreeMetrics(
        n=n,
        fanout=fanout,
        leaf_cap=leaf_cap,
        height=height,
        leaf_pages=leaf_pages,
        internal_pages=internal_pages,
        storage_pages=storage_pages,
        storage_bytes=storage_bytes,
        point_io=height,
        range_scan_io=range_scan_io(spec, n, m=leaf_cap),
    )


def _selftest():
    """公式自洽性检查：树高随 n 单调不减、存储随 n 线性、扇出为正等。"""
    spec = BTreeSpec(key_size=8, value_size=100)
    prev_h = 0
    for n in [1, 10, 100, 1000, 10_000, 100_000, 1_000_000, 10_000_000]:
        m = compute(spec, n)
        assert m.height >= prev_h, f"树高应单调不减: n={n} h={m.height} < {prev_h}"
        prev_h = m.height
        assert m.point_io == m.height
        assert m.storage_bytes == m.storage_pages * spec.page_size
        assert m.fanout >= 2 and m.leaf_cap >= 1
        # 存储字节下界：至少装得下 n 条记录
        assert m.storage_bytes >= n * (spec.key_size + spec.value_size)

    # 大键 -> 小扇出 -> 高树（符合直觉：键越大扇出越小树越高）
    small_key = compute(BTreeSpec(key_size=8, value_size=100), 1_000_000)
    big_key = compute(BTreeSpec(key_size=64, value_size=100), 1_000_000)
    assert small_key.fanout > big_key.fanout, "大键应使扇出变小"
    assert small_key.height <= big_key.height, "扇出小则树高"

    # 树高对数增长：n 涨 1000 倍，树高只涨几层
    h1 = compute(spec, 10_000).height
    h2 = compute(spec, 10_000_000).height
    growth = h2 - h1
    assert 0 < growth <= 4, f"n 涨 1000 倍树高只应涨几层, 实际涨 {growth}"

    print("selftest 全部通过")
    # 演示一个典型规模
    for n in [10_000, 1_000_000, 100_000_000]:
        m = compute(spec, n)
        print(f"n={n:>12,}  fanout={m.fanout:>4}  leaf_cap={m.leaf_cap:>4}  "
              f"height={m.height}  point_io={m.point_io}  "
              f"storage={m.storage_bytes/1024/1024:8.1f} MiB")


if __name__ == "__main__":
    _selftest()
