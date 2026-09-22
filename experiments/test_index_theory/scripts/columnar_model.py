# -*- coding: utf-8 -*-
"""
列式存储理论性能模型（核心计算器）

纯闭式解。列式存储的核心优势：分析查询只读需要的列（列裁剪），
顺序读整列（预取友好），压缩比高（同列同质数据），向量化执行（SIMD）。

核心公式：
- 扫描 IO = 命中列的字节数 / 页大小（向上取整）
- 命中列字节 = n × sum(命中列的平均字节) / 压缩比
- 扫描延迟 = 扫描 IO × 页大小 / 顺序读带宽
- 行式存储对比：扫描 IO = n × 行字节 / 页大小（读全部列）

压缩比取决于编码：字典编码（低基数字段）、RLE（重复值）、
位封装（小整数）、Delta（有序序列），典型 2-10 倍。

参数：
- n: 行数
- columns: [(列名, 平均字节, 基数)], 基数决定压缩率（低基数->字典编码压缩率高）
- query_columns: 查询涉及的列名列表
- page_size: 页大小
- compression_ratio: 实际压缩比（由列基数自动估算或手动指定）

本模块给出列式 vs 行式的扫描 IO 对比，验证列裁剪的优势。
"""
from dataclasses import dataclass
import math


@dataclass
class Column:
    name: str
    avg_bytes: float      # 平均每行字节（未压缩）
    cardinality: int      # 基数（不同值的个数）


@dataclass
class ColumnarSpec:
    n: int                           # 行数
    columns: list                    # [Column, ...]
    page_size: int = 4096            # 页字节数
    compression_ratios: dict = None  # {列名: 压缩比}，不指定则按基数自动估算


def estimate_compression_ratio(col: Column, n: int) -> float:
    """
    按基数估算压缩比。
    - 基数=1（常量列）：RLE 压到几个字节，压缩比 = n
    - 基数很小（< n/100）：字典编码，压缩比 ≈ log2(基数) / avg_bytes 的倒数
    - 基数=n（唯一值）：压不动，压缩比 ≈ 1
    """
    if col.cardinality == 1:
        return min(n, 1000)  # 常量列极限压缩
    if col.cardinality < n / 100:
        # 字典编码：存 log2(cardinality) 比特的索引，原 avg_bytes 字节
        bits_needed = max(1, math.log2(col.cardinality))
        dict_ratio = (col.avg_bytes * 8) / bits_needed
        return max(dict_ratio, 1.0)
    return 1.0  # 高基数列压不动


def columnar_scan_io(spec: ColumnarSpec, query_columns: list) -> dict:
    """
    列式扫描 IO。只读查询涉及的列，每列字节 / 压缩比 / 页大小。
    返回 {列名: IO次数, 总IO, 总字节(压缩后)}。
    """
    result = {}
    total_io = 0
    total_bytes = 0
    for col in spec.columns:
        if col.name not in query_columns:
            continue
        ratio = (spec.compression_ratios or {}).get(
            col.name, estimate_compression_ratio(col, spec.n)
        )
        col_bytes = spec.n * col.avg_bytes / ratio
        io = math.ceil(col_bytes / spec.page_size)
        result[col.name] = {
            "io": io,
            "bytes_compressed": col_bytes,
            "compression_ratio": ratio,
        }
        total_io += io
        total_bytes += col_bytes
    return {"columns": result, "total_io": total_io, "total_bytes": total_bytes}


def row_oriented_scan_io(spec: ColumnarSpec, query_columns: list) -> int:
    """
    行式扫描 IO（B+树/行存）。读全部列（即使只查几列），整行字节 / 页大小。
    """
    row_bytes = sum(col.avg_bytes for col in spec.columns)
    total_bytes = spec.n * row_bytes
    return math.ceil(total_bytes / spec.page_size)


def scan_latency(io: int, page_size: int, bandwidth_mbps: float) -> float:
    """
    扫描延迟（秒）。顺序读带宽单位 MB/s。
    延迟 = IO × 页大小 / 带宽
    """
    bytes_read = io * page_size
    return bytes_read / (bandwidth_mbps * 1024 * 1024)


def _selftest():
    """公式自洽性检查"""
    cols = [
        Column("id", 8, 1_000_000),        # 高基数列（主键）
        Column("age", 4, 100),              # 低基数列（0-100）
        Column("gender", 1, 2),             # 极低基数（男/女）
        Column("name", 20, 500_000),        # 高基数字符串
    ]
    spec = ColumnarSpec(n=1_000_000, columns=cols)

    # 列式扫描：只读 age 和 gender，应该比行式快很多
    q_cols = ["age", "gender"]
    col_result = columnar_scan_io(spec, q_cols)
    row_io = row_oriented_scan_io(spec, q_cols)

    assert col_result["total_io"] < row_io, "列式扫描 IO 应远小于行式"

    # 常量列压缩比应极高（>10），性别列（基数2）约8倍
    gender_ratio = col_result["columns"]["gender"]["compression_ratio"]
    assert gender_ratio > 5, f"性别列压缩比应 >5, 实际 {gender_ratio}"

    # 高基数列压缩比应接近 1
    id_ratio = estimate_compression_ratio(cols[0], spec.n)
    assert id_ratio < 1.5, f"主键列压缩比应接近 1, 实际 {id_ratio}"

    # 读全部列时，列式 IO 应接近行式（都读全部数据）
    all_cols_result = columnar_scan_io(spec, [c.name for c in cols])
    assert all_cols_result["total_io"] >= row_io * 0.8, \
        "读全部列时列式不应比行式差太多（压缩优势）"

    print("selftest 全部通过")
    # 演示
    print(f"\n行式 vs 列式扫描 IO 对比（n={spec.n:,}）:")
    row_all = row_oriented_scan_io(spec, [c.name for c in cols])
    print(f"  行式读全部列: {row_all} IO")
    for qc in [["age", "gender"], ["id"], ["name"], [c.name for c in cols]]:
        r = columnar_scan_io(spec, qc)
        speedup = row_all / r["total_io"] if r["total_io"] > 0 else 0
        print(f"  列式读 {str(qc):30s}: {r['total_io']:>5} IO  "
              f"({r['total_bytes']/1024/1024:.1f} MiB, 加速 {speedup:.1f}x)")


if __name__ == "__main__":
    _selftest()
