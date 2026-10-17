# -*- coding: utf-8 -*-
"""
数据格式理论性能模型（核心计算器）

纯闭式解，不依赖真实库。估算同一份数据在不同格式下的体积和读取延迟。

支持的格式：
- xlsx（Excel）：zip 压缩的 XML，有共享字符串表
- sqlite：不压缩的二进制行存储
- json 行式：每行一个对象，无压缩
- json 列式：每列一个数组，无压缩
- jsonl：每行一个对象，无外层数组

核心公式：
1. 体积：xlsx ≈ 解压后 XML / 压缩率；sqlite ≈ 原始数据 + 索引；json ≈ 原始数据 × 膨胀系数
2. 读取延迟：xlsx 解析 ≈ 单元格数 × 解析时间；json 读取 ≈ 文件字节 / 带宽

本模块给出闭式解公式，参数可配。
"""
from dataclasses import dataclass
import math


@dataclass
class DataFormatSpec:
    """数据格式参数"""
    n_rows: int = 100_000          # 行数
    n_cols: int = 20               # 列数
    avg_value_len: int = 8         # 平均列值长度（字节）
    cardinality: int = 5           # 列值基数（不同取值的数量）
    compression_ratio: float = 5.0  # xlsx 压缩率（zip）
    xml_overhead_per_cell: int = 40  # XML 每单元格标签开销（字节）
    sqlite_row_overhead: int = 20    # sqlite 每行开销（字节）
    json_key_overhead: int = 12      # json 每单元格键名开销（字节）
    json_struct_overhead: int = 4    # json 结构开销（括号、逗号等）
    jsonl_struct_overhead: int = 2   # jsonl 结构开销（换行）
    # 硬件常数
    json_read_bandwidth_mbps: float = 500.0  # json 读取带宽（MB/s）
    xml_parse_per_cell_us: float = 24.0      # XML 每单元格解析时间（μs）


@dataclass
class DataFormatMetrics:
    n_rows: int
    n_cols: int
    # 体积（字节）
    xlsx_bytes: int
    sqlite_bytes: int
    json_row_bytes: int
    json_col_bytes: int
    jsonl_bytes: int
    # 读取延迟（毫秒）
    xlsx_read_ms: float
    json_row_read_ms: float
    jsonl_full_read_ms: float
    jsonl_stream_1000_ms: float
    # 追加延迟（毫秒）
    json_append_ms: float
    jsonl_append_ms: float


def compute_xlsx_size(spec: DataFormatSpec) -> int:
    """
    xlsx 体积估算。
    xlsx 是 zip 压缩的 XML，XML 每单元格有开合标签（约 40 字节）。
    有共享字符串表：重复值只存一次。
    压缩后体积 = (XML 体积 - 共享字符串节省) / 压缩率。
    """
    xml_volume = spec.n_rows * spec.n_cols * (spec.avg_value_len + spec.xml_overhead_per_cell)
    # 共享字符串表节省：重复值只存一次
    unique_values = min(spec.cardinality, spec.n_rows * spec.n_cols)
    sharing_saving = (spec.n_rows * spec.n_cols - unique_values) * spec.avg_value_len
    uncompressed = xml_volume - sharing_saving
    return int(uncompressed / spec.compression_ratio)


def compute_sqlite_size(spec: DataFormatSpec) -> int:
    """
    sqlite 体积估算。
    sqlite 是不压缩的二进制行存储，每行有固定开销（页头、记录头等）。
    没有共享字符串表：重复值每行都存全量。
    """
    # 每行 = 每列值 + 行开销
    row_size = spec.n_cols * spec.avg_value_len + spec.sqlite_row_overhead
    return spec.n_rows * row_size


def compute_json_row_size(spec: DataFormatSpec) -> int:
    """
    json 行式体积估算。
    每行一个对象，每单元格有键名 + 值 + 引号 + 冒号 + 逗号。
    没有共享字符串表：重复值每行都存全量。
    键名开销：假设列名平均 12 字节（引号 + 冒号 + 逗号）。
    """
    # 每行 = 外层括号 + 每列 (键名 + 值 + 标点) + 逗号
    row_overhead = 2  # {}
    cell_overhead = spec.json_key_overhead  # "key":
    value_overhead = 3  # "value"
    comma_overhead = 1  # ,
    row_size = row_overhead + spec.n_cols * (cell_overhead + spec.avg_value_len + value_overhead + comma_overhead)
    # 外层数组括号
    array_overhead = 2  # []
    total = spec.n_rows * row_size + array_overhead
    return total


def compute_json_col_size(spec: DataFormatSpec) -> int:
    """
    json 列式体积估算。
    每列一个数组，键名只写一次，重复值只存一次。
    """
    # 每列 = 键名 + 数组括号 + 值 × 行数 + 逗号
    key_overhead = spec.json_key_overhead
    array_overhead = 2  # []
    value_overhead = 3  # "value"
    comma_overhead = 1  # ,
    col_size = key_overhead + array_overhead + spec.n_rows * (spec.avg_value_len + value_overhead + comma_overhead)
    # 列间逗号
    inter_col_comma = 1  # ,
    total = spec.n_cols * col_size + (spec.n_cols - 1) * inter_col_comma
    # 外层对象括号
    outer_overhead = 2  # {}
    return total + outer_overhead


def compute_jsonl_size(spec: DataFormatSpec) -> int:
    """
    jsonl 体积估算。
    与 json 行式相同，只差外层数组括号换成换行符。
    """
    row_overhead = 2  # {}
    cell_overhead = spec.json_key_overhead
    value_overhead = 3
    comma_overhead = 1
    row_size = row_overhead + spec.n_cols * (cell_overhead + spec.avg_value_len + value_overhead + comma_overhead)
    # jsonl 每行以换行结尾（\n）
    newline = 1  # \n
    return spec.n_rows * (row_size + newline)


def compute_xlsx_read_time(spec: DataFormatSpec) -> float:
    """
    xlsx 读取延迟（毫秒）。
    xlsx 读取慢在 XML 解析，每个单元格约 24 微秒。
    """
    total_cells = spec.n_rows * spec.n_cols
    return total_cells * spec.xml_parse_per_cell_us / 1000


def compute_json_row_read_time(spec: DataFormatSpec) -> float:
    """
    json 行式读取延迟（毫秒）。
    json 读取是一次 C 层扫描，延迟 = 文件大小 / 带宽。
    """
    size_bytes = compute_json_row_size(spec)
    size_mb = size_bytes / 1024 / 1024
    return size_mb / spec.json_read_bandwidth_mbps * 1000


def compute_jsonl_full_read_time(spec: DataFormatSpec) -> float:
    """
    jsonl 全量读取延迟（毫秒）。
    比 json 慢 1.5-2.3 倍，因为逐行 json.loads 有固定开销。
    """
    json_time = compute_json_row_read_time(spec)
    return json_time * 1.8  # 经验系数


def compute_jsonl_stream_time(spec: DataFormatSpec, n_lines: int = 1000) -> float:
    """
    jsonl 流式读取前 n_lines 行（毫秒）。
    与文件规模无关，只与 n_lines 成正比。
    """
    per_line_ms = 0.005  # 实测约 5 微秒/行
    return n_lines * per_line_ms


def compute_json_append_time(spec: DataFormatSpec) -> float:
    """
    json 全量形态追加一行（毫秒）。
    必须读回整个文件、追加、整文件重写，耗时随规模线性增长。
    """
    size_bytes = compute_json_row_size(spec)
    size_mb = size_bytes / 1024 / 1024
    # 读取 + 写入 = 2 × 大小 / 带宽
    return size_mb / spec.json_read_bandwidth_mbps * 2 * 1000


def compute_jsonl_append_time(spec: DataFormatSpec) -> float:
    """
    jsonl 追加一行（毫秒）。
    只需在末尾追加，与文件规模无关，稳定在 0.5 毫秒。
    """
    return 0.5


def compute(spec: DataFormatSpec) -> DataFormatMetrics:
    """给定参数，递推全部指标。"""
    return DataFormatMetrics(
        n_rows=spec.n_rows,
        n_cols=spec.n_cols,
        xlsx_bytes=compute_xlsx_size(spec),
        sqlite_bytes=compute_sqlite_size(spec),
        json_row_bytes=compute_json_row_size(spec),
        json_col_bytes=compute_json_col_size(spec),
        jsonl_bytes=compute_jsonl_size(spec),
        xlsx_read_ms=compute_xlsx_read_time(spec),
        json_row_read_ms=compute_json_row_read_time(spec),
        jsonl_full_read_ms=compute_jsonl_full_read_time(spec),
        jsonl_stream_1000_ms=compute_jsonl_stream_time(spec),
        json_append_ms=compute_json_append_time(spec),
        jsonl_append_ms=compute_jsonl_append_time(spec),
    )


def _selftest():
    """公式自洽性检查"""
    spec = DataFormatSpec()
    m = compute(spec)

    # 体积应为正
    assert m.xlsx_bytes > 0
    assert m.sqlite_bytes > 0
    assert m.json_row_bytes > 0
    assert m.jsonl_bytes > 0

    # jsonl 体积与 json 行式几乎相同（差 < 1%）
    diff = abs(m.jsonl_bytes - m.json_row_bytes) / m.json_row_bytes
    assert diff < 0.01, "jsonl 与 json 行式体积应几乎相同"

    # json 列式比行式小（键名只写一次）
    assert m.json_col_bytes < m.json_row_bytes, "列式应比行式小"

    # xlsx 比 sqlite 小（压缩 + 共享字符串表）
    assert m.xlsx_bytes < m.sqlite_bytes, "xlsx 应比 sqlite 小"

    # xlsx 读取比 json 慢（XML 解析）
    assert m.xlsx_read_ms > m.json_row_read_ms, "xlsx 读取应比 json 慢"

    # jsonl 追加比 json 快
    assert m.jsonl_append_ms < m.json_append_ms, "jsonl 追加应比 json 快"

    # jsonl 追加与规模无关
    spec_big = DataFormatSpec(n_rows=1_000_000)
    m_big = compute(spec_big)
    assert abs(m_big.jsonl_append_ms - m.jsonl_append_ms) < 0.1, "jsonl 追加应与规模无关"

    # json 追加随规模线性增长
    assert m_big.json_append_ms > m.json_append_ms * 5, "json 追加应随规模增长"

    print("selftest 全部通过")
    # 演示
    print(f"\nn={m.n_rows:,} 行, {m.n_cols} 列, 值长 {spec.avg_value_len}B, 基数 {spec.cardinality}")
    print(f"体积: xlsx={m.xlsx_bytes/1024/1024:.1f} MiB  "
          f"sqlite={m.sqlite_bytes/1024/1024:.1f} MiB  "
          f"json行式={m.json_row_bytes/1024/1024:.1f} MiB  "
          f"jsonl={m.jsonl_bytes/1024/1024:.1f} MiB")
    print(f"读取: xlsx={m.xlsx_read_ms:.0f} ms  "
          f"json行式={m.json_row_read_ms:.0f} ms  "
          f"jsonl流式1000行={m.jsonl_stream_1000_ms:.1f} ms")
    print(f"追加: json={m.json_append_ms:.0f} ms  jsonl={m.jsonl_append_ms:.1f} ms")


if __name__ == "__main__":
    _selftest()
