# -*- coding: utf-8 -*-
"""
数据格式实测验证

用自制编解码器验证理论模型的体积公式：
1. 生成模拟数据，实际编码为 xlsx/sqlite/json/jsonl
2. 对比实测字节数与理论公式
3. 验证 jsonl 体积 ≈ json 行式体积
4. 验证 json 列式 < json 行式

方法：不依赖 openpyxl/pandas，用纯 Python 模拟各格式的编码逻辑。
结果写入 results/data_format_<timestamp>.json
"""
import os
import sys
import json
import math
import random
import string
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from data_format_model import (
    DataFormatSpec, compute, compute_xlsx_size, compute_sqlite_size,
    compute_json_row_size, compute_json_col_size, compute_jsonl_size
)


def generate_data(n_rows: int, n_cols: int, cardinality: int, value_len: int) -> list:
    """生成模拟数据。返回 list of dict。"""
    random.seed(42)
    # 生成取值池
    pool = [''.join(random.choices(string.ascii_lowercase, k=value_len))
            for _ in range(cardinality)]
    data = []
    for _ in range(n_rows):
        row = {f"col{i}": random.choice(pool) for i in range(n_cols)}
        data.append(row)
    return data


def encode_json_row(data: list) -> bytes:
    """json 行式编码。"""
    return json.dumps(data).encode('utf-8')


def encode_json_col(data: list) -> bytes:
    """json 列式编码。"""
    cols = {}
    for key in data[0].keys():
        cols[key] = [row[key] for row in data]
    return json.dumps(cols).encode('utf-8')


def encode_jsonl(data: list) -> bytes:
    """jsonl 编码。"""
    lines = [json.dumps(row) for row in data]
    return '\n'.join(lines).encode('utf-8')


def simulate_xlsx(data: list, compression_ratio: float) -> int:
    """
    模拟 xlsx 体积。
    xlsx 是 zip 压缩的 XML，有共享字符串表。
    简化：XML 体积 = 每单元格 (标签 + 值)，压缩后 / 压缩率。
    """
    # XML 每单元格开销：开标签 + 闭标签 + 类型声明
    xml_overhead = 40
    # 共享字符串表：重复值只存一次
    all_values = set()
    for row in data:
        for v in row.values():
            all_values.add(v)
    sharing_saving = len(data) * len(data[0]) * 8 - len(all_values) * 8
    xml_volume = len(data) * len(data[0]) * (xml_overhead + 8) - max(0, sharing_saving)
    return int(xml_volume / compression_ratio)


def simulate_sqlite(data: list) -> int:
    """
    模拟 sqlite 体积。
    sqlite 是不压缩的二进制行存储，每行有固定开销。
    简化：每行 = 每列值 + 行开销。
    """
    row_overhead = 20
    return len(data) * (len(data[0]) * 8 + row_overhead)


def main():
    print("=== 数据格式实测验证 ===\n")

    # 测试多种场景
    scenarios = [
        {"n_rows": 10_000, "n_cols": 10, "cardinality": 5, "value_len": 8, "label": "小数据低基数"},
        {"n_rows": 10_000, "n_cols": 10, "cardinality": 5_000, "value_len": 8, "label": "小数据高基数"},
        {"n_rows": 50_000, "n_cols": 20, "cardinality": 5, "value_len": 8, "label": "中数据低基数"},
    ]

    results = []
    for sc in scenarios:
        print(f"--- {sc['label']} ---")
        print(f"  {sc['n_rows']:,} 行 × {sc['n_cols']} 列, 基数 {sc['cardinality']}, 值长 {sc['value_len']}B")

        # 生成数据
        data = generate_data(sc["n_rows"], sc["n_cols"], sc["cardinality"], sc["value_len"])

        # 实测体积
        actual_json_row = len(encode_json_row(data))
        actual_json_col = len(encode_json_col(data))
        actual_jsonl = len(encode_jsonl(data))
        actual_xlsx = simulate_xlsx(data, compression_ratio=5.0)
        actual_sqlite = simulate_sqlite(data)

        # 理论体积
        spec = DataFormatSpec(
            n_rows=sc["n_rows"],
            n_cols=sc["n_cols"],
            avg_value_len=sc["value_len"],
            cardinality=sc["cardinality"],
            compression_ratio=5.0,
        )
        theo = compute(spec)

        # 对比
        print(f"  {'格式':<12} {'实测':>12} {'理论':>12} {'比值':>8}")
        print(f"  {'json 行式':<12} {actual_json_row:>12,} {theo.json_row_bytes:>12,} "
              f"{actual_json_row/theo.json_row_bytes:>8.3f}")
        print(f"  {'json 列式':<12} {actual_json_col:>12,} {theo.json_col_bytes:>12,} "
              f"{actual_json_col/theo.json_col_bytes:>8.3f}")
        print(f"  {'jsonl':<12} {actual_jsonl:>12,} {theo.jsonl_bytes:>12,} "
              f"{actual_jsonl/theo.jsonl_bytes:>8.3f}")
        print(f"  {'xlsx(模拟)':<12} {actual_xlsx:>12,} {theo.xlsx_bytes:>12,} "
              f"{actual_xlsx/theo.xlsx_bytes:>8.3f}")
        print(f"  {'sqlite(模拟)':<12} {actual_sqlite:>12,} {theo.sqlite_bytes:>12,} "
              f"{actual_sqlite/theo.sqlite_bytes:>8.3f}")

        # 验证关系
        jsonl_vs_json = actual_jsonl / actual_json_row
        col_vs_row = actual_json_col / actual_json_row
        print(f"  jsonl/json 行式 = {jsonl_vs_json:.4f} (应 ≈ 1.0)")
        print(f"  json 列式/行式 = {col_vs_row:.4f} (应 < 1.0)")
        print()

        results.append({
            "scenario": sc["label"],
            "n_rows": sc["n_rows"],
            "n_cols": sc["n_cols"],
            "cardinality": sc["cardinality"],
            "value_len": sc["value_len"],
            "actual": {
                "json_row": actual_json_row,
                "json_col": actual_json_col,
                "jsonl": actual_jsonl,
                "xlsx": actual_xlsx,
                "sqlite": actual_sqlite,
            },
            "theoretical": {
                "json_row": theo.json_row_bytes,
                "json_col": theo.json_col_bytes,
                "jsonl": theo.jsonl_bytes,
                "xlsx": theo.xlsx_bytes,
                "sqlite": theo.sqlite_bytes,
            },
            "ratios": {
                "jsonl_vs_json": jsonl_vs_json,
                "col_vs_row": col_vs_row,
            },
        })

    # 汇总写入 JSON
    out = os.path.join(os.path.dirname(__file__), "..", "results",
                       f"data_format_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "meta": {
                "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "python": sys.version.split()[0],
                "note": "自制编码器验证体积公式，xlsx 和 sqlite 用简化模型模拟",
            },
            "scenarios": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"结果写入 {out}")


if __name__ == "__main__":
    main()
