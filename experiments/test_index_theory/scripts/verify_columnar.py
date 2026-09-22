# -*- coding: utf-8 -*-
"""
列式存储实测验证

验证两件事：
1. 压缩比：生成不同基数的列，用 numpy 存为二进制，对比未压缩 vs 简单压缩（字典编码）
2. 扫描延迟：顺序读大文件的带宽，用 mmap 或 read() 测

结果写入 results/columnar_<timestamp>.json
"""
import os
import sys
import json
import time
import tempfile
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from columnar_model import Column, ColumnarSpec, columnar_scan_io, row_oriented_scan_io, estimate_compression_ratio


def measure_compression(n: int, cardinality: int, dtype: np.dtype) -> dict:
    """
    实测压缩比。生成 n 个值（基数=cardinality），存为二进制，对比：
    - 原始大小：n × dtype.itemsize
    - 字典编码后：字典（cardinality × itemsize）+ 索引（n × log2(cardinality)/8 字节）
    """
    dt = np.dtype(dtype)
    item_size = dt.itemsize
    values = np.random.randint(0, cardinality, size=n).astype(dt)
    raw_bytes = values.nbytes

    # 字典编码模拟：去重后的字典 + 每行的索引
    unique = np.unique(values)
    dict_bytes = len(unique) * item_size
    bits_per_index = max(1, int(np.ceil(np.log2(len(unique)))))
    index_bytes = n * bits_per_index / 8
    compressed_bytes = dict_bytes + index_bytes

    ratio = raw_bytes / compressed_bytes if compressed_bytes > 0 else 1.0
    return {
        "n": n,
        "cardinality": cardinality,
        "raw_bytes": raw_bytes,
        "compressed_bytes": int(compressed_bytes),
        "actual_ratio": ratio,
        "theo_ratio": estimate_compression_ratio(
            Column("test", item_size, cardinality), n
        ),
    }


def measure_scan_bandwidth(file_size_mb: int = 100) -> dict:
    """
    实测顺序读带宽。生成 file_size_mb 的临时文件，顺序读一遍，测带宽。
    """
    fd, path = tempfile.mkstemp(suffix=".bin")
    os.close(fd)
    try:
        size = file_size_mb * 1024 * 1024
        # 写文件
        data = np.random.bytes(size)  # 生成 size 字节的随机数据
        with open(path, "wb") as f:
            f.write(data)

        # 顺序读，测带宽
        chunk_size = 1024 * 1024  # 1 MiB
        start = time.perf_counter()
        bytes_read = 0
        with open(path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                bytes_read += len(chunk)
        elapsed = time.perf_counter() - start
        bandwidth_mbps = (bytes_read / 1024 / 1024) / elapsed

        return {
            "file_size_mb": file_size_mb,
            "bytes_read": bytes_read,
            "elapsed_s": elapsed,
            "bandwidth_mbps": bandwidth_mbps,
        }
    finally:
        if os.path.exists(path):
            os.remove(path)


def main():
    print("=== 压缩比实测 ===")
    comp_results = []
    for card in [2, 10, 100, 1000, 100_000, 1_000_000]:
        r = measure_compression(n=1_000_000, cardinality=card, dtype=np.int32)
        comp_results.append(r)
        print(f"基数={card:>8,}  原始={r['raw_bytes']/1024/1024:.1f} MiB  "
              f"压缩后={r['compressed_bytes']/1024/1024:.1f} MiB  "
              f"实测比={r['actual_ratio']:.1f}x  "
              f"理论比={r['theo_ratio']:.1f}x")

    print("\n=== 顺序读带宽实测 ===")
    bw_results = []
    for sz in [10, 100, 500]:
        r = measure_scan_bandwidth(sz)
        bw_results.append(r)
        print(f"{sz} MiB 文件: {r['bandwidth_mbps']:.0f} MB/s  "
              f"({r['elapsed_s']:.3f}s)")

    # 用平均带宽做扫描延迟换算表
    avg_bw = np.mean([r["bandwidth_mbps"] for r in bw_results])
    print(f"\n平均顺序读带宽: {avg_bw:.0f} MB/s")

    cols = [
        Column("id", 8, 1_000_000),
        Column("age", 4, 100),
        Column("gender", 1, 2),
        Column("name", 20, 500_000),
    ]
    spec = ColumnarSpec(n=1_000_000, columns=cols)
    row_io = row_oriented_scan_io(spec, [c.name for c in cols])

    print("\n=== 扫描延迟换算（1M 行，带宽 %.0f MB/s）===" % avg_bw)
    latency_table = []
    for qc in [["age", "gender"], ["id"], ["name"], [c.name for c in cols]]:
        r = columnar_scan_io(spec, qc)
        latency_ms = r["total_bytes"] / (avg_bw * 1024 * 1024) * 1000
        latency_table.append({
            "query_columns": qc,
            "io": r["total_io"],
            "bytes_mb": r["total_bytes"] / 1024 / 1024,
            "latency_ms": latency_ms,
        })
        print(f"列式读 {str(qc):35s}: {r['total_io']:>5} IO  "
              f"{r['total_bytes']/1024/1024:>6.1f} MiB  "
              f"延迟 {latency_ms:>6.2f} ms")
    row_latency_ms = (row_io * 4096) / (avg_bw * 1024 * 1024) * 1000
    print(f"行式读全部列: {row_io} IO  "
          f"{row_io*4096/1024/1024:.1f} MiB  延迟 {row_latency_ms:.2f} ms")

    result = {
        "meta": {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "note": "压缩比用字典编码模拟，扫描带宽为本机实测",
        },
        "compression": comp_results,
        "bandwidth": bw_results,
        "avg_bandwidth_mbps": float(avg_bw),
        "latency_table": latency_table,
        "row_oriented": {
            "io": row_io,
            "latency_ms": row_latency_ms,
        },
    }
    out = os.path.join(os.path.dirname(__file__), "..", "results",
                       f"columnar_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n结果写入 {out}")


if __name__ == "__main__":
    main()
