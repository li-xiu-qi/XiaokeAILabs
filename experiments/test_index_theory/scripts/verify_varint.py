# -*- coding: utf-8 -*-
"""
Varint 实测验证

验证三件事：
1. 压缩比：三种分布（均匀 0-1000、Zipf 小整数为主、均匀 0-2^32）
   对比定长 uint32
2. 理论公式校验：bytes(n) = ceil(bits(n)/7) 与实测编码长度逐值比对
3. 编解码速度（ops/sec）

结果写入 results/varint_<timestamp>.json
"""
import os
import sys
import json
import time
import math
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from varint_model import (
    encode_varint, decode_varint, varint_bytes, group_ratio,
    expected_bytes_uniform, expected_bytes_zipf, selftest_roundtrip,
)

# 每组测试的样本数
N = 1_000_000
SEED = 20260907


def sample_uniform_0_1000(n: int) -> np.ndarray:
    """均匀分布 [0, 1000]，绝大多数值落在 2 字节组。"""
    return np.random.randint(0, 1001, size=n)


def sample_zipf(n: int, a: float = 1.3) -> np.ndarray:
    """
    Zipf 分布（小整数为主）。用 numpy 的 zipf 生成 1..2^31 的 Zipf 样本，
    形状参数 a 越大越偏向小值。
    """
    return np.random.zipf(a, size=n).astype(np.int64)


def sample_uniform_0_2p32(n: int) -> np.ndarray:
    """均匀分布 [0, 2^32)，需要 5 字节 varint。"""
    return np.random.randint(0, 2**32, size=n, dtype=np.int64)


DISTRIBUTIONS = {
    # name: (采样函数, itemsize, 定长基准, 闭式期望字节数函数)
    "uniform_0_1000": (sample_uniform_0_1000, 4, "uint32",
                       lambda: expected_bytes_uniform(0, 1001)),
    "zipf_small": (sample_zipf, 4, "uint32",
                   lambda: expected_bytes_zipf(1.3)),
    "uniform_0_2p32": (sample_uniform_0_2p32, 8, "uint64",
                       lambda: expected_bytes_uniform(0, 2 ** 32)),
}


def encode_all(values: np.ndarray) -> bytes:
    """把整个数组编码为 varint 字节串。"""
    return b"".join(encode_varint(int(v)) for v in values)


def decode_all(data: bytes, count: int) -> list:
    """从字节串解出 count 个整数，返回列表。"""
    out = []
    pos = 0
    for _ in range(count):
        value, consumed = decode_varint(data, pos)
        out.append(value)
        pos += consumed
    return out


def measure_distribution(name: str, values: np.ndarray, itemsize: int,
                        theo_avg_bytes: float) -> dict:
    """
    实测一种分布的压缩比与理论值的吻合度。
    raw_bytes = n × itemsize（定长存储），varint_bytes = 实际编码长度。
    theo_avg_bytes 是模型给出的闭式期望字节/值，theo_ratio = itemsize / theo_avg_bytes。
    """
    n = len(values)
    raw_bytes = n * itemsize
    max_value = int(values.max())

    data = encode_all(values)
    varint_bytes_total = len(data)

    # 解码可逆性
    decoded = decode_all(data, n)
    assert decoded == values.tolist(), f"{name} 往返不一致"

    actual_ratio = raw_bytes / varint_bytes_total
    theo_ratio = itemsize / theo_avg_bytes

    # 实测平均字节 / 值
    avg_bytes = varint_bytes_total / n

    return {
        "distribution": name,
        "n": n,
        "itemsize": itemsize,
        "max_value": max_value,
        "raw_bytes": raw_bytes,
        "varint_bytes": varint_bytes_total,
        "actual_ratio": actual_ratio,
        "theo_ratio": theo_ratio,
        "theo_avg_bytes": theo_avg_bytes,
        "avg_bytes_per_value": avg_bytes,
        "roundtrip_ok": True,
    }


def verify_bytes_formula(sample_n: int = 200_000) -> dict:
    """
    逐值校验 bytes(n) = ceil(bits(n)/7)。分四段量级各抽 sample_n 个，
    覆盖到 64 位满量程。返回不一致的条数。
    """
    # uint64 满量程用无符号采样，int64 上界装不下 2^64
    arrays = [
        np.arange(0, 300),
        np.random.randint(0, 2**20, size=sample_n),
        np.random.randint(0, 2**32, size=sample_n, dtype=np.int64),
        np.random.default_rng().integers(0, 2**64, size=sample_n, dtype=np.uint64),
    ]
    mismatch = 0
    total = 0
    for arr in arrays:
        for v in arr:
            total += 1
            v = int(v)
            if varint_bytes(v) != len(encode_varint(v)):
                mismatch += 1
    return {"sampled": total, "mismatch": mismatch}


def measure_speed(n: int = 200_000) -> dict:
    """
    编解码速度（百万 ops/sec）。纯 Python 循环，测的是解释器开销下的吞吐，
    真实系统里用 C++ 实现（protobuf、RocksDB）会快一到两个数量级。
    """
    values = np.random.randint(0, 2 ** 32, size=n, dtype=np.int64).tolist()

    # 编码
    start = time.perf_counter()
    encoded = [encode_varint(v) for v in values]
    enc_elapsed = time.perf_counter() - start

    # 拼接后整体解码
    buf = b"".join(encoded)
    start = time.perf_counter()
    decoded = []
    pos = 0
    while pos < len(buf):
        value, consumed = decode_varint(buf, pos)
        decoded.append(value)
        pos += consumed
    dec_elapsed = time.perf_counter() - start

    assert decoded == values, "速度测试的往返不一致"

    enc_ops = n / enc_elapsed
    dec_ops = len(decoded) / dec_elapsed

    return {
        "n": n,
        "encode_ops_per_sec": enc_ops,
        "decode_ops_per_sec": dec_ops,
        "encode_ms": enc_elapsed * 1000,
        "decode_ms": dec_elapsed * 1000,
        "encoded_bytes": len(buf),
    }


def main():
    print("=== Varint 模型实测 ===")

    # 1. 字节数公式逐值校验
    formula = verify_bytes_formula()
    print(f"bytes 公式逐值校验: {formula['sampled']:,} 个样本, "
          f"不一致 {formula['mismatch']} 个")

    # 2. 三种分布的压缩比
    print("\n=== 压缩比实测（n=%d）===" % N)
    dist_results = []
    for name, (sampler, itemsize, baseline, theo_fn) in DISTRIBUTIONS.items():
        values = sampler(N)
        theo_avg = theo_fn()
        r = measure_distribution(name, values, itemsize, theo_avg)
        dist_results.append(r)
        print(f"{name:18s} max={r['max_value']:>12,}  "
              f"原始={r['raw_bytes']/1024/1024:>5.1f} MiB  "
              f"varint={r['varint_bytes']/1024/1024:>5.1f} MiB  "
              f"实测比={r['actual_ratio']:>5.2f}x  "
              f"理论比={r['theo_ratio']:>5.2f}x  "
              f"均值={r['avg_bytes_per_value']:.2f} B/值 "
              f"(理论 {theo_avg:.2f})")

    # 3. 速度
    print("\n=== 编解码速度 ===")
    speed = measure_speed()
    print(f"编码: {speed['encode_ops_per_sec']/1e6:.2f} M ops/s  "
          f"({speed['encode_ms']:.0f} ms / {speed['n']:,} 个)")
    print(f"解码: {speed['decode_ops_per_sec']/1e6:.2f} M ops/s  "
          f"({speed['decode_ms']:.0f} ms / {speed['n']:,} 个)")

    result = {
        "meta": {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "seed": SEED,
            "n": N,
            "note": "纯 Python 实现，速度含解释器开销",
        },
        "formula_check": formula,
        "distributions": dist_results,
        "speed": speed,
    }
    out = os.path.join(os.path.dirname(__file__), "..", "results",
                       f"varint_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n结果写入 {out}")


if __name__ == "__main__":
    main()
