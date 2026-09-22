# -*- coding: utf-8 -*-
"""
Delta Encoding 实测验证

验证四件事：
1. 压缩比：四种模式（等差升序 / 有序均匀 / 无序均匀 / 大量重复）
2. 与直接 Varint（不差分）和定长存储的对比
3. Delta + Varint vs Delta + Zigzag + Varint（负差值能否处理）
4. 编解码速度（ops/sec）

结果写入 results/delta_<timestamp>.json
"""
import os
import sys
import json
import time
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from varint_model import encode_varint
from delta_model import (
    delta_encode, delta_decode, zigzag_encode, zigzag_decode,
    encode_delta_varint, decode_delta_varint,
    encode_delta_zigzag_varint, decode_delta_zigzag_varint,
    ratio_ascending_arith, ratio_sorted_uniform,
    ratio_heavy_duplicates, ratio_unsorted_uniform,
)

N = 1_000_000
SEED = 20260907
ITEMSIZE = 8  # 数组为 int64，定长基准取 uint64


def sample_asc_arith(n: int, step: int = 100) -> np.ndarray:
    """等差升序。差值恒为 step，全部落在 varint 的 1 字节组。"""
    return (np.arange(n, dtype=np.int64) * step)


def sample_asc_random(n: int, value_range: int = 2 ** 31) -> np.ndarray:
    """均匀撒点后排序（有序均匀）。差值近似指数分布，小且为正。"""
    rng = np.random.default_rng(SEED)
    vals = rng.integers(0, value_range, size=n, dtype=np.int64)
    vals.sort()
    return vals


def sample_random_unsorted(n: int, value_range: int = 2 ** 32) -> np.ndarray:
    """均匀随机不排序（无序均匀）。差值大且正负交替，差分失效的边界。"""
    return np.random.randint(0, value_range, size=n, dtype=np.int64)


def sample_heavy_dups(n: int, n_unique: int = 1000) -> np.ndarray:
    """大量重复。只有 n_unique 个不同值，其余全是重复。"""
    rng = np.random.default_rng(SEED)
    pool = np.sort(rng.integers(0, 2**31, size=n_unique, dtype=np.int64))
    # 每个唯一值连片重复若干次，重复段之间差值为 0
    run_lens = rng.integers(1, max(2, n // n_unique * 2), size=n_unique)
    out = np.repeat(pool, run_lens)
    if len(out) >= n:
        return out[:n]
    return np.pad(out, (0, n - len(out)), mode="edge")


def sample_desc_with_neg(n: int, step: int = 50) -> np.ndarray:
    """递减序列（差值为负），用于验证 zigzag 路径的必要性。"""
    return np.arange(n * step, 0, -step, dtype=np.int64)


DISTRIBUTIONS = {
    # name: (采样函数, itemsize, 定长基准, 闭式压缩比函数)
    # 数组是 int64，定长基准取 uint64（8 字节），否则实测与理论不同口径
    "asc_arith": (sample_asc_arith, 8, "uint64",
                  lambda n: ratio_ascending_arith(8, 100)),
    "asc_random": (sample_asc_random, 8, "uint64",
                   lambda n: ratio_sorted_uniform(8, n, 2 ** 31)),
    "random_unsorted": (sample_random_unsorted, 8, "uint64",
                        lambda n: ratio_unsorted_uniform(8, 2 ** 32)),
    "heavy_dups": (sample_heavy_dups, 8, "uint64",
                   lambda n: ratio_heavy_duplicates(8, n)),
}


def measure(name: str, values: np.ndarray, itemsize: int,
            theo_ratio: float) -> dict:
    """
    实测一种模式的四种编码体积与压缩比。
    - raw: 定长存储
    - plain_varint: 直接 Varint（不差分）
    - delta_varint: 差分 + Varint（要求非负差值）
    - delta_zigzag_varint: 差分 + Zigzag + Varint（可处理负差值）
    """
    n = len(values)
    raw_bytes_actual = values.nbytes
    arr = values.tolist()

    # 直接 Varint（不差分）
    plain = b"".join(encode_varint(int(v)) for v in arr)
    plain_bytes = len(plain)

    # 差分 + Varint（仅当全部差值非负时可用）
    deltas = delta_encode(arr)
    if any(d < 0 for d in deltas):
        dv_bytes = None  # 有负差值，纯 Varint 路径不可用
    else:
        dv_bytes = len(encode_delta_varint(arr))

    # 差分 + Zigzag + Varint（任何差值都可用）
    dz = encode_delta_zigzag_varint(arr)
    dz_bytes = len(dz)

    # 往返可逆性
    decoded = decode_delta_zigzag_varint(dz, n)
    assert decoded == arr, f"{name} 差分+zigzag+varint 往返不一致"

    result = {
        "distribution": name,
        "n": n,
        "itemsize": itemsize,
        "max_value": int(values.max()),
        "min_delta": int(np.min(np.diff(values))) if n > 1 else 0,
        "max_delta": int(np.max(np.diff(values))) if n > 1 else 0,
        "raw_bytes": raw_bytes_actual,
        "plain_varint_bytes": plain_bytes,
        "delta_varint_bytes": dv_bytes,
        "delta_zigzag_varint_bytes": dz_bytes,
        "theo_ratio": theo_ratio,
        "roundtrip_ok": True,
    }
    for key in ["plain_varint", "delta_varint", "delta_zigzag_varint"]:
        b = result[f"{key}_bytes"]
        if b:
            result[f"{key}_ratio"] = raw_bytes_actual / b
    return result


def verify_roundtrip_modes() -> dict:
    """四种模式各自的往返一致性，含负差值。"""
    checks = {}
    cases = {
        "asc_arith": sample_asc_arith(1000, 7).tolist(),
        "asc_random": sample_asc_random(1000).tolist(),
        "random_unsorted": sample_random_unsorted(1000).tolist(),
        "heavy_dups": sample_heavy_dups(1000).tolist(),
        "desc_neg": sample_desc_with_neg(1000, 13).tolist(),
    }
    for name, arr in cases.items():
        # 差分往返
        assert delta_decode(delta_encode(arr)) == arr
        # zigzag 往返
        assert [zigzag_decode(zigzag_encode(d)) for d in arr] == arr
        # 差分 + zigzag + varint 往返
        encoded = encode_delta_zigzag_varint(arr)
        assert decode_delta_zigzag_varint(encoded, len(arr)) == arr
        checks[name] = {"n": len(arr), "roundtrip_ok": True}
    return checks


def measure_speed(n: int = 200_000) -> dict:
    """差分与 zigzag 的编码/解码速度（百万 ops/sec）。"""
    arr = sample_asc_random(n).tolist()

    # 差分编码
    start = time.perf_counter()
    deltas = delta_encode(arr)
    enc_delta_s = time.perf_counter() - start

    # zigzag 编码
    start = time.perf_counter()
    zz = [zigzag_encode(d) for d in deltas]
    enc_zz_s = time.perf_counter() - start

    # varint 编码（差分 + zigzag + varint 的完整编码路径）
    start = time.perf_counter()
    encoded = encode_delta_zigzag_varint(arr)
    enc_full_s = time.perf_counter() - start

    # 完整解码
    start = time.perf_counter()
    decoded = decode_delta_zigzag_varint(encoded, n)
    dec_full_s = time.perf_counter() - start

    assert decoded == arr, "速度测试的往返不一致"

    return {
        "n": n,
        "delta_encode_ops_per_sec": n / enc_delta_s,
        "zigzag_encode_ops_per_sec": n / enc_zz_s,
        "delta_zigzag_varint_encode_ops_per_sec": n / enc_full_s,
        "delta_zigzag_varint_decode_ops_per_sec": n / dec_full_s,
        "encode_full_ms": enc_full_s * 1000,
        "decode_full_ms": dec_full_s * 1000,
    }


def main():
    print("=== Delta Encoding 实测 ===")

    # 1. 往返一致性
    rt = verify_roundtrip_modes()
    print(f"往返一致性: {len(rt)} 种模式全部通过 "
          f"（含递减序列的负差值）")

    # 2. 四种模式的压缩比
    print(f"\n=== 压缩比实测（n={N:,}, 定长基准 uint64）===")
    results = []
    for name, (sampler, itemsize, baseline, theo_fn) in DISTRIBUTIONS.items():
        values = sampler(N)
        theo = theo_fn(N)
        r = measure(name, values, itemsize, theo)
        results.append(r)

        dv = r["delta_varint_bytes"]
        dv_str = f"{dv/1024/1024:>5.2f} MiB" if dv else "   N/A(负差值)"
        dv_r = f"{r.get('delta_varint_ratio', 0):>5.2f}x" if dv else "   N/A"
        print(f"{name:16s} delta范围=[{r['min_delta']:>12,},{r['max_delta']:>12,}]")
        print(f"{'':16s} 定长={r['raw_bytes']/1024/1024:>5.2f} MiB  "
              f"直Varint={r['plain_varint_bytes']/1024/1024:>5.2f} MiB "
              f"({r['plain_varint_ratio']:>5.2f}x)")
        print(f"{'':16s} D+Varint={dv_str} ({dv_r})  "
              f"D+Zig+Varint={r['delta_zigzag_varint_bytes']/1024/1024:>5.2f} MiB "
              f"({r['delta_zigzag_varint_ratio']:>5.2f}x)  "
              f"理论={theo:.2f}x")

    # 3. 速度
    print("\n=== 编解码速度 ===")
    sp = measure_speed()
    print(f"差分编码:            {sp['delta_encode_ops_per_sec']/1e6:.2f} M ops/s")
    print(f"zigzag 编码:         {sp['zigzag_encode_ops_per_sec']/1e6:.2f} M ops/s")
    print(f"D+Zigzag+Varint 编码:{sp['delta_zigzag_varint_encode_ops_per_sec']/1e6:.2f} M ops/s")
    print(f"D+Zigzag+Varint 解码:{sp['delta_zigzag_varint_decode_ops_per_sec']/1e6:.2f} M ops/s")

    out_data = {
        "meta": {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "seed": SEED,
            "n": N,
            "itemsize": ITEMSIZE,
            "note": "纯 Python 实现，速度含解释器开销",
        },
        "roundtrip": rt,
        "distributions": results,
        "speed": sp,
    }
    out = os.path.join(os.path.dirname(__file__), "..", "results",
                       f"delta_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)
    print(f"\n结果写入 {out}")


if __name__ == "__main__":
    main()
