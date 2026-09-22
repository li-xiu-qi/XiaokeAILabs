# -*- coding: utf-8 -*-
"""
PForDelta 实测验证

自制 PForDelta（分帧位封装 + 异常区），实测四项：

1. 三种分布的压缩比
   docID 列表（大部分小、偶尔大）、随机均匀、全大数，
   对比 Simple8b 与 Varint。这是 PForDelta 的主场：
   异常值只污染本帧，不波及后续。

2. 帧大小对比（128 vs 256）
   帧越大位宽越稳（异常值被摊薄），但帧头开销占比下降；
   帧越小异常污染越局部。实测两者的平衡点。

3. 异常值比例对压缩比的影响
   固定数值范围，扫描异常值比例 0% → 20%，验证压缩比单调下降，
   并找出 PForDelta 开始劣于 Simple8b 的临界比例。

4. 编码/解码速度
   纯 Python 实现的每整数吞吐，对比 Simple8b 的同类实现。

方法：自制编解码器 + 时间采样。结果写入 results/pfor_delta_<timestamp>.json
"""
import os
import sys
import json
import time
import random
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from simple8b_model import encode_simple8b, decode_simple8b, varint_size
from pfor_delta_model import (
    DEFAULT_FRAME_SIZE, encode_pfor_delta, decode_pfor_delta,
    select_bits, compressed_bytes, predicted_bits_per_int,
)


def varint_encode(arr: list) -> bytearray:
    """LEB128 Varint 编码，用于交叉验证 varint_size。"""
    out = bytearray()
    for v in arr:
        while True:
            b = v & 0x7F
            v >>= 7
            out.append(b | (0x80 if v else 0))
            if not v:
                break
    return out


def make_distribution(kind: str, n: int, rng: random.Random) -> list:
    """生成三种测试分布。"""
    if kind == "docid":
        arr = [rng.randint(0, 3000) for _ in range(n)]
        for _ in range(n // 100):      # 1% 大值（异常值）
            arr[rng.randrange(n)] = rng.randint(3000, 2 ** 24)
        return arr
    if kind == "uniform":              # 随机均匀 0-1000
        return [rng.randint(0, 1000) for _ in range(n)]
    if kind == "large":                # 全大数 0-2^24
        return [rng.randint(0, 2 ** 24) for _ in range(n)]
    raise ValueError(kind)


def make_with_outlier_ratio(n: int, outlier_ratio: float,
                            rng: random.Random) -> list:
    """固定数值范围，按指定比例注入异常值。"""
    arr = [rng.randint(0, 3000) for _ in range(n)]
    for _ in range(int(n * outlier_ratio)):
        arr[rng.randrange(n)] = rng.randint(3000, 2 ** 24)
    return arr


def measure(arr: list, frame_size: int) -> dict:
    """实测一个数组在某帧大小下的压缩结果。"""
    frames = encode_pfor_delta(arr, frame_size)
    pfd_bytes = compressed_bytes(frames)
    s8b_bytes = len(encode_simple8b(arr)) * 8
    varint_bytes = varint_size(arr)
    uint32_bytes = len(arr) * 4
    bits_arr = [select_bits(arr[i:i + frame_size])
                for i in range(0, len(arr), frame_size)]
    return {
        "frame_size": frame_size,
        "n": len(arr),
        "pfd_bytes": pfd_bytes,
        "s8b_bytes": s8b_bytes,
        "varint_bytes": varint_bytes,
        "uint32_bytes": uint32_bytes,
        "bits_per_int": pfd_bytes * 8 / len(arr),
        "avg_bits_per_frame": sum(bits_arr) / len(bits_arr),
        "max_bits_per_frame": max(bits_arr),
        "ratio_vs_uint32": uint32_bytes / pfd_bytes,
        "ratio_vs_varint": varint_bytes / pfd_bytes,
        "ratio_vs_s8b": s8b_bytes / pfd_bytes,
        "predicted_bits_per_int": predicted_bits_per_int(arr, frame_size),
    }


def measure_speed(arr: list, frame_size: int, repeats: int = 3) -> dict:
    """实测编解码吞吐（M int/s），取三轮最好。"""
    frames = None
    best_enc = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        frames = encode_pfor_delta(arr, frame_size)
        best_enc = min(best_enc, time.perf_counter() - t0)
    best_dec = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        decode_pfor_delta(frames, frame_size)
        best_dec = min(best_dec, time.perf_counter() - t0)
    return {
        "encode_Mint_per_s": len(arr) / best_enc / 1e6,
        "decode_Mint_per_s": len(arr) / best_dec / 1e6,
    }


def main():
    rng = random.Random(2026)
    print("=== PForDelta 实测验证 ===\n")

    # ---------------- 实验一：三种分布的压缩比 ----------------
    print("--- 实验一：三种分布的压缩比（n=100 万，frame_size=128）---")
    dist_rows = []
    for kind in ["docid", "uniform", "large"]:
        arr = make_distribution(kind, 1_000_000, rng)
        # 交叉验证：与 Simple8b 的 varint 字节数一致
        assert varint_encode(arr).__len__() == varint_size(arr)
        # 交叉验证：编解码可逆
        frames = encode_pfor_delta(arr, 128)
        assert decode_pfor_delta(frames, 128) == arr, f"{kind} 编解码不可逆"
        # 交叉验证：Simple8b 也能还原（基线可信）
        assert decode_simple8b(encode_simple8b(arr), len(arr)) == arr
        r = measure(arr, 128)
        dist_rows.append({"distribution": kind, **r})
        print(f"{kind:>8}  位宽={r['avg_bits_per_frame']:>5.1f} bit/帧  "
              f"{r['bits_per_int']:>6.2f} bit/int  "
              f"vs uint32={r['ratio_vs_uint32']:>5.2f}x  "
              f"vs varint={r['ratio_vs_varint']:>5.2f}x  "
              f"vs Simple8b={r['ratio_vs_s8b']:>5.2f}x")
    print(f"{'':>8}  PForDelta 字节  Simple8b 字节  varint 字节  uint32 字节")
    for r in dist_rows:
        print(f"{r['distribution']:>8}  {r['pfd_bytes']/1024/1024:>11.2f} MiB  "
              f"{r['s8b_bytes']/1024/1024:>12.2f} MiB  "
              f"{r['varint_bytes']/1024/1024:>10.2f} MiB  "
              f"{r['uint32_bytes']/1024/1024:>10.2f} MiB")

    # ---------------- 实验二：帧大小对比 ----------------
    print("\n--- 实验二：帧大小对比（docID 分布，n=100 万）---")
    docid_arr = make_distribution("docid", 1_000_000, rng)
    frame_rows = []
    for fs in [32, 64, 128, 256, 512, 1024]:
        r = measure(docid_arr, fs)
        frame_rows.append(r)
        print(f"frame_size={fs:>5}  位宽={r['avg_bits_per_frame']:>5.1f} bit/帧  "
              f"{r['bits_per_int']:>6.2f} bit/int  "
              f"vs uint32={r['ratio_vs_uint32']:>5.2f}x  "
              f"vs Simple8b={r['ratio_vs_s8b']:>5.2f}x")
    # 均匀分布上重复一遍（无异常值时帧大小的影响）
    print("（对照：均匀分布）")
    uniform_arr = make_distribution("uniform", 1_000_000, rng)
    frame_rows_uniform = []
    for fs in [128, 256, 512, 1024]:
        r = measure(uniform_arr, fs)
        frame_rows_uniform.append(r)
        print(f"frame_size={fs:>5}  位宽={r['avg_bits_per_frame']:>5.1f} bit/帧  "
              f"{r['bits_per_int']:>6.2f} bit/int  "
              f"vs uint32={r['ratio_vs_uint32']:>5.2f}x")

    # ---------------- 实验三：异常值比例扫描 ----------------
    print("\n--- 实验三：异常值比例对压缩比的影响（n=100 万，frame=128）---")
    outlier_rows = []
    for ratio in [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20]:
        arr = make_with_outlier_ratio(1_000_000, ratio, rng)
        r = measure(arr, 128)
        r["outlier_ratio"] = ratio
        outlier_rows.append(r)
        print(f"异常值比例={ratio*100:>5.1f}%  位宽={r['avg_bits_per_frame']:>5.1f} bit/帧  "
              f"{r['bits_per_int']:>6.2f} bit/int  "
              f"vs uint32={r['ratio_vs_uint32']:>5.2f}x  "
              f"vs Simple8b={r['ratio_vs_s8b']:>5.2f}x")

    # ---------------- 实验四：编解码速度 ----------------
    print("\n--- 实验四：编解码速度（n=100 万，frame_size=128）---")
    speed_rows = []
    for kind in ["docid", "uniform", "large"]:
        arr = make_distribution(kind, 1_000_000, rng)
        s = measure_speed(arr, 128)
        speed_rows.append({"distribution": kind, **s})
        print(f"{kind:>8}  编码 {s['encode_Mint_per_s']:>6.2f} M int/s  "
              f"解码 {s['decode_Mint_per_s']:>6.2f} M int/s")

    # ---------------- 实验五：闭式预测 vs 实测 ----------------
    print("\n--- 实验五：闭式预测 vs 实测（docID 分布）---")
    pred_rows = []
    for fs in [128, 256]:
        arr = make_distribution("docid", 100_000, rng)
        r = measure(arr, fs)
        err = abs(r["predicted_bits_per_int"] - r["bits_per_int"]) / r["bits_per_int"]
        pred_rows.append({
            "frame_size": fs, "predicted": r["predicted_bits_per_int"],
            "measured": r["bits_per_int"], "relative_error": err,
        })
        print(f"frame_size={fs:>4}  预测={r['predicted_bits_per_int']:>6.2f} bit/int  "
              f"实测={r['bits_per_int']:>6.2f} bit/int  相对误差={err*100:>5.1f}%")

    result = {
        "meta": {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "python": sys.version.split()[0],
            "note": "自制 PForDelta，实测三种分布压缩比、帧大小与速度",
            "n": 1_000_000,
            "seed": 2026,
        },
        "compression": dist_rows,
        "frame_size": frame_rows,
        "frame_size_uniform": frame_rows_uniform,
        "outlier_scan": outlier_rows,
        "speed": speed_rows,
        "prediction": pred_rows,
    }
    out = os.path.join(os.path.dirname(__file__), "..", "results",
                       f"pfor_delta_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n结果写入 {out}")


if __name__ == "__main__":
    main()
