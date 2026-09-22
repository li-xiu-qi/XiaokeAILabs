# -*- coding: utf-8 -*-
"""
Simple8b 实测验证

自制 Simple8b（字对齐位封装），实测四项：

1. 三种分布的压缩比
   小整数（0-15）、中等整数（0-1000）、大整数（0-2^20），
   对比 Varint（LEB128）与定长 uint32。
   这是 Simple8b 的主场：数值越小，位宽越低，压缩比越高。

2. 闭式预测 vs 实测
   用 predicted_bits_per_int 分窗预测，对比实测每整数位宽，
   验证「档位由段内最大值决定」这一假设。

3. docID 列表的真实场景
   大部分小、偶尔大（带异常值的有序 docID），这是倒排索引的典型分布。

4. 编码/解码速度
   纯 Python 实现的每整数吞吐（M ints/s），
   说明字对齐编码的理论 SIMD 潜力与本实现的差距。

方法：自制编解码器 + 时间采样。结果写入 results/simple8b_<timestamp>.json
"""
import os
import sys
import json
import time
import random
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from simple8b_model import (
    SELECTOR_TABLE, encode_simple8b, decode_simple8b, bits_per_int,
    predicted_bits_per_int, ratio_vs_uint32, ratio_vs_varint, varint_size,
    compressed_bytes,
)


def varint_encode(arr: list) -> bytearray:
    """LEB128 Varint 编码，用于交叉验证 varint_size 的字节数。"""
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
    if kind == "small":            # 0-15，4 位足够
        return [rng.randint(0, 15) for _ in range(n)]
    if kind == "medium":           # 0-1000，10 位足够
        return [rng.randint(0, 1000) for _ in range(n)]
    if kind == "large":            # 0-2^20，21 位
        return [rng.randint(0, 2 ** 20) for _ in range(n)]
    if kind == "docid":            # docID 列表：小值为主，偶尔大
        arr = [rng.randint(0, 3000) for _ in range(n)]
        for _ in range(n // 100):  # 1% 的大值（异常值）
            arr[rng.randrange(n)] = rng.randint(3000, 2 ** 24)
        return arr
    raise ValueError(kind)


def measure_compression(arr: list, window: int = 0) -> dict:
    """实测一个数组的压缩比与位宽。"""
    words = encode_simple8b(arr)
    s8b_bytes = compressed_bytes(words)
    varint_bytes = varint_size(arr)
    uint32_bytes = len(arr) * 4
    return {
        "n": len(arr),
        "max_value": max(arr) if arr else 0,
        "bits_per_int": bits_per_int(arr),
        "predicted_bits_per_int": predicted_bits_per_int(arr, window),
        "s8b_bytes": s8b_bytes,
        "varint_bytes": varint_bytes,
        "uint32_bytes": uint32_bytes,
        "ratio_vs_uint32": uint32_bytes / s8b_bytes,
        "ratio_vs_varint": varint_bytes / s8b_bytes,
    }


def measure_speed(arr: list, repeats: int = 3) -> dict:
    """实测编解码吞吐（M int/s）。"""
    words = None
    # 编码：取最好的一轮
    best_enc = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        words = encode_simple8b(arr)
        best_enc = min(best_enc, time.perf_counter() - t0)
    # 解码：取最好的一轮
    best_dec = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        decode_simple8b(words, len(arr))
        best_dec = min(best_dec, time.perf_counter() - t0)
    return {
        "encode_Mint_per_s": len(arr) / best_enc / 1e6,
        "decode_Mint_per_s": len(arr) / best_dec / 1e6,
        "encode_s": best_enc,
        "decode_s": best_dec,
    }


def main():
    rng = random.Random(2026)
    print("=== Simple8b 实测验证 ===\n")

    # ---------------- 实验一：三种分布的压缩比 ----------------
    print("--- 实验一：三种分布的压缩比（n=100 万）---")
    n = 1_000_000
    dist_rows = []
    for kind in ["small", "medium", "large", "docid"]:
        arr = make_distribution(kind, n, rng)
        r = measure_compression(arr)
        # 交叉验证 varint 字节数
        assert varint_encode(arr).__len__() == r["varint_bytes"], \
            f"{kind} varint 字节数不一致"
        # 交叉验证可逆
        words = encode_simple8b(arr)
        assert decode_simple8b(words, n) == arr, f"{kind} 编解码不可逆"
        dist_rows.append({"distribution": kind, **r})
        print(f"{kind:>8}  最大值={r['max_value']:>10,}  "
              f"位宽={r['bits_per_int']:>6.2f} bit/int  "
              f"vs uint32={r['ratio_vs_uint32']:>5.2f}x  "
              f"vs varint={r['ratio_vs_varint']:>5.2f}x")
    print(f"{'':>8}  Simple8b 字节  varint 字节  uint32 字节")
    for r in dist_rows:
        print(f"{r['distribution']:>8}  {r['s8b_bytes']/1024/1024:>10.2f} MiB  "
              f"{r['varint_bytes']/1024/1024:>10.2f} MiB  "
              f"{r['uint32_bytes']/1024/1024:>10.2f} MiB")

    # ---------------- 实验二：闭式预测 vs 实测 ----------------
    print("\n--- 实验二：闭式预测 vs 实测（扫 cover 比例，docID 分布）---")
    pred_rows = []
    docid_arr = make_distribution("docid", 100_000, rng)
    for cover in [0.5, 0.9, 0.99, 1.0]:
        pred = predicted_bits_per_int(docid_arr, cover=cover)
        meas = bits_per_int(docid_arr)
        err = abs(pred - meas) / meas
        pred_rows.append({
            "cover": cover, "predicted": pred,
            "measured": meas, "relative_error": err,
        })
        print(f"cover={cover:>4}  预测={pred:>6.2f} bit/int  "
              f"实测={meas:>6.2f} bit/int  相对误差={err*100:>5.1f}%")

    # ---------------- 实验三：最大值扫描（位宽台阶） ----------------
    print("\n--- 实验三：最大值对位宽的影响（0 到 2^30）---")
    step_rows = []
    for k in range(0, 31, 3):
        top = 2 ** k - 1
        arr = make_distribution("small" if k < 5 else "medium", 100_000, rng)
        arr = [rng.randint(0, top) for _ in range(100_000)]
        r = measure_compression(arr)
        step_rows.append({
            "max_value": top, "bits_per_int": r["bits_per_int"],
            "ratio_vs_uint32": r["ratio_vs_uint32"],
        })
        print(f"最大值={top:>12,} ({k:>2} bit)  位宽={r['bits_per_int']:>6.2f} bit/int  "
              f"vs uint32={r['ratio_vs_uint32']:>5.2f}x")

    # ---------------- 实验四：编解码速度 ----------------
    print("\n--- 实验四：编解码速度（n=100 万，纯 Python）---")
    speed_rows = []
    for kind in ["small", "medium", "large"]:
        arr = make_distribution(kind, 1_000_000, rng)
        s = measure_speed(arr)
        speed_rows.append({"distribution": kind, **s})
        print(f"{kind:>8}  编码 {s['encode_Mint_per_s']:>6.2f} M int/s  "
              f"解码 {s['decode_Mint_per_s']:>6.2f} M int/s")

    result = {
        "meta": {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "python": sys.version.split()[0],
            "note": "自制 Simple8b，实测三种分布压缩比与编解码速度",
            "n": n,
            "seed": 2026,
        },
        "selector_table": [
            {"selector": i, "bits": b, "count": c}
            for i, (b, c) in enumerate(SELECTOR_TABLE)
        ],
        "compression": dist_rows,
        "prediction": pred_rows,
        "value_scan": step_rows,
        "speed": speed_rows,
    }
    out = os.path.join(os.path.dirname(__file__), "..", "results",
                       f"simple8b_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n结果写入 {out}")


if __name__ == "__main__":
    main()
