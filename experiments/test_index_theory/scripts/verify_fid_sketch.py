# -*- coding: utf-8 -*-
"""
FID-Sketch 实测验证

用纯 Python 自制迷你 FID-Sketch，实测四件事：
1. Morris 解码器的噪声常数（单个计数器与 d 行平均后）
2. 同等内存下与 32 位 Count-Min Sketch 的精度对比，覆盖多组 (w, d)
3. 低频项与高频项上的误差结构，验证「碰撞项加性、Morris 项乘性」的分工
4. 任务给定式 -m × ln(1 - p̂) 与 Morris 解码 ĉ = 2^v - 1 的偏差对照

实现要点：
- d 行 w 列，每格 4 位值 v（0-15）
- 更新：读到 v 后以概率 2^(-v) 把 v 加 1（v 封顶 15）
- 查询：逐行解码 ĉ_r = 2^v_r - 1，取 d 行算术平均
- 基线：scripts/verify_count_min_sketch.py 里的 Count-Min Sketch（32 位计数器）
- 流：Zipf 分布，10,000 个不同元素共 100,000 次更新，skew=1.0

结果写入 results/fid_sketch_<timestamp>.json
"""
import os
import sys
import json
import hashlib
import math
import random
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from fid_sketch_model import (
    FIDSketchSpec, compute,
    compute_width, compute_depth, compute_relative_error_bound,
    compute_increment_probability, compute_no_increment_probability,
    compute_decode, compute_morris_noise,
    compute_expected_relative_error, compute_saturation_limit,
)
from count_min_sketch_model import compute_relative_error_bound as cms_eps
from verify_count_min_sketch import CountMinSketch, gen_zipf


class FIDSketch:
    """自制迷你 FID-Sketch（4 位概率计数器）。"""

    def __init__(self, width: int, depth: int, counter_bits: int = 4,
                 seed: int = 42):
        self.w = width
        self.d = depth
        self.bits = counter_bits
        self.cap = (1 << counter_bits) - 1
        self.table = [[0] * width for _ in range(depth)]
        self.rng = random.Random(seed)
        self.total_updates = 0

    def _hash(self, item: str, row: int) -> int:
        """第 row 行的哈希函数。双重哈希模拟独立哈希族。"""
        h1 = int(hashlib.md5(item.encode("utf-8")).hexdigest(), 16)
        h2 = int(hashlib.sha1(item.encode("utf-8")).hexdigest(), 16)
        return (h1 + row * h2) % self.w

    def update(self, item: str):
        """更新。每行以概率 2^(-v) 递增。"""
        for row in range(self.d):
            j = self._hash(item, row)
            v = self.table[row][j]
            if v < self.cap and self.rng.random() < 2.0 ** (-v):
                self.table[row][j] = v + 1
        self.total_updates += 1

    def counters(self, item: str) -> list:
        """读出该元素在 d 行上的计数器值。"""
        return [self.table[row][self._hash(item, row)] for row in range(self.d)]

    def estimate(self, item: str) -> float:
        """Morris 解码后取 d 行平均。ĉ_r = 2^v_r - 1。"""
        vals = self.counters(item)
        return sum(compute_decode(v) for v in vals) / len(vals)

    def log_decode(self, item: str, m: int) -> float:
        """对照解码器：-m × ln(1 - p̂)，p̂ 为 d 行 p = 1 - 2^(-v) 的平均。

        这个式子量纲不成立（常数乘对数无法还原线性计数），
        留在代码里只为给出实测偏差数字。
        """
        vals = self.counters(item)
        p_hat = sum(compute_no_increment_probability(v) for v in vals) / len(vals)
        if p_hat >= 1.0:
            return float(m)
        return -m * math.log(1.0 - p_hat)

    def memory_bytes(self) -> int:
        """实际内存。每格 4 位。"""
        return self.w * self.d * self.bits // 8


def calibrate_morris_noise(g_list: list, depth_list: list) -> dict:
    """标定 Morris 解码器的噪声常数。"""
    rows = []
    for g in g_list:
        rng = random.Random(20260907)
        single = []
        for _ in range(6000):
            v = 0
            for _ in range(g):
                if rng.random() < 2.0 ** (-v) and v < 15:
                    v += 1
            single.append(compute_decode(v))
        mean = sum(single) / len(single)
        var = sum((e - mean) ** 2 for e in single) / max(1, len(single) - 1)
        std = math.sqrt(var)
        rows.append({
            "g": g,
            "mean_decode": mean,
            "bias_ratio": (mean - g) / g,
            "single_relative_std": std / mean,
        })

    avg_single = sum(r["single_relative_std"] for r in rows
                     if r["g"] >= 10) / max(1, sum(1 for r in rows if r["g"] >= 10))

    # d 行平均后的噪声
    avg_rows = []
    for d in depth_list:
        rng = random.Random(777)
        ests = []
        for _ in range(4000):
            vals = []
            for _ in range(d):
                v = 0
                for _ in range(rows[2]["g"]):
                    if rng.random() < 2.0 ** (-v) and v < 15:
                        v += 1
                vals.append(compute_decode(v))
            ests.append(sum(vals) / d)
        mean = sum(ests) / len(ests)
        var = sum((e - mean) ** 2 for e in ests) / max(1, len(ests) - 1)
        avg_rows.append({
            "d": d,
            "mean_decode": mean,
            "relative_std": math.sqrt(var) / mean,
            "predicted": compute_morris_noise(d, avg_single),
        })
    return {"single": rows, "average_single": avg_single, "by_depth": avg_rows}


def measure(true_counts: dict, sketch, n: int, epsilon: float) -> dict:
    """统计一个 sketch 在所有真实元素上的误差指标。"""
    abs_err_bound = epsilon * n
    total_abs = 0.0
    total_rel = 0.0
    head_abs = 0.0
    head_rel = 0.0
    head_n = 0
    over = 0
    within = 0
    max_rel = 0.0
    num = len(true_counts)

    for item, true_count in true_counts.items():
        est = sketch.estimate(item)
        err = est - true_count
        total_abs += abs(err)
        total_rel += abs(err) / true_count
        if err > 0:
            over += 1
        if err <= abs_err_bound:
            within += 1
        max_rel = max(max_rel, err / true_count)
        if true_count >= 100:
            head_abs += abs(err)
            head_rel += abs(err) / true_count
            head_n += 1

    return {
        "width": sketch.w,
        "depth": sketch.d,
        "epsilon": epsilon,
        "memory_bytes": sketch.memory_bytes(),
        "abs_error_bound": abs_err_bound,
        "mean_absolute_error": total_abs / num,
        "mean_abs_relative_error": total_rel / num,
        "head_mean_absolute_error": head_abs / max(1, head_n),
        "head_mean_abs_relative_error": head_rel / max(1, head_n),
        "head_count": head_n,
        "over_estimate_ratio": over / num,
        "within_bound_ratio": within / num,
        "max_relative_error": max_rel,
    }


def main():
    print("=== FID-Sketch 实测验证 ===\n")

    num_items = 10_000
    num_streams = 100_000
    stream = gen_zipf(num_items, num_streams, skew=1.0)
    true_counts = {}
    for item in stream:
        true_counts[item] = true_counts.get(item, 0) + 1
    n = len(stream)

    # 1. Morris 噪声标定
    print("--- Morris 解码器噪声标定（6000 次试验）---")
    cal = calibrate_morris_noise([1, 10, 100, 1000], [1, 3, 5, 7, 10, 20])
    print(f"{'g':>6}  {'解码均值':>10}  {'偏差占比':>9}  {'单计数器相对标准差':>20}")
    for r in cal["single"]:
        print(f"{r['g']:>6}  {r['mean_decode']:>10.1f}  "
              f"{r['bias_ratio']:>8.3%}  {r['single_relative_std']:>19.3f}")
    print(f"\n单计数器相对标准差均值 = {cal['average_single']:.3f}")
    print(f"\n{'d':>4}  {'实测相对标准差':>16}  {'理论 0.70/√d':>14}  {'比值':>7}")
    for r in cal["by_depth"]:
        print(f"{r['d']:>4}  {r['relative_std']:>15.3f}  "
              f"{r['predicted']:>13.3f}  {r['relative_std']/r['predicted']:>6.2f}")

    # 2. 日志解码器对照
    print(f"\n--- 解码器对照（w=2048, d=5，真实频次 1 到 10000）---")
    fid_probe = FIDSketch(2048, 5)
    for item in stream:
        fid_probe.update(item)
    print(f"\n{'真实频次':>9}  {'碰撞项 eN/(w·f)':>16}  {'Morris 解码':>13}  "
          f"{'-m·ln(1-p̂) 解码':>18}  {'Morris 相对误差':>16}  {'对数式相对误差':>16}")
    for target in [1, 5, 20, 100, 500, 2000]:
        # 找一个真实频次最接近 target 的元素
        best = min(true_counts.items(), key=lambda kv: abs(kv[1] - target))
        item, tc = best
        morris_est = fid_probe.estimate(item)
        log_est = fid_probe.log_decode(item, 2048)
        collision = math.e * num_streams / (2048 * tc)
        print(f"{tc:>9}  {collision:>15.2f}  {morris_est:>13.1f}  "
              f"{log_est:>18.1f}  "
              f"{(morris_est-tc)/tc:>15.2%}  {(log_est-tc)/tc:>15.2%}")

    # 3. 同等内存下的 (w, d) 对比
    print("\n--- 同等内存下 FID-Sketch vs Count-Min Sketch（Zipf, 100k 更新）---")
    budgets = [5120, 10240, 20480]
    # 每个预算给出 d 相同、FID 宽度为 CMS 8 倍的配对
    pairs = [
        (5120,  [(256, 5), (640, 2), (128, 10)]),
        (10240, [(512, 5), (1280, 2), (256, 10)]),
        (20480, [(1024, 5), (2560, 2), (512, 10)]),
    ]
    comparison = []
    for budget, cfgs in pairs:
        print(f"\n内存预算 {budget:,} B：")
        print(f"{'算法':>12}  {'w':>6}  {'d':>3}  {'平均绝对误差':>14}  "
              f"{'平均相对误差':>14}  {'头部相对误差':>14}  {'高估比例':>9}")
        for w, d in cfgs:
            cms = CountMinSketch(w, d)
            for item in stream:
                cms.update(item)
            eps = math.e / w
            r_cms = measure(true_counts, cms, n, eps)
            comparison.append({"algorithm": "CMS-32bit", **r_cms})
            print(f"{'CMS 32bit':>12}  {w:>6}  {d:>3}  "
                  f"{r_cms['mean_absolute_error']:>14.2f}  "
                  f"{r_cms['mean_abs_relative_error']:>13.2%}  "
                  f"{r_cms['head_mean_abs_relative_error']:>13.2%}  "
                  f"{r_cms['over_estimate_ratio']:>8.2%}")

            fw = w * 8
            fid = FIDSketch(fw, d)
            for item in stream:
                fid.update(item)
            r_fid = measure(true_counts, fid, n, math.e / fw)
            r_fid["memory_bytes"] = fid.memory_bytes()
            comparison.append({"algorithm": "FID-4bit", **r_fid})
            print(f"{'FID 4bit':>12}  {fw:>6}  {d:>3}  "
                  f"{r_fid['mean_absolute_error']:>14.2f}  "
                  f"{r_fid['mean_abs_relative_error']:>13.2%}  "
                  f"{r_fid['head_mean_abs_relative_error']:>13.2%}  "
                  f"{r_fid['over_estimate_ratio']:>8.2%}")

    # 4. 频率分档下的误差曲线
    print("\n--- 按真实频次分档的平均绝对相对误差（预算 10,240 B）---")
    brackets = [(1, 1), (2, 4), (5, 9), (10, 49), (50, 199),
                (200, 999), (1000, 10 ** 9)]
    cms = CountMinSketch(512, 5)
    fid = FIDSketch(4096, 5)
    for item in stream:
        cms.update(item)
        fid.update(item)
    print(f"{'频次区间':>14}  {'元素数':>7}  {'CMS 相对误差':>14}  {'FID 相对误差':>14}")
    bracket_rows = []
    for lo, hi in brackets:
        items = [it for it, c in true_counts.items() if lo <= c <= hi]
        if not items:
            continue
        cms_rel = sum(abs(cms.estimate(it) - true_counts[it]) / true_counts[it]
                      for it in items) / len(items)
        fid_rel = sum(abs(fid.estimate(it) - true_counts[it]) / true_counts[it]
                      for it in items) / len(items)
        bracket_rows.append({"lo": lo, "hi": hi, "count": len(items),
                             "cms_rel": cms_rel, "fid_rel": fid_rel})
        label = f"{lo}" if lo == hi else f"{lo}-{hi}"
        print(f"{label:>14}  {len(items):>7}  {cms_rel:>13.2%}  {fid_rel:>13.2%}")

    result = {
        "meta": {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "python": sys.version.split()[0],
            "note": "自制迷你 FID-Sketch（4 位 Morris 计数器），对比 32 位 Count-Min Sketch",
            "num_items": num_items,
            "num_streams": num_streams,
            "distribution": "zipf(skew=1.0)",
        },
        "morris_calibration": cal,
        "decoder_comparison_note": "log 解码式 -m*ln(1-p_hat) 量纲不成立，仅作对照",
        "equal_memory_comparison": comparison,
        "frequency_brackets": bracket_rows,
    }

    out = os.path.join(os.path.dirname(__file__), "..", "results",
                       f"fid_sketch_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print(f"\n结果写入 {out}")


if __name__ == "__main__":
    main()
