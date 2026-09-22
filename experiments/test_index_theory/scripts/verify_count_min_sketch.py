# -*- coding: utf-8 -*-
"""
Count-Min Sketch 实测验证

用纯 Python 自制迷你 Count-Min Sketch，实测三件事：
1. 误差界：估计值 <= 真实值 + εN 的成立比例是否 >= 1-δ
2. 估计偏差随 (w, d) 组合的变化，验证 w 控制误差大小、d 控制成立概率
3. 不同频次分布（均匀 / Zipf）下的估计质量

实现要点：
- d 行 w 列计数数组，每行用一个独立哈希函数（双重哈希模拟）
- 更新：每行对应计数器 +1
- 查询：取 d 行最小值
- 关键性质：估计值永不低于真实值（点查询不会低估），只会被哈希冲突推高

结果写入 results/count_min_sketch_<timestamp>.json
"""
import os
import sys
import json
import hashlib
import random
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from count_min_sketch_model import (
    CountMinSketchSpec, compute,
    compute_width, compute_depth,
    compute_absolute_error, compute_relative_error_bound,
)


class CountMinSketch:
    """自制迷你 Count-Min Sketch。"""

    def __init__(self, width: int, depth: int, seed: int = 42):
        self.w = width
        self.d = depth
        self.table = [[0] * width for _ in range(depth)]
        self.total_updates = 0

    def _hash(self, item: str, row: int) -> int:
        """第 row 行的哈希函数。双重哈希模拟独立哈希族。"""
        h1 = int(hashlib.md5(item.encode("utf-8")).hexdigest(), 16)
        h2 = int(hashlib.sha1(item.encode("utf-8")).hexdigest(), 16)
        return (h1 + row * h2) % self.w

    def update(self, item: str, count: int = 1):
        """更新。每行对应计数器 +count。"""
        for row in range(self.d):
            self.table[row][self._hash(item, row)] += count
        self.total_updates += count

    def estimate(self, item: str) -> int:
        """查询估计值。取 d 行最小值。"""
        return min(self.table[row][self._hash(item, row)]
                   for row in range(self.d))

    def memory_bytes(self) -> int:
        """内存占用估算（按 4 字节/计数器）。"""
        return self.w * self.d * 4


def gen_zipf(num_items: int, num_streams: int, skew: float = 1.0,
             seed: int = 42) -> list:
    """生成 Zipf 分布的流。频次 ∝ 1/rank^skew。"""
    rng = random.Random(seed)
    weights = [1.0 / ((i + 1) ** skew) for i in range(num_items)]
    total = sum(weights)
    cum = []
    acc = 0.0
    for w in weights:
        cum.append((acc + w / total))
        acc += w / total
    stream = []
    for _ in range(num_streams):
        r = rng.random()
        lo, hi = 0, num_items - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if cum[mid] < r:
                lo = mid + 1
            else:
                hi = mid
        stream.append(f"item_{lo}")
    return stream


def measure_error_bound(stream: list, width: int, depth: int,
                        epsilon: float, num_items: int) -> dict:
    """实测误差界的成立比例。

    对每个元素检查 估计值 <= 真实值 + εN，统计成立比例是否 >= 1-δ。
    """
    cms = CountMinSketch(width, depth)

    # 统计真实频次
    true_counts = {}
    for item in stream:
        true_counts[item] = true_counts.get(item, 0) + 1

    # 更新 sketch
    for item in stream:
        cms.update(item)

    n = cms.total_updates
    abs_err_bound = epsilon * n

    over_estimates = 0
    total_err = 0
    max_rel_err = 0.0
    err_within_bound = 0

    for item, true_count in true_counts.items():
        est = cms.estimate(item)
        err = est - true_count
        if err > 0:
            over_estimates += 1
        total_err += max(0, err)
        if true_count > 0:
            max_rel_err = max(max_rel_err, err / true_count)
        if err <= abs_err_bound:
            err_within_bound += 1

    num_queried = len(true_counts)
    return {
        "width": width,
        "depth": depth,
        "epsilon": epsilon,
        "N": n,
        "num_distinct_items": num_queried,
        "abs_error_bound": abs_err_bound,
        "within_bound_ratio": err_within_bound / num_queried,
        "over_estimate_ratio": over_estimates / num_queried,
        "mean_absolute_error": total_err / num_queried,
        "max_relative_error": max_rel_err,
        "memory_bytes": cms.memory_bytes(),
    }


def measure_vs_width(stream: list, width_list: list, depth: int) -> list:
    """固定 d，变化 w。验证 w 控制误差大小。"""
    results = []
    for w in width_list:
        eps = compute_relative_error_bound(w, len(stream))
        r = measure_error_bound(stream, w, depth, eps, 10000)
        results.append(r)
    return results


def measure_vs_depth(stream: list, width: int, depth_list: list,
                     epsilon: float) -> list:
    """固定 w，变化 d。验证 d 控制误差界的成立概率。"""
    results = []
    for d in depth_list:
        r = measure_error_bound(stream, width, d, epsilon, 10000)
        results.append(r)
    return results


def main():
    print("=== Count-Min Sketch 实测验证 ===\n")

    num_items = 10_000
    num_streams = 100_000
    depth = 5

    # 1. Zipf 流：不同 w 的误差
    print("--- Zipf 流（100k 次更新，d=5）：不同 w 的平均绝对误差 ---")
    stream = gen_zipf(num_items, num_streams, skew=1.0)
    width_results = measure_vs_width(stream, [256, 512, 1024, 2048, 4096], depth)
    print(f"{'w':>6}  {'ε':>8}  {'误差界':>10}  {'界内比例':>9}  "
          f"{'高估比例':>9}  {'平均绝对误差':>14}  {'内存':>10}")
    for r in width_results:
        print(f"{r['width']:>6}  {r['epsilon']:>8.5f}  {r['abs_error_bound']:>10,.0f}  "
              f"{r['within_bound_ratio']:>9.4f}  {r['over_estimate_ratio']:>9.4f}  "
              f"{r['mean_absolute_error']:>14.2f}  {r['memory_bytes']:>9,} B")

    # 2. 不同 d 的成立比例
    print("\n--- Zipf 流（100k 次更新，w=2048）：不同 d 的误差界成立比例 ---")
    depth_results = measure_vs_depth(stream, 2048, [1, 3, 5, 7, 10],
                                     compute_relative_error_bound(2048, num_streams))
    print(f"{'d':>3}  {'ε':>8}  {'误差界':>10}  {'界内比例':>9}  "
          f"{'平均绝对误差':>14}  {'内存':>10}")
    for r in depth_results:
        print(f"{r['depth']:>3}  {r['epsilon']:>8.5f}  {r['abs_error_bound']:>10,.0f}  "
              f"{r['within_bound_ratio']:>9.4f}  "
              f"{r['mean_absolute_error']:>14.2f}  {r['memory_bytes']:>9,} B")

    # 3. 均匀分布对比
    print("\n--- 均匀流（100k 次更新，w=2048, d=5）---")
    rng = random.Random(42)
    uniform_stream = [f"item_{rng.randrange(num_items)}"
                      for _ in range(num_streams)]
    uni = measure_error_bound(uniform_stream, 2048, 5,
                              compute_relative_error_bound(2048, num_streams),
                              num_items)
    print(f"w=2048  d=5  界内比例={uni['within_bound_ratio']:.4f}  "
          f"高估比例={uni['over_estimate_ratio']:.4f}  "
          f"平均绝对误差={uni['mean_absolute_error']:.2f}")

    result = {
        "meta": {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "python": sys.version.split()[0],
            "note": "自制迷你 Count-Min Sketch，双重哈希模拟 d 个独立哈希函数",
            "num_items": num_items,
            "num_streams": num_streams,
            "distribution": "zipf(skew=1.0)",
        },
        "zipf_vs_width": width_results,
        "zipf_vs_depth": depth_results,
        "uniform_stream": uni,
    }

    out = os.path.join(os.path.dirname(__file__), "..", "results",
                       f"count_min_sketch_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print(f"\n结果写入 {out}")


if __name__ == "__main__":
    main()
