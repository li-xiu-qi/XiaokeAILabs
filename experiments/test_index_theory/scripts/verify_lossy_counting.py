# -*- coding: utf-8 -*-
"""
Lossy Counting 实测验证

用纯 Python 自制 Lossy Counting，在 Zipf 分布数据流上实测四件事：
1. 确定性保证：频率 > εN 的元素是否 100% 留在表中
2. 阈值扫描：固定一批 φ 阈值，看每个 ε 下的召回率 / 精确率 / 保证率
3. 估计误差：被追踪元素的 估计值 <= 真实值 <= 估计值 + εN 的成立情况
4. 与 Space-Saving 同流对照：准确率、内存、桶边界淘汰次数

实现要点（Manku & Motwani 2002）：
- 桶宽 w = ceil(1/ε)，流被切成 N/w 个桶
- 表项 (元素, 频次, delta)，delta = 插入时的桶号 - 1，插入后不再变
- 命中则频次 +1；未命中则插入 (i, 1, 当前桶号 - 1)
- 桶结束时删掉 频次 + delta <= 当前桶号 的项
- 因为 频次 + delta 单调不降，用最小堆 + 惰性删除做批量淘汰，
  避免每个桶边界都全表扫描

内存只统计表本体（freq + delta 两个字典）。

结果写入 results/lossy_counting_<timestamp>.json
"""
import os
import sys
import json
import heapq
import random
import math
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from lossy_counting_model import (
    LossyCountingSpec, compute,
    compute_bucket_size, compute_error_bound, compute_guarantee_threshold,
)

PHI_LIST = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]


class LossyCounting:
    """自制迷你 Lossy Counting。最小堆 + 惰性删除做桶边界淘汰。"""

    def __init__(self, epsilon: float):
        self.epsilon = epsilon
        self.w = compute_bucket_size(epsilon)
        self.freqs = {}         # 元素 -> 频次
        self.deltas = {}        # 元素 -> delta（插入时桶号 - 1，固定）
        self.heap = []          # (频次+delta, 元素)，惰性删除
        self.num_processed = 0
        self.current_bucket = 1
        self.num_prunes = 0     # 桶边界淘汰次数
        self.num_deleted = 0    # 累计淘汰的表项数

    def _push(self, item):
        key = self.freqs[item] + self.deltas[item]
        heapq.heappush(self.heap, (key, item))

    def update(self, item):
        self.num_processed += 1
        if item in self.freqs:
            self.freqs[item] += 1
            self._push(item)
        else:
            self.freqs[item] = 1
            self.deltas[item] = self.current_bucket - 1
            self._push(item)
        # 桶边界：处理满 w 个元素后进入下一桶，并淘汰
        if self.num_processed % self.w == 0:
            self.current_bucket += 1
            self._prune()

    def _prune(self):
        """删掉 频次 + delta <= 当前桶号 的项。"""
        b = self.current_bucket
        deleted = 0
        while self.heap:
            key, item = self.heap[0]
            if key > b:
                break
            heapq.heappop(self.heap)
            # 惰性删除：只认仍指向同一代计数的项
            if item in self.freqs and self.freqs[item] + self.deltas[item] == key:
                del self.freqs[item]
                del self.deltas[item]
                deleted += 1
        if deleted:
            self.num_prunes += 1
            self.num_deleted += deleted

    def estimate(self, item):
        if item in self.freqs:
            return self.freqs[item]
        return 0

    def error_of(self, item):
        if item in self.deltas:
            return self.deltas[item]
        return 0

    def tracked_items(self):
        return set(self.freqs.keys())

    def num_buckets(self):
        return self.current_bucket

    def table_bytes(self) -> int:
        """表本体内存（freqs + deltas 两个字典）。"""
        return _deep_sizeof(self.freqs) + _deep_sizeof(self.deltas)

    def heap_bytes(self) -> int:
        """堆的内存占用（实现副产物，不计入理论空间）。"""
        return _deep_sizeof(self.heap)


def _deep_sizeof(obj, seen=None) -> int:
    if seen is None:
        seen = set()
    oid = id(obj)
    if oid in seen:
        return 0
    seen.add(oid)
    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        for k, v in obj.items():
            size += _deep_sizeof(k, seen) + _deep_sizeof(v, seen)
    elif isinstance(obj, (list, tuple, set)):
        for v in obj:
            size += _deep_sizeof(v, seen)
    return size


def gen_zipf(num_items: int, num_streams: int, skew: float = 1.0,
             seed: int = 42) -> list:
    """生成 Zipf 分布的流。频次 ∝ 1/rank^skew。"""
    rng = random.Random(seed)
    weights = [1.0 / ((i + 1) ** skew) for i in range(num_items)]
    total = sum(weights)
    cum = []
    acc = 0.0
    for w in weights:
        acc += w / total
        cum.append(acc)
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


def evaluate(lc: LossyCounting, true_counts: dict, epsilon: float,
             n: int) -> dict:
    """对一次运行做完整评估。"""
    ranked = sorted(true_counts.items(), key=lambda kv: -kv[1])
    tracked = lc.tracked_items()
    bound = compute_error_bound(epsilon, n)

    sweep = []
    for phi in PHI_LIST:
        threshold = int(phi * n)
        true_hh = {it for it, c in true_counts.items() if c > threshold}
        detected = {it for it in tracked if lc.estimate(it) > threshold}
        tp = len(true_hh & detected)
        sweep.append({
            "phi": phi,
            "threshold": threshold,
            "num_true_heavy_hitters": len(true_hh),
            "guarantee_rate": (len(true_hh & tracked) / len(true_hh)
                               if true_hh else 1.0),
            "recall": tp / len(true_hh) if true_hh else 1.0,
            "precision": tp / len(detected) if detected else 1.0,
        })

    top_k = {it for it, _ in ranked[:lc.w]}
    top_k_tracked = len(top_k & tracked)

    mean_count = n / len(true_counts)
    over_bound = 0
    max_rel_err = 0.0
    total_rel_err = 0.0
    num_checked = 0
    for it in tracked:
        true_c = true_counts[it]
        if true_c < mean_count:
            continue
        est = lc.estimate(it)
        if est > true_c + bound:
            over_bound += 1
        num_checked += 1
        if true_c > 0:
            rel = abs(est - true_c) / true_c
            max_rel_err = max(max_rel_err, rel)
            total_rel_err += rel

    own_threshold = compute_guarantee_threshold(epsilon, n)
    own_sweep = [s for s in sweep if s["threshold"] <= own_threshold]
    at_guarantee = own_sweep[-1] if own_sweep else sweep[0]

    spec_m = compute(LossyCountingSpec(n=n, epsilon=epsilon))
    return {
        "epsilon": epsilon,
        "bucket_size": lc.w,
        "num_buckets": lc.num_buckets(),
        "N": n,
        "error_bound": bound,
        "guarantee_threshold": own_threshold,
        "num_true_heavy_hitters": at_guarantee["num_true_heavy_hitters"],
        "num_tracked": len(tracked),
        "guarantee_rate": at_guarantee["guarantee_rate"],
        "recall": at_guarantee["recall"],
        "precision": at_guarantee["precision"],
        "top_k_total": len(top_k),
        "top_k_tracked": top_k_tracked,
        "top_k_recall": top_k_tracked / len(top_k) if top_k else 1.0,
        "num_checked": num_checked,
        "over_bound_violations": over_bound,
        "max_relative_error": max_rel_err,
        "mean_relative_error": total_rel_err / num_checked if num_checked else 0.0,
        "num_prunes": lc.num_prunes,
        "num_deleted": lc.num_deleted,
        "memory_bytes_theory": spec_m.space_bytes,
        "memory_bytes_table": lc.table_bytes(),
        "memory_bytes_heap_artifact": lc.heap_bytes(),
        "sweep": sweep,
    }


def run_epsilon(epsilon: float, stream: list, true_counts: dict,
                n: int) -> dict:
    lc = LossyCounting(epsilon)
    for item in stream:
        lc.update(item)
    return evaluate(lc, true_counts, epsilon, n)


def main():
    print("=== Lossy Counting 实测验证 ===\n")

    num_items = 10_000
    n = 1_000_000
    eps_list = [0.001, 0.01, 0.1]

    print(f"Zipf 流（skew=1.0）：{num_items:,} 个不同元素，共 {n:,} 次更新\n")

    stream = gen_zipf(num_items, n, skew=1.0)
    true_counts = {}
    for item in stream:
        true_counts[item] = true_counts.get(item, 0) + 1

    print("--- 不同 ε 在各自保证阈值上的表现 ---")
    print(f"{'ε':>7}  {'桶宽':>6}  {'桶数':>7}  {'保证阈值':>9}  {'真实HH':>7}  "
          f"{'追踪数':>7}  {'保证率':>7}  {'召回率':>7}  {'精确率':>7}  "
          f"{'平均相对误差':>12}  {'越界':>4}  {'内存(理论)':>10}")
    results = []
    for eps in eps_list:
        r = run_epsilon(eps, stream, true_counts, n)
        results.append(r)
        print(f"{r['epsilon']:>7}  {r['bucket_size']:>6}  "
              f"{r['num_buckets']:>7,}  {r['guarantee_threshold']:>9,}  "
              f"{r['num_true_heavy_hitters']:>7,}  {r['num_tracked']:>7,}  "
              f"{r['guarantee_rate']:>7.4f}  {r['recall']:>7.4f}  "
              f"{r['precision']:>7.4f}  {r['mean_relative_error']:>12.4f}  "
              f"{r['over_bound_violations']:>4}  "
              f"{r['memory_bytes_theory']:>8,} B")

    print("\n--- 阈值扫描：固定 φ 下的召回率 ---")
    print(f"{'ε':>7}  " + "  ".join(f"φ={p:<7}" for p in PHI_LIST))
    for r in results:
        cells = []
        for s in r["sweep"]:
            n_hh = s["num_true_heavy_hitters"]
            cells.append(f"{s['recall']:.3f}({n_hh:<4})" if n_hh else "  n/a    ")
        print(f"{r['epsilon']:>7}  " + "  ".join(cells))

    # 与 Space-Saving 同流对照（同 m=w 参数）
    print("\n--- 与 Space-Saving 同流对照（同 m = 1/ε 参数）---")
    from space_saving_model import compute as ss_compute, SpaceSavingSpec
    from verify_space_saving import SpaceSaving, evaluate as ss_evaluate
    print(f"{'ε':>7}  {'m=w':>6}  {'LC召回':>8}  {'SS召回':>8}  "
          f"{'LC精确':>8}  {'SS精确':>8}  {'LC平均误差':>11}  {'SS平均误差':>11}  "
          f"{'LC内存':>9}  {'SS内存':>9}")
    comparison = []
    for eps in eps_list:
        w = compute_bucket_size(eps)
        lc = LossyCounting(eps)
        ss = SpaceSaving(w)
        for item in stream:
            lc.update(item)
            ss.update(item)
        lc_r = evaluate(lc, true_counts, eps, n)
        ss_r = ss_evaluate(ss, true_counts, w, n)
        comparison.append({
            "epsilon": eps,
            "m": w,
            "lossy_counting": lc_r,
            "space_saving": ss_r,
        })
        print(f"{eps:>7}  {w:>6}  {lc_r['recall']:>8.4f}  {ss_r['recall']:>8.4f}  "
              f"{lc_r['precision']:>8.4f}  {ss_r['precision']:>8.4f}  "
              f"{lc_r['mean_relative_error']:>11.4f}  "
              f"{ss_r['mean_relative_error']:>11.4f}  "
              f"{lc_r['memory_bytes_theory']:>7,} B  "
              f"{ss_r['memory_bytes_theory']:>7,} B")

    print("\n--- 桶边界淘汰开销 ---")
    print(f"{'ε':>7}  {'桶数':>7}  {'有效淘汰次数':>13}  {'累计淘汰项':>11}")
    for r in results:
        print(f"{r['epsilon']:>7}  {r['num_buckets']:>7,}  "
              f"{r['num_prunes']:>13,}  {r['num_deleted']:>11,}")

    result = {
        "meta": {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "python": sys.version.split()[0],
            "note": "自制迷你 Lossy Counting，最小堆 + 惰性删除做桶边界淘汰",
            "num_items": num_items,
            "num_streams": n,
            "distribution": "zipf(skew=1.0)",
        },
        "by_epsilon": results,
        "vs_space_saving": comparison,
    }

    out = os.path.join(os.path.dirname(__file__), "..", "results",
                       f"lossy_counting_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print(f"\n结果写入 {out}")


if __name__ == "__main__":
    main()
