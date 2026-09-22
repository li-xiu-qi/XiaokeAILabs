# -*- coding: utf-8 -*-
"""
Space-Saving 实测验证

用纯 Python 自制 Space-Saving，在 Zipf 分布数据流上实测四件事：
1. 确定性保证：频率 > N/m 的元素是否 100% 留在摘要中
2. 阈值扫描：固定一批 φ 阈值，看每个 m 下的召回率 / 精确率 / 保证率
3. 估计误差：被追踪元素的 估计值 <= 真实值 <= 估计值 + N/m 的成立情况
4. 与 Misra-Gries 同流对照：准确率与内存占用

实现要点（Metwally, Agrawal & El Abbadi 2005）：
- m 个 (元素, count, error) 三元组
- 命中则 count +1；未命中则替换 count 最小的项，新项 (i, 1, 原最小计数)
- 用最小堆定位最小值，替换时弹堆顶，摊还 O(1)
- 堆中残留的过期条目用惰性删除处理（比对计数是否仍是当前代）

内存只统计摘要本体（counts + errors 字典）。

结果写入 results/space_saving_<timestamp>.json
"""
import os
import sys
import json
import heapq
import random
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from space_saving_model import (
    SpaceSavingSpec, compute,
    compute_error_bound, compute_guarantee_threshold,
)

PHI_LIST = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]


class SpaceSaving:
    """自制迷你 Space-Saving。最小堆 + 惰性删除。"""

    def __init__(self, m: int):
        self.m = m
        self.counts = {}       # 元素 -> count（估计频次）
        self.errors = {}       # 元素 -> error（进入时继承的最小计数）
        self.heap = []         # (count, error, 元素)，按 count 最小堆
        self.num_updates = 0
        self.num_replacements = 0

    def _peek_min(self):
        """取堆顶有效的最小项。"""
        while self.heap:
            c, e, item = self.heap[0]
            # 只认仍指向同一代计数的条目
            if item in self.counts and self.counts[item] == c \
                    and self.errors[item] == e:
                return c, e, item
            heapq.heappop(self.heap)
        return None

    def update(self, item):
        self.num_updates += 1
        if item in self.counts:
            self.counts[item] += 1
            heapq.heappush(self.heap,
                           (self.counts[item], self.errors[item], item))
            return
        if len(self.counts) < self.m:
            self.counts[item] = 1
            self.errors[item] = 0
            heapq.heappush(self.heap, (1, 0, item))
            return
        # 摘要已满：替换 count 最小的项
        victim = self._peek_min()
        if victim is None:
            return
        min_count, _, v_item = victim
        del self.counts[v_item]
        del self.errors[v_item]
        heapq.heappop(self.heap)
        self.counts[item] = 1
        self.errors[item] = min_count
        heapq.heappush(self.heap, (1, min_count, item))
        self.num_replacements += 1

    def estimate(self, item):
        if item in self.counts:
            return self.counts[item]
        return 0

    def error_of(self, item):
        if item in self.errors:
            return self.errors[item]
        return 0

    def tracked_items(self):
        return set(self.counts.keys())

    def counter_sum(self):
        return sum(self.counts.values())

    def table_bytes(self) -> int:
        """摘要本体内存（counts + errors 两个字典）。"""
        return _deep_sizeof(self.counts) + _deep_sizeof(self.errors)

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


def evaluate(ss: SpaceSaving, true_counts: dict, m: int, n: int) -> dict:
    """对一次运行做完整评估。"""
    ranked = sorted(true_counts.items(), key=lambda kv: -kv[1])
    tracked = ss.tracked_items()
    bound = compute_error_bound(n, m)

    sweep = []
    for phi in PHI_LIST:
        threshold = int(phi * n)
        true_hh = {it for it, c in true_counts.items() if c > threshold}
        detected = {it for it in tracked if ss.estimate(it) > threshold}
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

    top_k = {it for it, _ in ranked[:m]}
    top_k_tracked = len(top_k & tracked)

    # 估计误差：检查 估计值 <= 真实值 + N/m
    mean_count = n / len(true_counts)
    over_bound = 0
    max_rel_err = 0.0
    total_rel_err = 0.0
    num_checked = 0
    for it in tracked:
        true_c = true_counts[it]
        if true_c < mean_count:
            continue
        est = ss.estimate(it)
        if est > true_c + bound:
            over_bound += 1
        num_checked += 1
        if true_c > 0:
            rel = abs(est - true_c) / true_c
            max_rel_err = max(max_rel_err, rel)
            total_rel_err += rel

    own_threshold = compute_guarantee_threshold(n, m)
    own_sweep = [s for s in sweep if s["threshold"] <= own_threshold]
    at_guarantee = own_sweep[-1] if own_sweep else sweep[0]

    spec_m = compute(SpaceSavingSpec(n=n, m=m))
    return {
        "m": m,
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
        "counter_sum": ss.counter_sum(),
        "counter_sum_ratio": ss.counter_sum() / n,
        "num_replacements": ss.num_replacements,
        "memory_bytes_theory": spec_m.space_bytes,
        "memory_bytes_table": ss.table_bytes(),
        "memory_bytes_heap_artifact": ss.heap_bytes(),
        "sweep": sweep,
    }


def run_m(m: int, stream: list, true_counts: dict, n: int) -> dict:
    ss = SpaceSaving(m)
    for item in stream:
        ss.update(item)
    return evaluate(ss, true_counts, m, n)


def main():
    print("=== Space-Saving 实测验证 ===\n")

    num_items = 10_000
    n = 1_000_000
    m_list = [50, 200, 1000]

    print(f"Zipf 流（skew=1.0）：{num_items:,} 个不同元素，共 {n:,} 次更新\n")

    stream = gen_zipf(num_items, n, skew=1.0)
    true_counts = {}
    for item in stream:
        true_counts[item] = true_counts.get(item, 0) + 1

    print("--- 不同 m 在各自保证阈值上的表现 ---")
    print(f"{'m':>5}  {'保证阈值':>8}  {'真实HH':>7}  {'追踪数':>7}  "
          f"{'保证率':>7}  {'召回率':>7}  {'精确率':>7}  {'Top-m召回':>10}  "
          f"{'平均相对误差':>12}  {'越界':>4}  {'计数总和':>10}  {'内存(理论)':>10}")
    results = []
    for m in m_list:
        r = run_m(m, stream, true_counts, n)
        results.append(r)
        print(f"{r['m']:>5}  {r['guarantee_threshold']:>8,}  "
              f"{r['num_true_heavy_hitters']:>7,}  {r['num_tracked']:>7,}  "
              f"{r['guarantee_rate']:>7.4f}  {r['recall']:>7.4f}  "
              f"{r['precision']:>7.4f}  {r['top_k_recall']:>10.4f}  "
              f"{r['mean_relative_error']:>12.4f}  "
              f"{r['over_bound_violations']:>4}  "
              f"{r['counter_sum']:>10,}  {r['memory_bytes_theory']:>8,} B")

    print("\n--- 阈值扫描：固定 φ 下的召回率 ---")
    print(f"{'m':>5}  " + "  ".join(f"φ={p:<7}" for p in PHI_LIST))
    for r in results:
        cells = []
        for s in r["sweep"]:
            n_hh = s["num_true_heavy_hitters"]
            cells.append(f"{s['recall']:.3f}({n_hh:<4})" if n_hh else "  n/a    ")
        print(f"{r['m']:>5}  " + "  ".join(cells))

    # 与 Misra-Gries 同流对照
    print("\n--- 与 Misra-Gries 同流对照（同 k=m 参数）---")
    from misra_gries_model import compute as mg_compute, MisraGriesSpec
    from verify_misra_gries import MisraGries, evaluate as mg_evaluate
    print(f"{'k=m':>6}  {'MG召回':>8}  {'SS召回':>8}  {'MG精确':>8}  "
          f"{'SS精确':>8}  {'MG平均误差':>11}  {'SS平均误差':>11}  "
          f"{'MG内存':>8}  {'SS内存':>8}")
    comparison = []
    for k in [50, 200, 1000]:
        mg = MisraGries(k)
        ss = SpaceSaving(k)
        for item in stream:
            mg.update(item)
            ss.update(item)
        mg_r = mg_evaluate(mg, true_counts, k, n)
        ss_r = evaluate(ss, true_counts, k, n)
        comparison.append({
            "k": k,
            "misra_gries": mg_r,
            "space_saving": ss_r,
        })
        print(f"{k:>6}  {mg_r['recall']:>8.4f}  {ss_r['recall']:>8.4f}  "
              f"{mg_r['precision']:>8.4f}  {ss_r['precision']:>8.4f}  "
              f"{mg_r['mean_relative_error']:>11.4f}  "
              f"{ss_r['mean_relative_error']:>11.4f}  "
              f"{mg_compute(MisraGriesSpec(n=n, k=k)).space_bytes:>6,} B  "
              f"{compute(SpaceSavingSpec(n=n, m=k)).space_bytes:>6,} B")

    print("\n--- 计数总和守恒性检查 ---")
    print(f"{'m':>5}  {'计数总和':>10}  {'N':>10}  {'总和/N':>8}")
    for r in results:
        print(f"{r['m']:>5}  {r['counter_sum']:>10,}  {n:>10,}  "
              f"{r['counter_sum_ratio']:>8.4f}")

    result = {
        "meta": {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "python": sys.version.split()[0],
            "note": "自制迷你 Space-Saving，最小堆 + 惰性删除",
            "num_items": num_items,
            "num_streams": n,
            "distribution": "zipf(skew=1.0)",
        },
        "by_m": results,
        "vs_misra_gries": comparison,
    }

    out = os.path.join(os.path.dirname(__file__), "..", "results",
                       f"space_saving_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print(f"\n结果写入 {out}")


if __name__ == "__main__":
    main()
