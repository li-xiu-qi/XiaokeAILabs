# -*- coding: utf-8 -*-
"""
Misra-Gries 实测验证

用纯 Python 自制 Misra-Gries，在 Zipf 分布数据流上实测四件事：
1. 确定性保证：频率 > N/k 的元素是否 100% 留在候选表中
2. 阈值扫描：固定一批 φ 阈值，看每个 k 下的召回率 / 精确率 / 保证率
3. Top-k 召回：真实频次前 k 名的元素有多少被追踪
4. 估计误差：被追踪元素的 估计值 <= 真实值 + N/k 的成立比例

实现要点（标准 Misra-Gries，Misra & Gries 1982）：
- 一张最多 k 个 (元素, 计数) 的表
- 命中则 +1；表未满则插入并置 1；表满则全体 -1 并删归零项
- 用「全局待减量 + 惰性删除堆」把全体 -1 摊还到 O(1)，
  语义与逐元素减 1 完全等价

内存只统计候选表本体（counts 字典）。惰性删除堆是摊还技巧的实现副产物，
不参与理论空间口径，另行列示。

结果写入 results/misra_gries_<timestamp>.json
"""
import os
import sys
import json
import heapq
import random
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from misra_gries_model import (
    MisraGriesSpec, compute,
    compute_error_bound, compute_guarantee_threshold,
)

# 阈值扫描档位（φ：频率占全流的比例）
PHI_LIST = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]


class MisraGries:
    """自制迷你 Misra-Gries。全局待减量 + 惰性删除堆实现。"""

    def __init__(self, k: int):
        self.k = k
        self.counts = {}        # 元素 -> 绝对存储计数（含待减量）
        self.sub = 0            # 全局待减量，有效计数 = 存储值 - sub
        self.heap = []          # (绝对计数, 元素)，惰性删除
        self.num_decrements = 0  # 全体减 1 的次数
        self.num_updates = 0

    def _purge(self):
        """清除有效计数已归零的表项。"""
        while self.heap:
            val, item = self.heap[0]
            if val - self.sub > 0:
                break
            heapq.heappop(self.heap)
            # 只删仍指向同一代计数的项，避免误删后插入的同名元素
            if self.counts.get(item) == val:
                del self.counts[item]

    def update(self, item):
        self.num_updates += 1
        if item in self.counts:
            # 命中：对应计数 +1
            self.counts[item] += 1
            heapq.heappush(self.heap, (self.counts[item], item))
            return
        # 未命中
        if len(self.counts) < self.k:
            # 表未满：直接插入，有效计数为 1
            self.counts[item] = self.sub + 1
            heapq.heappush(self.heap, (self.counts[item], item))
        else:
            # 表满：全体 -1（sub += 1），删归零项
            self.sub += 1
            self.num_decrements += 1
            self._purge()
            if len(self.counts) < self.k:
                self.counts[item] = self.sub + 1
                heapq.heappush(self.heap, (self.counts[item], item))

    def estimate(self, item):
        if item in self.counts:
            return self.counts[item] - self.sub
        return 0

    def tracked_items(self):
        """当前表里的元素集合。"""
        return set(self.counts.keys())

    def table_bytes(self) -> int:
        """候选表本体的内存占用（counts 字典）。"""
        return _deep_sizeof(self.counts)

    def heap_bytes(self) -> int:
        """惰性删除堆的内存占用（实现副产物，不计入理论空间）。"""
        return _deep_sizeof(self.heap)


def _deep_sizeof(obj, seen=None) -> int:
    """递归估算 Python 对象的真实内存占用。"""
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


def evaluate(mg: MisraGries, true_counts: dict, k: int, n: int) -> dict:
    """对一次运行做完整评估。"""
    # 按真实频次降序，供 Top-k 召回使用
    ranked = sorted(true_counts.items(), key=lambda kv: -kv[1])
    tracked = mg.tracked_items()
    bound = compute_error_bound(n, k)

    # 1. 阈值扫描
    sweep = []
    for phi in PHI_LIST:
        threshold = int(phi * n)
        true_hh = {it for it, c in true_counts.items() if c > threshold}
        detected = {it for it in tracked if mg.estimate(it) > threshold}
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

    # 2. Top-k 召回：真实频次前 k 名有多少被追踪
    top_k = {it for it, _ in ranked[:k]}
    top_k_tracked = len(top_k & tracked)

    # 3. 估计误差（只看被追踪且频次 >= 均值的元素，避免除零与噪声）
    mean_count = n / len(true_counts)
    over_bound = 0
    max_rel_err = 0.0
    total_rel_err = 0.0
    num_checked = 0
    for it in tracked:
        true_c = true_counts[it]
        if true_c < mean_count:
            continue
        est = mg.estimate(it)
        if est > true_c + bound:
            over_bound += 1
        num_checked += 1
        if true_c > 0:
            rel = abs(est - true_c) / true_c
            max_rel_err = max(max_rel_err, rel)
            total_rel_err += rel

    # 算法自身保证的阈值
    own_threshold = compute_guarantee_threshold(n, k)
    own_sweep = [s for s in sweep if s["threshold"] <= own_threshold]
    at_guarantee = own_sweep[-1] if own_sweep else sweep[0]

    m = compute(MisraGriesSpec(n=n, k=k))
    return {
        "k": k,
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
        "num_decrements": mg.num_decrements,
        "decrement_ratio": mg.num_decrements / n,
        "memory_bytes_theory": m.space_bytes,
        "memory_bytes_table": mg.table_bytes(),
        "memory_bytes_heap_artifact": mg.heap_bytes(),
        "sweep": sweep,
    }


def run_k(k: int, stream: list, true_counts: dict, n: int) -> dict:
    """给定 k 跑一次完整评估。"""
    mg = MisraGries(k)
    for item in stream:
        mg.update(item)
    return evaluate(mg, true_counts, k, n)


def main():
    print("=== Misra-Gries 实测验证 ===\n")

    num_items = 10_000
    n = 1_000_000
    k_list = [10, 100, 1000]

    print(f"Zipf 流（skew=1.0）：{num_items:,} 个不同元素，共 {n:,} 次更新\n")

    stream = gen_zipf(num_items, n, skew=1.0)
    true_counts = {}
    for item in stream:
        true_counts[item] = true_counts.get(item, 0) + 1

    print("--- 不同 k 在各自保证阈值上的表现 ---")
    print(f"{'k':>5}  {'保证阈值':>8}  {'真实HH':>7}  {'追踪数':>7}  "
          f"{'保证率':>7}  {'召回率':>7}  {'精确率':>7}  {'Top-k召回':>10}  "
          f"{'平均相对误差':>12}  {'越界':>4}  {'内存(理论)':>10}")
    results = []
    for k in k_list:
        r = run_k(k, stream, true_counts, n)
        results.append(r)
        print(f"{r['k']:>5}  {r['guarantee_threshold']:>8,}  "
              f"{r['num_true_heavy_hitters']:>7,}  {r['num_tracked']:>7,}  "
              f"{r['guarantee_rate']:>7.4f}  {r['recall']:>7.4f}  "
              f"{r['precision']:>7.4f}  {r['top_k_recall']:>10.4f}  "
              f"{r['mean_relative_error']:>12.4f}  "
              f"{r['over_bound_violations']:>4}  "
              f"{r['memory_bytes_theory']:>8,} B")

    # 阈值扫描明细
    print("\n--- 阈值扫描：固定 φ 下的召回率 ---")
    header = f"{'k':>5}  " + "  ".join(
        f"φ={p:<7}" for p in PHI_LIST)
    print(header)
    for r in results:
        cells = []
        for s in r["sweep"]:
            n_hh = s["num_true_heavy_hitters"]
            cells.append(f"{s['recall']:.3f}({n_hh:<4})" if n_hh else "  n/a    ")
        print(f"{r['k']:>5}  " + "  ".join(cells))

    print("\n--- 全体减 1 的代价（Misra-Gries 的主要开销）---")
    print(f"{'k':>5}  {'减1次数':>10}  {'占比':>8}  {'等效逐元素操作':>16}")
    for r in results:
        equiv = r["num_decrements"] * r["k"]
        print(f"{r['k']:>5}  {r['num_decrements']:>10,}  "
              f"{r['decrement_ratio']:>8.4f}  {equiv:>16,}")

    # 与 Space-Saving 的对照
    print("\n--- 同参数下与 Space-Saving 的对照 ---")
    from space_saving_model import compute as ss_compute, SpaceSavingSpec
    print(f"{'k=m':>6}  {'MG误差界':>9}  {'SS误差界':>9}  "
          f"{'MG内存':>9}  {'SS内存':>9}")
    for k in k_list:
        mg_m = compute(MisraGriesSpec(n=n, k=k))
        ss_m = ss_compute(SpaceSavingSpec(n=n, m=k))
        print(f"{k:>6}  {mg_m.error_bound:>9,}  {ss_m.error_bound:>9,}  "
              f"{mg_m.space_bytes:>7,} B  {ss_m.space_bytes:>7,} B")

    result = {
        "meta": {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "python": sys.version.split()[0],
            "note": "自制迷你 Misra-Gries，全局待减量 + 惰性删除堆摊还 O(1)",
            "num_items": num_items,
            "num_streams": n,
            "distribution": "zipf(skew=1.0)",
        },
        "by_k": results,
    }

    out = os.path.join(os.path.dirname(__file__), "..", "results",
                       f"misra_gries_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print(f"\n结果写入 {out}")


if __name__ == "__main__":
    main()
