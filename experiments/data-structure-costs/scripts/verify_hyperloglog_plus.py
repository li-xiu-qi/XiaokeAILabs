# -*- coding: utf-8 -*-
"""
HyperLogLog++ 实测验证

用纯 Python 自制迷你 HyperLogLog++，实测四件事：
1. 小基数（100 到 10 万）下稀疏表示的精度，对比原版 HyperLogLog
2. 稀疏表示的内存曲线，以及稀疏→密集的切换点
3. 偏差校正在 2.5m 到 5m 区间的效果（原始估计 vs 校正后 vs 真实值）
4. 大基数（1M / 10M）下 64 位哈希的表现

实现要点：
- 64 位哈希，前 p 位选桶，剩余位中前导零个数 +1 为寄存器值
- 稀疏模式：dict{桶索引: 寄存器值}，只存非零桶
- 条目数达到 m × 6/32 时切换到密集数组，切换后不再回退
- 稀疏模式用线性计数 E = m × ln(m/V)，V = m - 不同桶索引数
- 密集模式：E < 2.5m 走线性计数，E <= 5m 走偏差校正，更大走大基数修正
- 基线：scripts/verify_hyperloglog.py 里的原版 HyperLogLog

结果写入 results/hyperloglog_plus_<timestamp>.json
"""
import os
import sys
import json
import hashlib
import math
import random
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from hyperloglog_plus_model import (
    HyperLogLogPlusSpec, compute,
    compute_alpha, compute_standard_error,
    compute_raw_estimate, compute_cardinality_plus,
    compute_sparse_threshold, compute_sparse_estimate,
    compute_sparse_memory_bytes, build_bias_table,
    compute_bias, compute_bias_corrected_estimate,
)
from hyperloglog_model import compute_cardinality
from verify_hyperloglog import HyperLogLog


class HyperLogLogPlus:
    """自制迷你 HyperLogLog++（含稀疏与密集两种表示）。"""

    def __init__(self, p: int = 14, bits_per_register: int = 6):
        self.p = p
        self.m = 2 ** p
        self.bits_per_register = bits_per_register
        self.register_mask = (1 << bits_per_register) - 1
        # 稀疏表示：桶索引 -> 寄存器值，只存非零桶
        self.sparse = {}
        # 密集表示：长度 m 的数组，切换后启用
        self.dense = None
        self.threshold = compute_sparse_threshold(self.m)
        self.switched_at = None  # 记录切换时已插入的元素数
        self.num_added = 0

    @property
    def mode(self) -> str:
        return "dense" if self.dense is not None else "sparse"

    def _hash64(self, item: str) -> int:
        """64 位哈希值。"""
        h = hashlib.sha1(item.encode("utf-8")).digest()
        return int.from_bytes(h[:8], "big")

    def _rank(self, h: int) -> tuple:
        """由哈希值算出桶索引与寄存器值。"""
        idx = h >> (64 - self.p)
        w = h & ((1 << (64 - self.p)) - 1)
        if w == 0:
            rank = 64 - self.p
        else:
            rank = (64 - self.p) - w.bit_length()
        rank = min(rank + 1, self.register_mask)
        return idx, rank

    def add(self, item: str):
        """插入元素。稀疏模式更新 dict，条目数过阈值后切到密集数组。"""
        self.num_added += 1
        idx, rank = self._rank(self._hash64(item))

        if self.dense is not None:
            if rank > self.dense[idx]:
                self.dense[idx] = rank
            return

        cur = self.sparse.get(idx)
        if cur is None or rank > cur:
            self.sparse[idx] = rank

        if len(self.sparse) >= self.threshold:
            self.dense = [0] * self.m
            for j, r in self.sparse.items():
                self.dense[j] = r
            self.sparse = {}
            self.switched_at = self.num_added

    def count(self, table=None, use_bias: bool = True) -> float:
        """基数估计。"""
        if self.dense is not None:
            return compute_cardinality_plus(
                self.m, self.m, self.dense, table=table, use_bias=use_bias)
        return compute_cardinality_plus(
            self.m, len(self.sparse), None, table=table, use_bias=use_bias)

    def dense_count(self) -> float:
        """对照组：强制按密集数组估计（含原版的全部修正）。"""
        if self.dense is None:
            return compute_cardinality_plus(
                self.m, self.m,
                [self.sparse.get(j, 0) for j in range(self.m)])
        return compute_cardinality(self.dense)

    def raw_dense_estimate(self) -> float:
        """密集数组上的原始估计（无任何修正），用于建立偏差表。"""
        if self.dense is None:
            return compute_cardinality_plus(
                self.m, self.m,
                [self.sparse.get(j, 0) for j in range(self.m)])
        return compute_raw_estimate(self.dense)

    def num_distinct_indices(self) -> int:
        """不同桶索引数（稀疏模式下等于条目数）。"""
        if self.dense is not None:
            return sum(1 for r in self.dense if r > 0)
        return len(self.sparse)

    def memory_bytes(self) -> int:
        """当前表示的实际内存占用。"""
        if self.dense is not None:
            return math.ceil(self.m * self.bits_per_register / 8)
        return compute_sparse_memory_bytes(len(self.sparse), self.p,
                                           self.bits_per_register)


def measure_small_cardinality(n: int, num_trials: int = 30) -> dict:
    """小基数下 HLL++ 与原版 HLL 的误差对比。"""
    plus_errs = []
    plain_errs = []
    plus_sparse_bytes = []
    plus_dense_bytes = []
    for trial in range(num_trials):
        offset = trial * 1_000_000
        hllp = HyperLogLogPlus(14)
        plain = HyperLogLog(14)
        for i in range(n):
            item = f"element_{offset + i}"
            hllp.add(item)
            plain.add(item)
        plus_errs.append(hllp.count() - n)
        plain_errs.append(plain.count() - n)
        plus_sparse_bytes.append(hllp.memory_bytes())
        plus_dense_bytes.append(math.ceil(16384 * 6 / 8))

    mean_plus = sum(plus_errs) / len(plus_errs)
    mean_plain = sum(plain_errs) / len(plain_errs)
    var_plus = sum((e - mean_plus) ** 2 for e in plus_errs) / max(1, len(plus_errs) - 1)
    var_plain = sum((e - mean_plain) ** 2 for e in plain_errs) / max(1, len(plain_errs) - 1)

    return {
        "true_cardinality": n,
        "num_trials": num_trials,
        "plus_mean_relative_error": mean_plus / n,
        "plus_mean_abs_relative_error": sum(abs(e) for e in plus_errs) / n / num_trials,
        "plus_std_relative_error": math.sqrt(var_plus) / n,
        "plain_mean_relative_error": mean_plain / n,
        "plain_mean_abs_relative_error": sum(abs(e) for e in plain_errs) / n / num_trials,
        "plain_std_relative_error": math.sqrt(var_plain) / n,
        "theoretical_sigma": compute_standard_error(2 ** 14),
        "plus_memory_bytes": sum(plus_sparse_bytes) // len(plus_sparse_bytes),
        "dense_memory_bytes": plus_dense_bytes[0],
    }


def measure_memory_curve(n_list: list) -> list:
    """稀疏内存随基数增长的曲线，并定位切换点。"""
    rows = []
    for n in n_list:
        hllp = HyperLogLogPlus(14)
        for i in range(n):
            hllp.add(f"element_{i}")
        rows.append({
            "n": n,
            "mode": hllp.mode,
            "memory_bytes": hllp.memory_bytes(),
            "dense_memory_bytes": math.ceil(16384 * 6 / 8),
            "switched_at": hllp.switched_at,
        })
    return rows


def measure_bias_correction(anchor_list: list, test_list: list,
                            num_trials: int = 20) -> dict:
    """偏差校正在 2.5m 到 5m 区间的效果。

    按论文的做法，偏差表本身也由实测得到：先在锚点基数上跑多次试验取平均
    原始估计，得到 (rawEstimate, bias) 表；再在测试基数上比较
    原始估计与用该表校正后的估计。
    """
    # 1. 建表：锚点基数的平均原始估计与真实值的差
    table = []
    for n in anchor_list:
        raws = []
        for trial in range(num_trials):
            offset = trial * 1_000_000
            hllp = HyperLogLogPlus(14)
            for i in range(n):
                hllp.add(f"element_{offset + i}")
            raws.append(hllp.raw_dense_estimate())
        table.append((sum(raws) / len(raws), sum(raws) / len(raws) - n))

    # 2. 测试：原始 vs 校正
    rows = []
    for n in test_list:
        raw_errs, corr_errs = [], []
        for trial in range(num_trials):
            offset = trial * 1_000_000
            hllp = HyperLogLogPlus(14)
            for i in range(n):
                hllp.add(f"element_{offset + i}")
            raw = compute_raw_estimate(hllp.dense)
            corr = compute_bias_corrected_estimate(raw, hllp.m, table)
            raw_errs.append(raw - n)
            corr_errs.append(corr - n)
        rows.append({
            "true_cardinality": n,
            "raw_mean_relative_error": sum(raw_errs) / len(raw_errs) / n,
            "raw_mean_abs_relative_error":
                sum(abs(e) for e in raw_errs) / n / num_trials,
            "corrected_mean_relative_error": sum(corr_errs) / len(corr_errs) / n,
            "corrected_mean_abs_relative_error":
                sum(abs(e) for e in corr_errs) / n / num_trials,
            "applied_bias": compute_bias(
                table[0][0], 16384, table),
        })
    return {"bias_table": [{"raw": r, "bias": b} for r, b in table],
            "test": rows}


def measure_large_cardinality(n: int, num_trials: int = 3) -> dict:
    """大基数（1M / 10M）下 64 位哈希的表现。"""
    errs = []
    for trial in range(num_trials):
        offset = trial * 1_000_000
        hllp = HyperLogLogPlus(14)
        for i in range(n):
            hllp.add(f"element_{offset + i}")
        errs.append(hllp.count() - n)
    mean_err = sum(errs) / len(errs)
    return {
        "true_cardinality": n,
        "num_trials": num_trials,
        "mean_estimate": n + mean_err,
        "mean_relative_error": mean_err / n,
        "mean_abs_relative_error": sum(abs(e) for e in errs) / n / num_trials,
        "mode": "dense",
        "theoretical_sigma": compute_standard_error(2 ** 14),
    }


def main():
    print("=== HyperLogLog++ 实测验证 ===\n")

    table = build_bias_table(16384)
    dense_bytes = math.ceil(16384 * 6 / 8)
    threshold = compute_sparse_threshold(16384)

    # 1. 小基数精度对比
    print(f"--- 小基数精度：HLL++ vs 原版 HLL（p=14, m=16384, 30 次试验）---")
    print(f"{'基数':>9}  {'HLL++ 相对误差':>16}  {'HLL++ 绝对误差':>16}  "
          f"{'原版相对误差':>14}  {'原版绝对误差':>14}  {'HLL++ 内存':>11}")
    card_rows = []
    for n in [100, 1000, 3000, 8000, 12000, 20000, 50000, 100000]:
        r = measure_small_cardinality(n, 30)
        card_rows.append(r)
        print(f"{n:>9,}  {r['plus_mean_relative_error']:>15.4%}  "
              f"{r['plus_mean_abs_relative_error']:>15.4%}  "
              f"{r['plain_mean_relative_error']:>13.4%}  "
              f"{r['plain_mean_abs_relative_error']:>13.4%}  "
              f"{r['plus_memory_bytes']:>10,} B")

    # 2. 内存曲线与切换点
    print(f"\n--- 稀疏内存曲线（密集固定 {dense_bytes:,} B，切换点 {threshold:,}）---")
    print(f"{'n':>9}  {'模式':>7}  {'实际内存':>10}  {'密集内存':>10}  "
          f"{'压缩比':>8}  {'切换发生于':>12}")
    mem_rows = measure_memory_curve(
        [100, 1000, 2000, 3000, 3072, 4000, 10000, 100000])
    for r in mem_rows:
        ratio = r["dense_memory_bytes"] / r["memory_bytes"]
        sw = f"{r['switched_at']:,}" if r["switched_at"] else "-"
        print(f"{r['n']:>9,}  {r['mode']:>7}  {r['memory_bytes']:>9,} B  "
              f"{r['dense_memory_bytes']:>9,} B  {ratio:>7.2f}x  {sw:>12}")

    # 3. 偏差校正
    print(f"\n--- 偏差校正效果（2.5m={int(2.5*16384):,} 到 5m={int(5*16384):,}，20 次试验）---")
    bias = measure_bias_correction(
        anchor_list=[35_000, 40_000, 45_000, 50_000, 55_000, 60_000,
                    65_000, 70_000, 75_000, 80_000, 85_000, 90_000],
        test_list=[40_960, 50_000, 60_000, 70_000, 81_920],
        num_trials=20)
    print("实测偏差表（锚点基数 -> 平均原始估计的偏差）:")
    print(f"{'锚点基数':>9}  {'平均原始估计':>13}  {'偏差':>9}  {'偏差占比':>9}")
    for row in bias["bias_table"]:
        n = row["raw"] - row["bias"]
        print(f"{n:>9,.0f}  {row['raw']:>13,.0f}  {row['bias']:>9,.0f}  "
              f"{row['bias']/n:>8.3%}")
    print(f"\n{'测试基数':>8}  {'原始相对误差':>14}  {'原始绝对误差':>14}  "
          f"{'校正后相对误差':>16}  {'校正后绝对误差':>16}")
    for r in bias["test"]:
        print(f"{r['true_cardinality']:>8,}  "
              f"{r['raw_mean_relative_error']:>13.4%}  "
              f"{r['raw_mean_abs_relative_error']:>13.4%}  "
              f"{r['corrected_mean_relative_error']:>15.4%}  "
              f"{r['corrected_mean_abs_relative_error']:>15.4%}")

    # 4. 大基数
    print("\n--- 大基数（64 位哈希）---")
    large_rows = []
    for n in [1_000_000, 10_000_000]:
        r = measure_large_cardinality(n, 3)
        large_rows.append(r)
        print(f"基数={n:>10,}  平均估计={r['mean_estimate']:>13,.0f}  "
              f"相对误差={r['mean_relative_error']:>8.4%}  "
              f"绝对误差={r['mean_abs_relative_error']:>8.4%}")

    result = {
        "meta": {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "python": sys.version.split()[0],
            "note": "自制迷你 HyperLogLog++，64 位哈希，含稀疏/密集与实测偏差表",
            "p": 14,
            "m": 16384,
            "sparse_threshold": threshold,
            "num_trials": 30,
        },
        "small_cardinality": card_rows,
        "memory_curve": mem_rows,
        "bias_correction": bias,
        "large_cardinality": large_rows,
    }

    out = os.path.join(os.path.dirname(__file__), "..", "results",
                       f"hyperloglog_plus_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print(f"\n结果写入 {out}")


if __name__ == "__main__":
    main()
