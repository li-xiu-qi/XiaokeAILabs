# -*- coding: utf-8 -*-
"""
HyperLogLog 实测验证

用纯 Python 自制迷你 HyperLogLog，实测三件事：
1. 不同基数（1K/10K/100K/1M）的估计误差，验证 σ = 1.04/√m
2. 不同 m（2^10/2^14/2^16）的精度，验证桶数越多误差越小
3. 小基数修正（线性计数）的有效性

实现要点：
- 64 位哈希，前 p 位选桶，剩余位中前导零个数 +1 为寄存器值
- 寄存器只增不减（取历史最大）
- 估计：E = α_m × m² / Σ(2^(-M[j]))
- 小基数：E < 2.5m 且有空桶时用线性计数 E* = m × ln(m/V)
- 大基数：E > 2^32/30 时做饱和修正

结果写入 results/hyperloglog_<timestamp>.json
"""
import os
import sys
import json
import hashlib
import math
import random
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from hyperloglog_model import (
    HyperLogLogSpec, compute,
    compute_alpha, compute_standard_error,
    compute_raw_estimate, compute_cardinality,
    compute_linear_counting_estimate,
)


class HyperLogLog:
    """自制迷你 HyperLogLog。"""

    def __init__(self, p: int = 14, bits_per_register: int = 6):
        self.p = p
        self.m = 2 ** p
        self.bits_per_register = bits_per_register
        self.registers = [0] * self.m  # 寄存器值，取历史最大
        self.register_mask = (1 << bits_per_register) - 1

    def _hash64(self, item: str) -> int:
        """64 位哈希值。"""
        h = hashlib.sha1(item.encode("utf-8")).digest()
        return int.from_bytes(h[:8], "big")

    def add(self, item: str):
        """插入元素。更新对应桶的寄存器值（只增不减）。"""
        h = self._hash64(item)
        # 前 p 位选桶
        idx = h >> (64 - self.p)
        # 剩余位：取低 64-p 位，数前导零
        w = h & ((1 << (64 - self.p)) - 1)
        if w == 0:
            rank = 64 - self.p  # 剩余位全零，前导零个数为 64-p
        else:
            rank = (64 - self.p) - w.bit_length()
        rank += 1  # 寄存器值 = 前导零个数 + 1
        # 截断到寄存器位数
        rank = min(rank, self.register_mask)
        if rank > self.registers[idx]:
            self.registers[idx] = rank

    def count(self) -> float:
        """基数估计（含修正）。"""
        return compute_cardinality(self.registers)

    def raw_count(self) -> float:
        """原始估计（无修正）。"""
        return compute_raw_estimate(self.registers)

    def num_empty(self) -> int:
        """空桶数量。"""
        return sum(1 for r in self.registers if r == 0)

    def memory_bytes(self) -> int:
        """内存占用。"""
        return math.ceil(self.m * self.bits_per_register / 8)


def measure_cardinality(n: int, p: int = 14, num_trials: int = 30) -> dict:
    """实测给定基数的估计误差，多次试验取平均。"""
    errors = []
    raw_errors = []
    for trial in range(num_trials):
        hll = HyperLogLog(p)
        # 用随机偏移避免哈希模式干扰
        offset = trial * 1_000_000
        for i in range(n):
            hll.add(f"element_{offset + i}")
        est = hll.count()
        raw = hll.raw_count()
        errors.append(est - n)
        raw_errors.append(raw - n)

    mean_err = sum(errors) / len(errors)
    abs_errs = [abs(e) for e in errors]
    mean_abs = sum(abs_errs) / len(abs_errs)

    # 样本标准差（估计误差的标准差）
    var = sum((e - mean_err) ** 2 for e in errors) / max(1, len(errors) - 1)
    std = math.sqrt(var)

    sigma_theo = compute_standard_error(2 ** p)

    # 统计有多少次落在理论 σ 区间内
    within_sigma = sum(1 for e in errors if abs(e - mean_err) <= sigma_theo * n) / len(errors)

    return {
        "true_cardinality": n,
        "p": p,
        "m": 2 ** p,
        "num_trials": num_trials,
        "mean_estimate": n + mean_err,
        "mean_relative_error": mean_err / n,
        "mean_absolute_relative_error": mean_abs / n,
        "std_relative_error": std / n,
        "theoretical_sigma": sigma_theo,
        "within_sigma_ratio": within_sigma,
        "memory_bytes": HyperLogLog(p).memory_bytes(),
    }


def measure_precision_vs_p(n: int, p_list: list, num_trials: int = 20) -> list:
    """实测不同 p（桶数）的精度。"""
    results = []
    for p in p_list:
        r = measure_cardinality(n, p, num_trials)
        results.append(r)
    return results


def main():
    print("=== HyperLogLog 实测验证 ===\n")

    # 1. 不同基数的估计误差（p=14）
    print("--- 不同基数的估计误差（p=14, m=16384, 30 次试验）---")
    card_results = []
    for n in [1_000, 10_000, 100_000, 1_000_000]:
        r = measure_cardinality(n, 14, 30)
        card_results.append(r)
        print(f"基数={n:>9,}  平均估计={r['mean_estimate']:>12,.0f}  "
              f"相对误差={r['mean_relative_error']:>8.4%}  "
              f"实测标准差={r['std_relative_error']:.4%}  "
              f"理论 σ={r['theoretical_sigma']:.4%}")

    # 2. 不同 p 的精度
    print("\n--- 不同 p（桶数）的精度（基数=100K, 20 次试验）---")
    p_results = measure_precision_vs_p(100_000, [10, 12, 14, 16], 20)
    print(f"{'p':>3}  {'m':>7}  {'理论 σ':>9}  {'实测标准差':>12}  "
          f"{'平均绝对相对误差':>18}  {'内存':>10}")
    for r in p_results:
        print(f"{r['p']:>3}  {r['m']:>7,}  {r['theoretical_sigma']:>9.4%}  "
              f"{r['std_relative_error']:>12.4%}  "
              f"{r['mean_absolute_relative_error']:>18.4%}  "
              f"{r['memory_bytes']:>9,} B")

    result = {
        "meta": {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "python": sys.version.split()[0],
            "note": "自制迷你 HyperLogLog，64 位哈希，含小基数与大基数修正",
            "num_trials": 30,
        },
        "cardinality_vs_error": card_results,
        "precision_vs_p": p_results,
    }

    out = os.path.join(os.path.dirname(__file__), "..", "results",
                       f"hyperloglog_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print(f"\n结果写入 {out}")


if __name__ == "__main__":
    main()
