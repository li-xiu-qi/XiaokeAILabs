# -*- coding: utf-8 -*-
"""
布隆过滤器实测验证

用纯 Python 自制迷你布隆过滤器，实测假阳性率：
1. 插入 n 个元素后，查询不存在的元素，统计假阳性率
2. 对比不同 bits_per_key 的假阳性率
3. 验证最优 k 确实最小化假阳性率

方法：用位数组（bytearray）+ k 个哈希函数，插入元素，查询不存在的元素。
结果写入 results/bloom_filter_<timestamp>.json
"""
import os
import sys
import json
import math
import hashlib
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from bloom_filter_model import BloomFilterSpec, compute, compute_optimal_k, compute_fpr


class BloomFilter:
    """自制迷你布隆过滤器。"""
    def __init__(self, m: int, k: int):
        self.m = m
        self.k = k
        self.bits = bytearray((m + 7) // 8)  # 位数组

    def _hashes(self, item: str) -> list:
        """计算 k 个哈希值。用双重哈希模拟 k 个独立哈希函数。"""
        h1 = int(hashlib.md5(item.encode()).hexdigest(), 16)
        h2 = int(hashlib.sha1(item.encode()).hexdigest(), 16)
        return [(h1 + i * h2) % self.m for i in range(self.k)]

    def add(self, item: str):
        """插入元素。"""
        for pos in self._hashes(item):
            self.bits[pos // 8] |= (1 << (pos % 8))

    def contains(self, item: str) -> bool:
        """查询元素。返回 True 表示"可能存在"。"""
        for pos in self._hashes(item):
            if not (self.bits[pos // 8] & (1 << (pos % 8))):
                return False
        return True


def measure_fpr(n: int, bits_per_key: int, num_queries: int = 10000) -> dict:
    """实测假阳性率。插入 n 个元素，查询 num_queries 个不存在的元素。"""
    m = n * bits_per_key
    k = compute_optimal_k(m, n)
    bf = BloomFilter(m, k)

    # 插入 n 个元素
    for i in range(n):
        bf.add(f"item_{i}")

    # 查询不存在的元素
    false_positives = 0
    for i in range(num_queries):
        if bf.contains(f"nonexistent_{i}"):
            false_positives += 1

    actual_fpr = false_positives / num_queries
    theo_fpr = compute_fpr(m, n, k)

    return {
        "n": n,
        "m": m,
        "k": k,
        "num_queries": num_queries,
        "false_positives": false_positives,
        "actual_fpr": actual_fpr,
        "theoretical_fpr": theo_fpr,
    }


def measure_fpr_vs_bits(n: int, bits_per_key_list: list) -> list:
    """实测不同 bits_per_key 的假阳性率。"""
    results = []
    for bpk in bits_per_key_list:
        r = measure_fpr(n, bpk, num_queries=5000)
        results.append(r)
    return results


def measure_fpr_vs_k(n: int, bits_per_key: int, k_list: list) -> list:
    """实测不同 k 的假阳性率，验证最优 k。"""
    m = n * bits_per_key
    results = []
    for k in k_list:
        bf = BloomFilter(m, k)
        for i in range(n):
            bf.add(f"item_{i}")

        false_positives = 0
        for i in range(5000):
            if bf.contains(f"nonexistent_{i}"):
                false_positives += 1

        actual_fpr = false_positives / 5000
        theo_fpr = compute_fpr(m, n, k)
        results.append({
            "k": k,
            "actual_fpr": actual_fpr,
            "theoretical_fpr": theo_fpr,
        })
    return results


def main():
    print("=== 布隆过滤器实测验证 ===\n")

    n = 10_000

    # 1. 不同 bits_per_key 的假阳性率
    print("--- 不同 bits_per_key 的假阳性率 ---")
    bpk_results = measure_fpr_vs_bits(n, [4, 8, 10, 16, 24])
    print(f"{'bits/key':>10}  {'k':>3}  {'实测 FPR':>10}  {'理论 FPR':>10}")
    for r in bpk_results:
        print(f"{r['m']//n:>10}  {r['k']:>3}  {r['actual_fpr']:>10.4f}  {r['theoretical_fpr']:>10.4f}")

    # 2. 不同 k 的假阳性率（验证最优 k）
    print("\n--- 不同 k 的假阳性率（验证最优 k）---")
    optimal_k = compute_optimal_k(n * 10, n)
    k_list = sorted(set([1, 3, 5, 7, 9, optimal_k, 15, 20]))
    k_results = measure_fpr_vs_k(n, 10, k_list)
    print(f"{'k':>3}  {'实测 FPR':>10}  {'理论 FPR':>10}  {'标记'}")
    for r in k_results:
        marker = " <-- 最优" if r['k'] == optimal_k else ""
        print(f"{r['k']:>3}  {r['actual_fpr']:>10.4f}  {r['theoretical_fpr']:>10.4f}{marker}")

    # 汇总写入 JSON
    result = {
        "meta": {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "python": sys.version.split()[0],
            "note": "自制迷你布隆过滤器，实测假阳性率",
            "n": n,
        },
        "fpr_vs_bits_per_key": bpk_results,
        "fpr_vs_k": k_results,
        "optimal_k": optimal_k,
    }

    out = os.path.join(os.path.dirname(__file__), "..", "results",
                       f"bloom_filter_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n结果写入 {out}")


if __name__ == "__main__":
    main()
