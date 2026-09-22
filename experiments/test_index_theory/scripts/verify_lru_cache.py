# -*- coding: utf-8 -*-
"""
LRU 缓存实测验证

用纯 Python 自制迷你 LRU 缓存，实测命中率：
1. 生成访问序列（均匀分布 + Zipf 分布）
2. 运行 LRU 缓存，统计命中率
3. 对比不同容量和 Zipf 系数的命中率

方法：用 OrderedDict 实现 LRU，生成访问序列，统计命中率。
结果写入 results/lru_cache_<timestamp>.json
"""
import os
import sys
import json
import random
import time
from collections import OrderedDict
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from lru_cache_model import LRUSpec, compute, compute_expected_hit_rate


class LRUCache:
    """自制迷你 LRU 缓存。"""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0

    def access(self, key: int):
        """访问一个键。返回是否命中。"""
        if key in self.cache:
            # 命中，移到头部（最近使用）
            self.cache.move_to_end(key)
            self.hits += 1
            return True
        else:
            # 未命中，插入
            self.misses += 1
            self.cache[key] = True
            if len(self.cache) > self.capacity:
                # 淘汰尾部（最久未使用）
                self.cache.popitem(last=False)
            return False


def generate_uniform_access(key_space: int, length: int, seed: int = 42) -> list:
    """生成均匀分布的访问序列。"""
    random.seed(seed)
    return [random.randint(0, key_space - 1) for _ in range(length)]


def generate_zipf_access(key_space: int, length: int, s: float, seed: int = 42) -> list:
    """生成 Zipf 分布的访问序列。s 越大，访问越集中。"""
    random.seed(seed)
    # 计算 Zipf 权重
    weights = [1.0 / (i + 1) ** s for i in range(key_space)]
    total = sum(weights)
    probabilities = [w / total for w in weights]
    return random.choices(range(key_space), weights=probabilities, k=length)


def measure_hit_rate(capacity: int, access_sequence: list) -> dict:
    """实测命中率。运行 LRU 缓存，统计命中率。"""
    cache = LRUCache(capacity)
    for key in access_sequence:
        cache.access(key)
    total = cache.hits + cache.misses
    return {
        "capacity": capacity,
        "total_accesses": total,
        "hits": cache.hits,
        "misses": cache.misses,
        "actual_hit_rate": cache.hits / total,
    }


def main():
    print("=== LRU 缓存实测验证 ===\n")

    key_space = 10_000
    access_length = 100_000

    # 1. 均匀分布，不同容量
    print("--- 均匀分布，不同容量 ---")
    uniform_seq = generate_uniform_access(key_space, access_length)
    print(f"{'容量':>8}  {'实测命中率':>12}  {'理论命中率':>12}")
    uniform_results = []
    for capacity in [100, 500, 1000, 2000, 5000]:
        r = measure_hit_rate(capacity, uniform_seq)
        spec = LRUSpec(capacity=capacity, key_space=key_space, zipf_s=0.0)
        theo_rate = compute_expected_hit_rate(spec)
        r["theoretical_hit_rate"] = theo_rate
        uniform_results.append(r)
        print(f"{capacity:>8}  {r['actual_hit_rate']:>12.2%}  {theo_rate:>12.2%}")

    # 2. Zipf 分布，不同 s
    print("\n--- Zipf 分布，不同 s（容量=1000）---")
    print(f"{'Zipf s':>8}  {'实测命中率':>12}  {'理论命中率':>12}")
    zipf_results = []
    for s in [0.0, 0.5, 1.0, 1.5, 2.0]:
        seq = generate_zipf_access(key_space, access_length, s)
        r = measure_hit_rate(1000, seq)
        spec = LRUSpec(capacity=1000, key_space=key_space, zipf_s=s)
        theo_rate = compute_expected_hit_rate(spec)
        r["zipf_s"] = s
        r["theoretical_hit_rate"] = theo_rate
        zipf_results.append(r)
        print(f"{s:>8}  {r['actual_hit_rate']:>12.2%}  {theo_rate:>12.2%}")

    # 汇总写入 JSON
    result = {
        "meta": {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "python": sys.version.split()[0],
            "note": "自制迷你 LRU 缓存，实测命中率",
            "key_space": key_space,
            "access_length": access_length,
        },
        "uniform_distribution": uniform_results,
        "zipf_distribution": zipf_results,
    }

    out = os.path.join(os.path.dirname(__file__), "..", "results",
                       f"lru_cache_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n结果写入 {out}")


if __name__ == "__main__":
    main()
