# -*- coding: utf-8 -*-
"""
哈希索引实测验证

验证平均探测次数公式。实现一个简单的哈希表（拉链法 + 开放寻址），
插入 n 个键，测查询时的平均探测次数，对比理论公式。

结果写入 results/hash_<timestamp>.json
"""
import os
import sys
import json
import time
import random
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from hash_model import HashSpec, compute, avg_probes_chaining, avg_probes_linear, avg_probes_double


class HashTableChaining:
    """拉链法哈希表"""

    def __init__(self, num_buckets):
        self.buckets = [[] for _ in range(num_buckets)]
        self.size = 0

    def _hash(self, key):
        # 用 key * 大常数 + 异或，让连续整数产生冲突，模拟真实哈希行为
        h = (key * 2654435761) & 0xFFFFFFFF
        h ^= h >> 15
        return h % len(self.buckets)

    def insert(self, key):
        idx = self._hash(key)
        for k in self.buckets[idx]:
            if k == key:
                return
        self.buckets[idx].append(key)
        self.size += 1

    def lookup_probes(self, key):
        """返回查找 key 需要的探测次数"""
        idx = self._hash(key)
        probes = 0
        for k in self.buckets[idx]:
            probes += 1
            if k == key:
                return probes
        return probes


class HashTableOpenAddressing:
    """开放寻址哈希表（线性/双重探测）"""

    def __init__(self, num_buckets, probe="linear"):
        self.table = [None] * num_buckets
        self.size = 0
        self.probe = probe
        self.num_buckets = num_buckets

    def _hash1(self, key):
        h = (key * 2654435761) & 0xFFFFFFFF
        h ^= h >> 15
        return h % self.num_buckets

    def _hash2(self, key):
        """双重哈希的第二个哈希函数。步长取奇数且小于桶数，保证与桶数互质。"""
        h = (key * 2654435761) & 0xFFFFFFFF
        h ^= h >> 13
        # 取奇数步长，范围 [1, num_buckets-1]
        step = (h % self.num_buckets) | 1
        if step >= self.num_buckets:
            step = 1
        return step

    def _probe_sequence(self, key):
        """生成探测序列"""
        h1 = self._hash1(key)
        if self.probe == "linear":
            i = 0
            while True:
                yield (h1 + i) % self.num_buckets
                i += 1
        else:  # double
            h2 = self._hash2(key)
            i = 0
            while True:
                yield (h1 + i * h2) % self.num_buckets
                i += 1

    def insert(self, key):
        probes = 0
        for idx in self._probe_sequence(key):
            probes += 1
            if probes > self.num_buckets * 2:
                raise RuntimeError(f"探测次数超限 {probes}，哈希函数可能有死循环")
            if self.table[idx] is None:
                self.table[idx] = key
                self.size += 1
                return

    def lookup_probes(self, key):
        """返回查找 key 需要的探测次数"""
        probes = 0
        for idx in self._probe_sequence(key):
            probes += 1
            if probes > self.num_buckets * 2:
                raise RuntimeError(f"探测次数超限 {probes}，哈希函数可能有死循环")
            if self.table[idx] == key:
                return probes
            if self.table[idx] is None:
                return probes
        return probes


def measure_probes(n, load_factor, method="chaining", probe="linear"):
    """实测平均探测次数"""
    # 桶数略大于 n/load_factor，留出空位，避免开放寻址在高压负载下绕圈
    raw_buckets = max(3, int(n / load_factor) + 2)
    if raw_buckets % 2 == 0:
        raw_buckets += 1
    num_buckets = raw_buckets
    keys = list(range(n))
    random.seed(42)
    random.shuffle(keys)

    if method == "chaining":
        ht = HashTableChaining(num_buckets)
    else:
        ht = HashTableOpenAddressing(num_buckets, probe)

    for k in keys:
        ht.insert(k)

    # 测查找这些键的探测次数
    total_probes = sum(ht.lookup_probes(k) for k in keys)
    avg = total_probes / n
    return avg, num_buckets


def main():
    print("=== 哈希索引探测次数实测 ===\n")
    results = []

    for lf in [0.5, 0.75, 0.9]:
        # 拉链法
        avg_chain, nb = measure_probes(n=1000, load_factor=lf, method="chaining")
        theo_chain = avg_probes_chaining(lf)

        # 开放寻址线性
        avg_linear, _ = measure_probes(n=1000, load_factor=lf, method="open_addressing", probe="linear")
        theo_linear = avg_probes_linear(lf)

        # 开放寻址双重
        avg_double, _ = measure_probes(n=1000, load_factor=lf, method="open_addressing", probe="double")
        theo_double = avg_probes_double(lf)

        results.append({
            "load_factor": lf,
            "chaining": {"measured": avg_chain, "theoretical": theo_chain},
            "linear": {"measured": avg_linear, "theoretical": theo_linear},
            "double": {"measured": avg_double, "theoretical": theo_double},
        })

        print(f"α={lf:.2f}  桶数={nb}")
        print(f"  拉链法: 实测 {avg_chain:.2f}  理论 {theo_chain:.2f}")
        print(f"  线性探测: 实测 {avg_linear:.2f}  理论 {theo_linear:.2f}")
        print(f"  双重哈希: 实测 {avg_double:.2f}  理论 {theo_double:.2f}")
        print()

    result = {
        "meta": {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "python": sys.version.split()[0],
            "note": "实测平均探测次数 vs 理论公式",
        },
        "probe_measurements": results,
    }
    out = os.path.join(os.path.dirname(__file__), "..", "results",
                       f"hash_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"结果写入 {out}")


if __name__ == "__main__":
    main()
