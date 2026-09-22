# -*- coding: utf-8 -*-
"""
可扩展哈希（Extendible Hashing）实测验证

用纯 Python 自制迷你可扩展哈希（目录 + 桶），实测：
1. 插入 n 个元素，统计目录翻倍次数、桶分裂次数
2. 不同桶容量（2/4/8）下的空间利用率与目录开销
3. 平均查找延迟（目录索引 + 桶内线性查找）

方法：
- 目录：2^d 个槽，每槽指向一个桶，全局深度 d
- 桶：固定容量 B，局部深度 d'，元素存 (key, value)
- 插入：用 key 哈希值的低 d 位索引目录；桶满则分裂；
  若桶局部深度 == 全局深度，先目录翻倍，再分裂
结果写入 results/extendible_hashing_<timestamp>.json
"""
import os
import sys
import json
import time
import random
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from extendible_hashing_model import (
    ExtendibleSpec, compute, estimate_global_depth, estimate_splits,
)


class Bucket:
    """桶：固定容量，局部深度。"""

    def __init__(self, capacity: int, local_depth: int):
        self.capacity = capacity
        self.local_depth = local_depth
        self.entries = {}  # key -> value（用 dict 便于 O(1) 桶内查找）

    def is_full(self) -> bool:
        return len(self.entries) >= self.capacity


class ExtendibleHashTable:
    """自制迷你可扩展哈希。"""

    def __init__(self, bucket_capacity: int, initial_global_depth: int = 1):
        self.bucket_capacity = bucket_capacity
        self.global_depth = initial_global_depth
        # 目录：2^d 个槽，每槽指向一个 Bucket
        first = Bucket(bucket_capacity, initial_global_depth)
        self.directory = [first] * (2 ** initial_global_depth)
        # 统计
        self.directory_doubles = 0
        self.bucket_splits = 0
        self.size = 0

    def _bucket_index(self, key: int) -> int:
        """用 key 哈希值的低 global_depth 位索引目录。"""
        h = hash(key) & 0x7FFFFFFF  # 取正数
        return h & ((1 << self.global_depth) - 1)

    def _hash_bits(self, key: int, depth: int) -> int:
        """取 key 哈希值的低 depth 位。"""
        h = hash(key) & 0x7FFFFFFF
        return h & ((1 << depth) - 1)

    def insert(self, key: int, value=True) -> bool:
        """插入 key。桶满则分裂，必要时目录翻倍。返回 True 成功。

        value 默认 True（不用 None，避免与 lookup 的存在性判断冲突）。
        """
        if self.lookup(key):
            return True  # 已存在

        idx = self._bucket_index(key)
        bucket = self.directory[idx]

        if not bucket.is_full():
            bucket.entries[key] = value
            self.size += 1
            return True

        # 桶满：分裂
        self._split_bucket(bucket)
        # 分裂后重试插入（递归一次即可，分裂保证有空位）
        return self.insert(key, value)

    def _split_bucket(self, bucket: Bucket):
        """分裂桶。局部深度+1；若追平全局深度则目录先翻倍。"""
        if bucket.local_depth == self.global_depth:
            self._double_directory()

        bucket.local_depth += 1
        self.bucket_splits += 1
        new_depth = bucket.local_depth
        # 区分位：新深度的最高位（第 new_depth-1 位，0 起）
        split_bit = 1 << (new_depth - 1)

        # 按区分位把旧桶元素分流到旧桶与新桶
        old_entries = bucket.entries
        bucket.entries = {}
        new_bucket = Bucket(self.bucket_capacity, new_depth)
        for k, v in old_entries.items():
            if self._hash_bits(k, new_depth) & split_bit:
                new_bucket.entries[k] = v
            else:
                bucket.entries[k] = v

        # 更新目录中指向旧桶的槽：按区分位决定指向旧桶还是新桶
        for i in range(len(self.directory)):
            if self.directory[i] is bucket:
                if self._hash_bits(i, new_depth) & split_bit:
                    self.directory[i] = new_bucket
                # 否则保持指向旧桶

    def _double_directory(self):
        """目录翻倍：每个槽复制一份，全局深度+1。"""
        self.global_depth += 1
        self.directory_doubles += 1
        self.directory = self.directory + self.directory[:]

    def lookup(self, key: int) -> bool:
        """查找 key 是否存在。目录索引一次 + 桶内查找。"""
        idx = self._bucket_index(key)
        return key in self.directory[idx].entries

    def num_buckets(self) -> int:
        """去重后的桶数量。"""
        seen = set()
        for b in self.directory:
            seen.add(id(b))
        return len(seen)


def measure_extendible(n: int, bucket_capacity: int) -> dict:
    """实测插入 n 个元素后的目录翻倍、桶分裂、利用率。"""
    table = ExtendibleHashTable(bucket_capacity, initial_global_depth=1)
    keys = list(range(n))
    random.seed(42)
    random.shuffle(keys)
    for k in keys:
        table.insert(k)

    num_buckets = table.num_buckets()
    utilization = table.size / (num_buckets * bucket_capacity)
    # 确定性规律校验：桶满即分裂实现里，分裂次数 = 桶数 - 1
    splits_eq_buckets_minus_1 = (table.bucket_splits == num_buckets - 1)

    return {
        "n": n,
        "bucket_capacity": bucket_capacity,
        "global_depth": table.global_depth,
        "directory_slots": 2 ** table.global_depth,
        "num_buckets": num_buckets,
        "directory_doubles": table.directory_doubles,
        "bucket_splits": table.bucket_splits,
        "space_utilization": utilization,
        "splits_eq_buckets_minus_1": splits_eq_buckets_minus_1,
    }


def measure_lookup_latency(n: int, bucket_capacity: int,
                           num_lookups: int = 50000) -> dict:
    """实测查找延迟（目录索引 + 桶内查找）。"""
    table = ExtendibleHashTable(bucket_capacity, initial_global_depth=1)
    for k in range(n):
        table.insert(k)

    lookup_keys = [random.randint(0, n - 1) for _ in range(num_lookups)]
    start = time.perf_counter_ns()
    hits = 0
    for k in lookup_keys:
        if table.lookup(k):
            hits += 1
    elapsed_ns = time.perf_counter_ns() - start

    return {
        "n": n,
        "bucket_capacity": bucket_capacity,
        "num_lookups": num_lookups,
        "avg_lookup_ns": elapsed_ns / num_lookups,
        "hit_rate": hits / num_lookups,
    }


def main():
    print("=== 可扩展哈希实测验证 ===\n")

    n = 20_000

    # 1. 不同桶容量的目录翻倍/桶分裂/利用率
    print(f"--- 不同桶容量（n={n:,}）---")
    cap_results = []
    for bc in [2, 4, 8]:
        r = measure_extendible(n, bc)
        theo = compute(ExtendibleSpec(n=n, bucket_capacity=bc))
        r["theoretical_global_depth"] = theo.global_depth
        r["theoretical_directory_slots"] = theo.directory_slots
        r["theoretical_splits_prob"] = theo.est_splits
        r["theoretical_splits_deterministic"] = theo.est_splits_deterministic
        cap_results.append(r)
        print(f"桶容量={bc}  全局深度={r['global_depth']}（理论 {theo.global_depth}）  "
              f"目录槽={r['directory_slots']:,}（理论 {theo.directory_slots:,}）")
        print(f"         桶数={r['num_buckets']:,}  目录翻倍={r['directory_doubles']}  "
              f"桶分裂={r['bucket_splits']:,}"
              f"（确定 {theo.est_splits_deterministic:,} / 概率 {theo.est_splits:,.0f}）  "
              f"分裂=桶数-1:{r['splits_eq_buckets_minus_1']}  "
              f"利用率={r['space_utilization']:.2f}")

    # 2. 不同桶容量的查找延迟
    print("\n--- 查找延迟（n=5000, 5 万次查找）---")
    lat_results = []
    for bc in [2, 4, 8]:
        lat = measure_lookup_latency(5000, bc, num_lookups=50000)
        lat_results.append(lat)
        print(f"桶容量={bc}  平均查找延迟={lat['avg_lookup_ns']:.0f} ns  "
              f"命中率={lat['hit_rate']*100:.1f}%")

    # 汇总写入 JSON
    result = {
        "meta": {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "python": sys.version.split()[0],
            "note": "自制迷你可扩展哈希，实测目录翻倍/桶分裂/查找延迟",
            "n": n,
        },
        "bucket_capacity_results": cap_results,
        "lookup_latency": lat_results,
    }

    out = os.path.join(os.path.dirname(__file__), "..", "results",
                       f"extendible_hashing_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n结果写入 {out}")


if __name__ == "__main__":
    main()
