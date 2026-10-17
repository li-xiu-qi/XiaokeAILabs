# -*- coding: utf-8 -*-
"""
布谷鸟哈希实测验证

用纯 Python 自制迷你布谷鸟哈希（两个数组 + 递归踢出），实测：
1. 不同负载因子（0.3/0.4/0.45/0.5）下的插入成功率
2. 最大踢出链长度（递归深度），触发 rehash 的边界
3. 查找延迟（O(1) 最坏，只查两个位置）

方法：两个等长数组 t1、t2，两个独立哈希函数 h1、h2。
插入冲突时递归踢出，设最大踢出次数 MAX_KICKS，超限判定插入失败。
结果写入 results/cuckoo_hashing_<timestamp>.json
"""
import os
import sys
import json
import time
import random
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from cuckoo_hashing_model import CuckooSpec, compute, compute_table_capacity


MAX_KICKS = 500  # 最大踢出次数，超限触发 rehash（工程标准做法）


class CuckooHashTable:
    """自制迷你布谷鸟哈希。两个数组 + 递归踢出。"""

    def __init__(self, table_capacity: int):
        self.cap = table_capacity
        self.t1 = [None] * table_capacity
        self.t2 = [None] * table_capacity
        self.size = 0
        self.max_kick_chain = 0      # 记录出现过的最大踢出链长度
        self.total_kicks = 0         # 累计踢出次数

    def _h1(self, key: int) -> int:
        h = (key * 2654435761) & 0xFFFFFFFF
        h ^= h >> 15
        return h % self.cap

    def _h2(self, key: int) -> int:
        # 用不同种子/位移，保证 h1、h2 独立，避免强相关导致成环
        h = (key * 40503) & 0xFFFFFFFF
        h ^= h >> 13
        return h % self.cap

    def _other(self, key: int, which: int) -> int:
        """返回 key 的另一个候选位置下标（跨表）。"""
        return self._h2(key) if which == 1 else self._h1(key)

    def insert(self, key: int) -> bool:
        """
        插入 key。返回 True 成功，False 失败（超 MAX_KICKS）。
        先查两个候选位置是否已存在该 key（去重）。
        """
        if self.lookup(key):
            return True

        cur = key
        which = 1  # 先试 t1
        chain = 0
        for _ in range(MAX_KICKS):
            pos = self._h1(cur) if which == 1 else self._h2(cur)
            table = self.t1 if which == 1 else self.t2
            occupant = table[pos]
            if occupant is None:
                table[pos] = cur
                self.size += 1
                if chain > self.max_kick_chain:
                    self.max_kick_chain = chain
                return True
            # 踢出 occupant，自己占位
            table[pos] = cur
            self.total_kicks += 1
            chain += 1
            cur = occupant
            which = 2 if which == 1 else 1
        # 超限：插入失败，需 rehash
        if chain > self.max_kick_chain:
            self.max_kick_chain = chain
        return False

    def lookup(self, key: int) -> bool:
        """查找 key。只查两个候选位置，O(1) 最坏。"""
        return self.t1[self._h1(key)] == key or self.t2[self._h2(key)] == key

    def delete(self, key: int) -> bool:
        """删除 key。直接清空，O(1)。"""
        p1, p2 = self._h1(key), self._h2(key)
        if self.t1[p1] == key:
            self.t1[p1] = None
            self.size -= 1
            return True
        if self.t2[p2] == key:
            self.t2[p2] = None
            self.size -= 1
            return True
        return False


def measure_load_factor(actual_load: float, total_slots: int = 20000,
                        trials: int = 5) -> dict:
    """
    实测给定实际负载因子下的插入成功率与踢出链长度。

    固定两表合计槽位 total_slots，插入 n = round(actual_load × total_slots) 个元素。
    这样实际负载因子被直接控制，可在 0.5 阈值附近观察插入失败。
    每个负载跑 trials 次（不同 seed）统计失败率。
    """
    cap = total_slots // 2
    n = round(actual_load * total_slots)
    fail_count = 0
    max_chain_overall = 0
    total_kicks_overall = 0
    success_count = 0
    for seed in range(trials):
        table = CuckooHashTable(cap)
        keys = list(range(n))
        random.seed(seed)
        random.shuffle(keys)
        ok = True
        for k in keys:
            if not table.insert(k):
                ok = False
                break
        if ok:
            success_count += 1
        else:
            fail_count += 1
        max_chain_overall = max(max_chain_overall, table.max_kick_chain)
        total_kicks_overall += table.total_kicks

    actual = success_count  # 成功插入的试验里元素全部落地
    return {
        "target_load": actual_load,
        "total_slots": total_slots,
        "table_capacity": cap,
        "n": n,
        "trials": trials,
        "success_trials": success_count,
        "fail_trials": fail_count,
        "insert_success_rate": success_count / trials,
        "max_kick_chain": max_chain_overall,
        "avg_total_kicks": total_kicks_overall / trials,
    }


def measure_lookup_latency(total_slots: int = 20000, actual_load: float = 0.45,
                        num_lookups: int = 50000) -> dict:
    """
    实测查找延迟。按实际负载填充后，重复查找已有 key，统计平均耗时（纳秒/次）。
    """
    cap = total_slots // 2
    n = round(actual_load * total_slots)
    table = CuckooHashTable(cap)
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
        "actual_load": actual_load,
        "num_lookups": num_lookups,
        "avg_lookup_ns": elapsed_ns / num_lookups,
        "hit_rate": hits / num_lookups,
    }


def main():
    print("=== 布谷鸟哈希实测验证 ===\n")

    # 固定两表合计槽位，按实际负载因子填充
    total_slots = 20000  # 每表 10000 槽
    trials = 5

    # 1. 扫描实际负载 0.30→0.52 的插入成功率与踢出链
    print(f"--- 不同实际负载的插入成功率（合计槽位={total_slots:,}, 每负载 {trials} 次试验）---")
    load_results = []
    for load in [0.30, 0.40, 0.45, 0.48, 0.50, 0.52, 0.55, 0.60, 0.70]:
        r = measure_load_factor(load, total_slots=total_slots, trials=trials)
        load_results.append(r)
        print(f"负载={load:.2f}  n={r['n']:>5,}  "
              f"成功率={r['insert_success_rate']*100:>5.1f}%  "
              f"失败={r['fail_trials']}/{r['trials']}  "
              f"最大踢出链={r['max_kick_chain']}  "
              f"平均踢出次数={r['avg_total_kicks']:.0f}")

    # 2. 查找延迟（负载 0.45）
    print("\n--- 查找延迟（负载 0.45, 5 万次查找）---")
    lat = measure_lookup_latency(total_slots=total_slots, actual_load=0.45,
                                 num_lookups=50000)
    print(f"平均查找延迟={lat['avg_lookup_ns']:.0f} ns  命中率={lat['hit_rate']*100:.1f}%")

    # 汇总写入 JSON
    result = {
        "meta": {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "python": sys.version.split()[0],
            "note": "自制迷你布谷鸟哈希，按实际负载填充，实测插入成功率与踢出链长度",
            "max_kicks": MAX_KICKS,
            "total_slots": total_slots,
            "trials": trials,
        },
        "load_factor_results": load_results,
        "lookup_latency": lat,
    }

    out = os.path.join(os.path.dirname(__file__), "..", "results",
                       f"cuckoo_hashing_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n结果写入 {out}")


if __name__ == "__main__":
    main()
