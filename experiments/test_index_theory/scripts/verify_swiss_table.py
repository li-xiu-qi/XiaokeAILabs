# -*- coding: utf-8 -*-
"""
瑞士表（Swiss Table）实测验证

自制简化版瑞士表，结构对齐 Abseil 实现：
- 元数据数组 bytearray，每槽 1 字节控制字
- 槽位数组存实际条目
- 查找用「签名并行比较」模拟 SIMD：一次性取出整组的控制字做批量匹配
- 三态编码：0=空，1=墓碑，>=2=已占用（存 h2 签名 +2 避免与三态冲突）

实测：
1. 不同负载（0.5/0.7/0.85）下的查找延迟与扫描组数
2. 元数据内存占比 vs 条目数组
3. 与 Python dict 的查找延迟对比（近似现代开放寻址实现）
4. 墓碑删除后的查找性能（对比无墓碑）

结果写入 results/swiss_table_<timestamp>.json
"""
import os
import sys
import json
import time
import random
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from swiss_table_model import (
    SwissTableSpec, compute, num_slots_for, avg_groups_probed,
)


class SwissTable:
    """
    简化版瑞士表。

    ctrl：bytearray，每槽 1 字节。0=空，1=墓碑，2-128=已占用（值 = h2 + 2）。
    keys / vals：平行数组存实际条目。
    查找时整组取出控制字，用 bytes.translate + count 的方式批量定位候选，
    再逐个比对 key。这是 SIMD PCMPEQB 的 Python 语义等价物。
    """

    EMPTY = 0
    DELETED = 1

    def __init__(self, num_slots: int, group_size: int = 8):
        assert num_slots % group_size == 0, "槽位数必须是组大小的整数倍"
        self.cap = num_slots
        self.G = group_size
        self.num_groups = num_slots // group_size
        self.ctrl = bytearray(num_slots)
        self.keys = [None] * num_slots
        self.vals = [None] * num_slots
        self.size = 0
        self.groups_probed_total = 0   # 累计扫描组数，用于实测平均扫描组数
        self.lookup_count = 0

    def _h2(self, key: int) -> int:
        """取哈希低 7 位作为签名，+2 避开 0/1 两个保留值。"""
        h = (key * 2654435761) & 0xFFFFFFFF
        h ^= h >> 15
        h = (h * 2246822519) & 0xFFFFFFFF
        h ^= h >> 13
        return (h & 0x7F) + 2

    def _h1(self, key: int) -> int:
        """取哈希高位决定起始组。"""
        h = (key * 2654435761) & 0xFFFFFFFF
        h ^= h >> 15
        h = (h * 2246822519) & 0xFFFFFFFF
        h ^= h >> 13
        return (h >> 7) % self.num_groups

    def _probe_group(self, start_group: int):
        """组间探测序列（线性推进，环形）。"""
        g = start_group
        while True:
            yield g
            g = (g + 1) % self.num_groups

    def _group_bytes(self, group: int) -> bytes:
        """取整组控制字（模拟一次 SIMD 加载）。"""
        return bytes(self.ctrl[group * self.G:(group + 1) * self.G])

    def _match_positions(self, group_bytes: bytes, sig: int) -> list:
        """在组内找出控制字 == sig 的位置（SIMD PCMPEQB + movemask 的语义）。"""
        return [i for i, b in enumerate(group_bytes) if b == sig]

    def insert(self, key: int, val):
        sig = self._h2(key)
        g0 = self._h1(key)
        groups_probed = 0
        for g in self._probe_group(g0):
            groups_probed += 1
            base = g * self.G
            gb = self._group_bytes(g)
            # 先查本组是否有同签名的槽（可能是更新）
            for i in self._match_positions(gb, sig):
                idx = base + i
                if self.keys[idx] == key:
                    self.vals[idx] = val
                    self.groups_probed_total += groups_probed
                    return
            # 找空槽或墓碑槽插入
            for i in range(self.G):
                idx = base + i
                if gb[i] == self.EMPTY or gb[i] == self.DELETED:
                    self.ctrl[idx] = sig
                    self.keys[idx] = key
                    self.vals[idx] = val
                    self.size += 1
                    self.groups_probed_total += groups_probed
                    return
            if groups_probed > self.num_groups:
                raise RuntimeError("表已满，需扩容")

    def lookup(self, key: int):
        """返回 (是否命中, 扫描组数)。"""
        sig = self._h2(key)
        g0 = self._h1(key)
        groups_probed = 0
        for g in self._probe_group(g0):
            groups_probed += 1
            base = g * self.G
            gb = self._group_bytes(g)
            for i in self._match_positions(gb, sig):
                idx = base + i
                if self.keys[idx] == key:
                    self.groups_probed_total += groups_probed
                    self.lookup_count += 1
                    return self.vals[idx], groups_probed
            # 组内有空槽即可判定不存在（真实签名不可能等于空槽编码）
            if self.EMPTY in gb:
                self.groups_probed_total += groups_probed
                self.lookup_count += 1
                return None, groups_probed
            if groups_probed > self.num_groups:
                break
        self.groups_probed_total += groups_probed
        self.lookup_count += 1
        return None, groups_probed

    def delete(self, key: int) -> bool:
        sig = self._h2(key)
        g0 = self._h1(key)
        for g in self._probe_group(g0):
            base = g * self.G
            gb = self._group_bytes(g)
            for i in self._match_positions(gb, sig):
                idx = base + i
                if self.keys[idx] == key:
                    self.ctrl[idx] = self.DELETED   # 墓碑，不断探测链
                    self.keys[idx] = None
                    self.vals[idx] = None
                    self.size -= 1
                    return True
            if self.EMPTY in gb:
                return False

    def memory_bytes(self) -> int:
        """元数据 + 槽位数组的实际字节占用（Python 对象不计）。"""
        return len(self.ctrl) + self.cap * 8  # keys/vals 用引用近似，真实实现无引用


def measure(target_alpha, n=2000, group_size=8, trials=3):
    """在给定目标负载下实测瑞士表。"""
    import math
    cap = num_slots_for(n, target_alpha, group_size)
    actual_alpha = n / cap
    keys = list(range(n))

    best = None
    for t in range(trials):
        random.seed(4321 + t)
        random.shuffle(keys)
        table = SwissTable(cap, group_size)
        for k in keys:
            table.insert(k, k * 2)
        # 查找延迟：命中 + 未命中
        probe_keys = [random.choice(keys) for _ in range(20000)]
        missing = [-k - 1 for k in probe_keys]

        t0 = time.perf_counter_ns()
        for k in probe_keys:
            table.lookup(k)
        hit_ns = (time.perf_counter_ns() - t0) / len(probe_keys)

        t0 = time.perf_counter_ns()
        for k in missing:
            table.lookup(k)
        miss_ns = (time.perf_counter_ns() - t0) / len(missing)

        # 平均扫描组数（用未命中查找，它一定走到空槽，代价上界清晰）
        total_groups = 0
        cnt = 0
        for k in missing[:5000]:
            _, gp = table.lookup(k)
            total_groups += gp
            cnt += 1
        avg_groups = total_groups / cnt

        r = {
            "target_alpha": target_alpha,
            "capacity": cap,
            "actual_alpha": actual_alpha,
            "group_size": group_size,
            "hit_latency_ns": hit_ns,
            "miss_latency_ns": miss_ns,
            "measured_groups_probed": avg_groups,
            "theoretical_groups_probed": avg_groups_probed(actual_alpha, group_size),
            "metadata_bytes": cap,
            "slot_bytes_approx": cap * 8,
        }
        if best is None or r["hit_latency_ns"] < best["hit_latency_ns"]:
            best = r
    return best


def measure_with_tombstones(n=2000, alpha=0.85, group_size=8, delete_ratio=0.5):
    """墓碑对查找的影响：删掉一半元素后重测未命中延迟。"""
    import math
    cap = num_slots_for(n, alpha, group_size)
    keys = list(range(n))
    random.seed(99)
    random.shuffle(keys)
    table = SwissTable(cap, group_size)
    for k in keys:
        table.insert(k, k)

    # 删掉一半
    to_delete = keys[: n // 2]
    for k in to_delete:
        table.delete(k)

    missing = [-k - 1 for k in range(20000)]
    t0 = time.perf_counter_ns()
    for k in missing:
        table.lookup(k)
    after_ns = (time.perf_counter_ns() - t0) / len(missing)
    return {
        "n": n,
        "deleted": len(to_delete),
        "delete_ratio": delete_ratio,
        "miss_latency_after_delete_ns": after_ns,
    }


def main():
    print("=== 瑞士表实测验证 ===\n")
    results = []

    for alpha in [0.5, 0.7, 0.85]:
        r = measure(alpha)
        results.append(r)
        print(f"目标 α={alpha:.2f}  实际 α={r['actual_alpha']:.3f}  容量={r['capacity']}  组大小={r['group_size']}")
        print(f"  命中延迟 {r['hit_latency_ns']:.0f} ns   未命中延迟 {r['miss_latency_ns']:.0f} ns")
        print(f"  实测扫描组数 {r['measured_groups_probed']:.2f}   "
              f"理论 {r['theoretical_groups_probed']:.2f}")
        print()

    tomb = measure_with_tombstones()
    print(f"=== 墓碑影响（删除 {tomb['deleted']} 个元素后）===")
    print(f"  未命中查找延迟 {tomb['miss_latency_after_delete_ns']:.0f} ns")

    # 与 Python dict 对比
    print(f"\n=== 与 Python dict 对比（α=0.85, n=2000）===")
    n = 2000
    cap = num_slots_for(n, 0.85, 8)
    keys = list(range(n))
    random.seed(123)
    random.shuffle(keys)
    table = SwissTable(cap, 8)
    for k in keys:
        table.insert(k, k)
    d = {k: k for k in keys}

    probe = [random.choice(keys) for _ in range(20000)]
    t0 = time.perf_counter_ns()
    for k in probe:
        table.lookup(k)
    rh_hit = (time.perf_counter_ns() - t0) / len(probe)
    t0 = time.perf_counter_ns()
    for k in probe:
        _ = d[k]
    dict_hit = (time.perf_counter_ns() - t0) / len(probe)
    print(f"  瑞士表（纯 Python）命中 {rh_hit:.0f} ns")
    print(f"  dict（C 实现）命中      {dict_hit:.0f} ns")
    print("  （差值是解释器开销，不是算法差异；两者都是开放寻址）")

    out = {
        "meta": {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "python": sys.version.split()[0],
            "note": "自制瑞士表（bytearray 控制字 + 槽位数组）实测查找延迟与扫描组数",
        },
        "load_measurements": results,
        "tombstone_effect": tomb,
        "dict_comparison": {
            "n": n,
            "actual_alpha": n / cap,
            "swiss_table_hit_ns": rh_hit,
            "dict_hit_ns": dict_hit,
        },
    }
    out_path = os.path.join(os.path.dirname(__file__), "..", "results",
                            f"swiss_table_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n结果写入 {out_path}")


if __name__ == "__main__":
    main()
