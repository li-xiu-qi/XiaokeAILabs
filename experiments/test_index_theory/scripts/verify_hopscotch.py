# -*- coding: utf-8 -*-
"""
跳房子哈希（Hopscotch Hashing）实测验证

自制跳房子哈希表：桶数组 + 每桶一个 H 位邻域位图。
- 插入：先找 home 桶的邻域内空位；邻域内无空位则线性探测更远处，
  找到空位后用「交换位移」把它一步步挪进邻域（每次与邻域边界上的元素交换）
- 查找：home 桶 + 位图标记的邻域位置，上界 H 次比较
- 删除：清位图对应位

实测：
1. 不同 H（4/8/16/32）与负载（0.5/0.7/0.9）下的查找延迟
2. 查找上界是否真的被 H 约束（实测最大查找位置数）
3. 插入时的位移次数（交换开销）
4. 与普通线性探测的查找延迟对比

结果写入 results/hopscotch_<timestamp>.json
"""
import os
import sys
import json
import time
import random
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from hopscotch_model import (
    HopscotchSpec, compute, num_buckets_for, avg_lookup_positions,
)


class HopscotchTable:
    """
    跳房子哈希表。

    keys：桶数组
    bitmap：每桶一个 H 位整数，第 i 位为 1 表示 home 桶 + i 位置存了本桶的元素
    """

    def __init__(self, num_buckets: int, neighborhood: int):
        self.cap = num_buckets
        self.H = neighborhood
        self.keys = [None] * num_buckets
        self.bitmap = [0] * num_buckets
        self.size = 0
        self.displacements = 0          # 累计交换次数
        self.max_displace_chain = 0     # 单次插入的最大交换链

    def _home(self, key: int) -> int:
        h = (key * 2654435761) & 0xFFFFFFFF
        h ^= h >> 15
        h = (h * 2246822519) & 0xFFFFFFFF
        h ^= h >> 13
        return h % self.cap

    def _nb_free_slot(self, home: int) -> int:
        """在 home 的邻域 [home, home+H) 内找空位，返回绝对下标或 -1。"""
        for i in range(self.H):
            idx = (home + i) % self.cap
            if self.keys[idx] is None:
                return idx
        return -1

    def insert(self, key: int) -> bool:
        home = self._home(key)

        # 已存在则跳过
        for i in range(self.H):
            if self.bitmap[home] >> i & 1:
                if self.keys[(home + i) % self.cap] == key:
                    return True

        # 在邻域内找空位
        free = self._nb_free_slot(home)
        if free != -1:
            self._place(key, home, free)
            return True

        # 邻域满了：线性探测更远，找空位后一步步挪进邻域
        chain = 0
        probe = self.H
        while probe < self.cap:
            idx = (home + probe) % self.cap
            if self.keys[idx] is None:
                # 把 idx 处的空位一步步交换到邻域内
                if self._relocate(home, idx):
                    free = self._nb_free_slot(home)
                    if free != -1:
                        self._place(key, home, free)
                        return True
                chain += 1
                if chain > 200:
                    return False   # 位移链过长，触发扩容
            probe += 1
        return False

    def _place(self, key: int, home: int, idx: int):
        """把 key 放到 idx，并在 home 的位图里标记。"""
        self.keys[idx] = key
        offset = (idx - home) % self.cap
        self.bitmap[home] |= (1 << offset)
        self.size += 1

    def _relocate(self, home: int, empty_idx: int) -> bool:
        """
        把 empty_idx 处的空位一步步挪进 home 的邻域。
        每次找 empty_idx 前 H-1 个位置内、其 home 能把空位移过来的元素，交换之。
        """
        cur = empty_idx
        for _ in range(self.H * 4):
            # 在 cur 前 H-1 个桶中，找一个元素其 home 覆盖 cur
            for back in range(1, self.H):
                idx = (cur - back) % self.cap
                k = self.keys[idx]
                if k is None:
                    continue
                khome = self._home(k)
                # khome 的邻域是否覆盖 cur？
                offset = (cur - khome) % self.cap
                if offset < self.H:
                    # 交换 k 到 cur，idx 变空
                    self.keys[cur] = k
                    self.keys[idx] = None
                    self.displacements += 1
                    # 更新位图：k 的标记从 idx 移到 cur
                    self.bitmap[khome] &= ~(1 << ((idx - khome) % self.cap))
                    self.bitmap[khome] |= (1 << offset)
                    cur = idx
                    if (cur - home) % self.cap < self.H:
                        return True
                    break
            else:
                return False
        return False

    def lookup(self, key: int) -> int:
        """返回查找位置数（探测步数）。"""
        home = self._home(key)
        bm = self.bitmap[home]
        steps = 0
        i = 0
        while bm:
            if bm & 1:
                idx = (home + i) % self.cap
                steps += 1
                if self.keys[idx] == key:
                    return steps
            bm >>= 1
            i += 1
            if i >= self.H:
                break
        # home 桶本身（位图第 0 位）
        if self.bitmap[home] & 1:
            idx = home
            if self.keys[idx] == key:
                return 1
        return steps

    def lookup_miss(self, key: int) -> int:
        """不命中查找的位置数（上界由 H 决定）。"""
        home = self._home(key)
        bm = self.bitmap[home]
        steps = 0
        i = 0
        while bm:
            if bm & 1:
                steps += 1
            bm >>= 1
            i += 1
            if i >= self.H:
                break
        return steps


class LinearProbeTable:
    """普通线性探测表（对照）。"""

    def __init__(self, num_buckets: int):
        self.cap = num_buckets
        self.keys = [None] * num_buckets
        self.size = 0

    def _home(self, key: int) -> int:
        h = (key * 2654435761) & 0xFFFFFFFF
        h ^= h >> 15
        h = (h * 2246822519) & 0xFFFFFFFF
        h ^= h >> 13
        return h % self.cap

    def insert(self, key: int) -> int:
        home = self._home(key)
        for i in range(self.cap):
            idx = (home + i) % self.cap
            if self.keys[idx] is None:
                self.keys[idx] = key
                self.size += 1
                return i + 1
        return -1

    def lookup(self, key: int) -> int:
        home = self._home(key)
        for i in range(self.cap):
            idx = (home + i) % self.cap
            if self.keys[idx] is None:
                return i + 1
            if self.keys[idx] == key:
                return i + 1
        return self.cap


def measure(neighborhood, target_alpha, n=2000, trials=3):
    """
    在给定 H 与负载下实测跳房子表。

    容量 = ceil(n / 目标负载) × 1.05。跳房子在邻域内找不到空位时会触发线性探测
    位移，容量取整到刚好贴着上限时位移链容易触顶，工程实现都留余量
    （参考 verify_hash.py 同样留 2 个空位）。实测负载因子按真实 n/容量报告。
    """
    import math
    cap = math.ceil(num_buckets_for(n, target_alpha) * 1.05)
    actual_alpha = n / cap
    keys = list(range(n))

    best = None
    for t in range(trials):
        random.seed(777 + t)
        random.shuffle(keys)
        table = HopscotchTable(cap, neighborhood)
        failed = 0
        for k in keys:
            if not table.insert(k):
                failed += 1
        if failed > 0:
            return {
                "neighborhood": neighborhood,
                "target_alpha": target_alpha,
                "capacity": cap,
                "actual_alpha": actual_alpha,
                "insert_failures": failed,
            }

        # 查找延迟
        probe_keys = [random.choice(keys) for _ in range(20000)]
        missing = [-k - 1 for k in probe_keys]

        t0 = time.perf_counter_ns()
        for k in probe_keys:
            table.lookup(k)
        hit_ns = (time.perf_counter_ns() - t0) / len(probe_keys)

        t0 = time.perf_counter_ns()
        for k in missing:
            table.lookup_miss(k)
        miss_ns = (time.perf_counter_ns() - t0) / len(missing)

        # 查找位置上界
        max_steps = 0
        total_steps = 0
        for k in keys:
            s = table.lookup(k)
            max_steps = max(max_steps, s)
            total_steps += s
        avg_steps = total_steps / len(keys)

        max_miss = max(table.lookup_miss(k) for k in missing[:5000])

        r = {
            "neighborhood": neighborhood,
            "target_alpha": target_alpha,
            "capacity": cap,
            "actual_alpha": actual_alpha,
            "insert_failures": 0,
            "hit_latency_ns": hit_ns,
            "miss_latency_ns": miss_ns,
            "measured_max_lookup_steps": max_steps,
            "measured_max_miss_steps": max_miss,
            "measured_avg_lookup_steps": avg_steps,
            "theoretical_avg_lookup_steps": avg_lookup_positions(actual_alpha, neighborhood),
            "displacements_per_insert": table.displacements / n,
        }
        if best is None or r["hit_latency_ns"] < best["hit_latency_ns"]:
            best = r
    return best


def measure_linear_control(target_alpha, n=2000, trials=3):
    """普通线性探测对照。"""
    cap = num_buckets_for(n, target_alpha)
    actual_alpha = n / cap
    keys = list(range(n))
    best = None
    for t in range(trials):
        random.seed(777 + t)
        random.shuffle(keys)
        table = LinearProbeTable(cap)
        for k in keys:
            table.insert(k)
        probe_keys = [random.choice(keys) for _ in range(20000)]
        t0 = time.perf_counter_ns()
        for k in probe_keys:
            table.lookup(k)
        hit_ns = (time.perf_counter_ns() - t0) / len(probe_keys)
        max_steps = max(table.lookup(k) for k in keys)
        r = {
            "target_alpha": target_alpha,
            "actual_alpha": actual_alpha,
            "hit_latency_ns": hit_ns,
            "max_lookup_steps": max_steps,
        }
        if best is None or r["hit_latency_ns"] < best["hit_latency_ns"]:
            best = r
    return best


def main():
    print("=== 跳房子哈希实测验证 ===\n")
    results = []

    for alpha in [0.5, 0.7, 0.9]:
        for h in [4, 8, 16, 32]:
            r = measure(h, alpha)
            results.append(r)
            if r.get("insert_failures", 0) > 0:
                print(f"H={h:>2}  α={alpha:.2f}  插入失败 {r['insert_failures']} 个（邻域过小或位移链过长）")
                continue
            print(f"H={h:>2}  α={r['actual_alpha']:.3f}  桶数={r['capacity']}")
            print(f"  命中 {r['hit_latency_ns']:.0f} ns   未命中 {r['miss_latency_ns']:.0f} ns")
            print(f"  最大查找步数 {r['measured_max_lookup_steps']}（H={h}）  "
                  f"最大未命中步数 {r['measured_max_miss_steps']}")
            print(f"  平均查找步数 {r['measured_avg_lookup_steps']:.2f}  "
                  f"理论 {r['theoretical_avg_lookup_steps']:.2f}  "
                  f"位移/插入 {r['displacements_per_insert']:.2f}")
            print()

    # 对照：普通线性探测
    print("=== 普通线性探测对照 ===")
    control = []
    for alpha in [0.5, 0.7, 0.9]:
        c = measure_linear_control(alpha)
        control.append(c)
        print(f"α={c['actual_alpha']:.3f}  命中 {c['hit_latency_ns']:.0f} ns  "
              f"最大查找步数 {c['max_lookup_steps']}")

    out = {
        "meta": {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "python": sys.version.split()[0],
            "note": "自制跳房子哈希（桶数组 + H 位邻域位图）实测查找延迟与上界",
        },
        "hopscotch_measurements": results,
        "linear_probe_control": control,
    }
    out_path = os.path.join(os.path.dirname(__file__), "..", "results",
                            f"hopscotch_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n结果写入 {out_path}")


if __name__ == "__main__":
    main()
