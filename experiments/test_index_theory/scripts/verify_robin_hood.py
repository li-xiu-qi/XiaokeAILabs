# -*- coding: utf-8 -*-
"""
罗宾汉哈希实测验证

自制罗宾汉哈希表（开放寻址 + 线性探测 + PSL 交换），实测：
1. 不同负载因子（0.5/0.7/0.9/0.95）下的平均探测次数，对比理论理想值
2. PSL 分布（均值/标准差/p50/p95/p99/最大值），对比普通线性探测
3. 查找延迟，对比 Python dict（近似现代开放寻址实现）
4. 存储开销（含 PSL 字段）

对照组是同一哈希函数下的普通线性探测表，差异只来自「是否做劫富济贫」。
结果写入 results/robin_hood_<timestamp>.json
"""
import os
import sys
import json
import time
import random
import statistics
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from robin_hood_model import (
    RobinHoodSpec, compute, avg_probe_success, avg_probe_unsuccess,
    linear_probe_success,
)


def _hash(key: int) -> int:
    """乘法哈希 + 折叠，让连续整数分散，避免退化成恒等映射。"""
    h = (key * 2654435761) & 0xFFFFFFFF
    h ^= h >> 15
    h = (h * 2246822519) & 0xFFFFFFFF
    h ^= h >> 13
    return h


class RobinHoodTable:
    """
    罗宾汉哈希表。槽位存 (key, psl)，psl 是该元素相对自己理想位置的偏移。

    插入：线性探测，遇占位且对方 psl 更小则抢占，被抢者携带自己 psl+1 继续。
    查找：遇空槽即失败；遇对方 psl 比自己还小即提前失败（后面不可能有）。
    """

    def __init__(self, capacity: int):
        self.cap = capacity
        self.keys = [None] * capacity      # key
        self.psl = [0] * capacity          # 相对理想位置的探测距离
        self.size = 0
        self.swaps = 0                     # 累计抢占次数，度量插入代价

    def _slot(self, key: int):
        """理想槽位 = home(key)"""
        return _hash(key) % self.cap

    def insert(self, key: int) -> int:
        """插入，返回本次插入的探测步数。"""
        home = self._slot(key)
        dist = 0
        probes = 0
        cur = key
        while True:
            idx = (home + dist) % self.cap
            probes += 1
            if self.keys[idx] is None:
                self.keys[idx] = cur
                self.psl[idx] = dist
                self.size += 1
                return probes
            if self.psl[idx] < dist:
                # 对方更富（更靠近自己的理想位置），抢占它
                self.swaps += 1
                old_key, old_psl = self.keys[idx], self.psl[idx]
                self.keys[idx] = cur
                self.psl[idx] = dist
                cur = old_key
                dist = old_psl + 1        # 被抢者相对自己理想位置的偏移
                home = self._slot(cur)    # 被抢者的理想位置是它自己的 home
            else:
                dist += 1
            if probes > self.cap * 4:
                raise RuntimeError(f"探测超限 {probes}，容量可能不足")

    def lookup(self, key: int) -> int:
        """返回探测步数；查不到时返回的步数即「不成功查找」代价。"""
        home = self._slot(key)
        dist = 0
        probes = 0
        while True:
            idx = (home + dist) % self.cap
            probes += 1
            if self.keys[idx] is None:
                return probes          # 提前终止：遇到空槽
            if self.keys[idx] == key:
                return probes
            if self.psl[idx] < dist:
                return probes          # 提前终止：后面没有比自己更穷的
            dist += 1
            if probes > self.cap * 4:
                return probes

    def psl_distribution(self):
        """返回所有已占用槽位的 PSL 列表。"""
        return [self.psl[i] for i in range(self.cap) if self.keys[i] is not None]


class LinearProbeTable:
    """普通线性探测表（对照）。无 PSL 记录、无交换，差异只在是否劫富济贫。"""

    def __init__(self, capacity: int):
        self.cap = capacity
        self.keys = [None] * capacity
        self.size = 0

    def _slot(self, key: int):
        return _hash(key) % self.cap

    def insert(self, key: int) -> int:
        home = self._slot(key)
        probes = 0
        dist = 0
        while True:
            idx = (home + dist) % self.cap
            probes += 1
            if self.keys[idx] is None:
                self.keys[idx] = key
                self.size += 1
                return probes
            dist += 1
            if probes > self.cap * 4:
                raise RuntimeError(f"探测超限 {probes}，容量可能不足")

    def lookup(self, key: int) -> int:
        home = self._slot(key)
        probes = 0
        dist = 0
        while True:
            idx = (home + dist) % self.cap
            probes += 1
            if self.keys[idx] is None:
                return probes
            if self.keys[idx] == key:
                return probes
            dist += 1
            if probes > self.cap * 4:
                return probes

    def displacement_distribution(self):
        """普通线性探测下每个元素相对理想位置的偏移（等于成功查找代价）。"""
        dists = []
        for i, k in enumerate(self.keys):
            if k is None:
                continue
            home = self._slot(k)
            # 从 home 线性扫到 i 的步数
            d = (i - home) % self.cap
            dists.append(d)
        return dists


def _percentile(sorted_vals, q):
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def measure(target_alpha, n=2000, trials=3):
    """在给定目标负载下实测罗宾汉与普通线性探测。"""
    cap = -(-n // int(n / target_alpha)) if target_alpha > 0 else n
    # 精确一点：cap = ceil(n / target_alpha)
    import math
    cap = math.ceil(n / target_alpha)
    actual_alpha = n / cap

    keys = list(range(n))
    out = {
        "target_alpha": target_alpha,
        "capacity": cap,
        "actual_alpha": actual_alpha,
    }

    # ---- 罗宾汉 ----
    rh = RobinHoodTable(cap)
    rh_insert = []
    for k in keys:
        rh_insert.append(rh.insert(k))
    rh_succ = [rh.lookup(k) for k in keys]              # 成功查找探测
    rh_unsucc = [rh.lookup(-k - 1) for k in range(n)]   # 不存在的键
    rh_psl = rh.psl_distribution()
    rh_swaps = rh.swaps

    rh_psl_sorted = sorted(rh_psl)
    out["robin_hood"] = {
        "insert_probes": sum(rh_insert) / len(rh_insert),
        "measured_success": sum(rh_succ) / len(rh_succ),
        "theoretical_success": avg_probe_success(actual_alpha),
        "measured_unsuccessful": sum(rh_unsucc) / len(rh_unsucc),
        "theoretical_unsuccessful": avg_probe_unsuccess(actual_alpha),
        "psl_mean": statistics.mean(rh_psl),
        "psl_std": statistics.pstdev(rh_psl),
        "psl_p50": _percentile(rh_psl_sorted, 0.50),
        "psl_p95": _percentile(rh_psl_sorted, 0.95),
        "psl_p99": _percentile(rh_psl_sorted, 0.99),
        "psl_max": max(rh_psl),
        "swaps_per_insert": rh_swaps / n,
    }

    # ---- 普通线性探测 ----
    lp = LinearProbeTable(cap)
    lp_insert = []
    for k in keys:
        lp_insert.append(lp.insert(k))
    lp_succ = [lp.lookup(k) for k in keys]
    lp_unsucc = [lp.lookup(-k - 1) for k in range(n)]
    lp_disp = lp.displacement_distribution()

    lp_sorted = sorted(lp_disp)
    out["linear_probe"] = {
        "insert_probes": sum(lp_insert) / len(lp_insert),
        "measured_success": sum(lp_succ) / len(lp_succ),
        "theoretical_success": linear_probe_success(actual_alpha),
        "measured_unsuccessful": sum(lp_unsucc) / len(lp_unsucc),
        "disp_mean": statistics.mean(lp_disp),
        "disp_std": statistics.pstdev(lp_disp),
        "disp_p95": _percentile(lp_sorted, 0.95),
        "disp_p99": _percentile(lp_sorted, 0.99),
        "disp_max": max(lp_disp),
    }
    return out


def measure_lookup_latency(n=2000, alpha=0.9):
    """对比罗宾汉表与 Python dict 的查找延迟。"""
    import math
    cap = math.ceil(n / alpha)
    keys = list(range(n))
    random.seed(7)
    random.shuffle(keys)

    rh = RobinHoodTable(cap)
    for k in keys:
        rh.insert(k)
    probe_keys = [random.choice(keys) for _ in range(20000)]
    missing_keys = [-k - 1 for k in probe_keys]

    # 罗宾汉表命中查找
    t0 = time.perf_counter_ns()
    for k in probe_keys:
        rh.lookup(k)
    rh_hit_ns = (time.perf_counter_ns() - t0) / len(probe_keys)

    # 罗宾汉表未命中查找
    t0 = time.perf_counter_ns()
    for k in missing_keys:
        rh.lookup(k)
    rh_miss_ns = (time.perf_counter_ns() - t0) / len(missing_keys)

    # Python dict（C 实现开放寻址，作现代实现的下界参考）
    d = {k: k * 2 for k in keys}
    t0 = time.perf_counter_ns()
    for k in probe_keys:
        _ = d[k]
    dict_hit_ns = (time.perf_counter_ns() - t0) / len(probe_keys)

    t0 = time.perf_counter_ns()
    for k in missing_keys:
        _ = d.get(k)
    dict_miss_ns = (time.perf_counter_ns() - t0) / len(missing_keys)

    return {
        "n": n,
        "capacity": cap,
        "actual_alpha": n / cap,
        "probe_keys": len(probe_keys),
        "robin_hood_hit_ns": rh_hit_ns,
        "robin_hood_miss_ns": rh_miss_ns,
        "dict_hit_ns": dict_hit_ns,
        "dict_miss_ns": dict_miss_ns,
    }


def main():
    print("=== 罗宾汉哈希实测验证 ===\n")
    results = []

    for alpha in [0.5, 0.7, 0.9, 0.95]:
        r = measure(alpha)
        results.append(r)
        rh = r["robin_hood"]
        lp = r["linear_probe"]
        print(f"目标 α={alpha:.2f}  实际 α={r['actual_alpha']:.3f}  容量={r['capacity']}")
        print(f"  罗宾汉 成功: 实测 {rh['measured_success']:.2f}  理论 {rh['theoretical_success']:.2f}")
        print(f"  罗宾汉 失败: 实测 {rh['measured_unsuccessful']:.2f}  理论 {rh['theoretical_unsuccessful']:.2f}")
        print(f"  罗宾汉 插入: 实测 {rh['insert_probes']:.2f}")
        print(f"  罗宾汉 PSL: 均值 {rh['psl_mean']:.2f}  std {rh['psl_std']:.2f}  "
              f"p95 {rh['psl_p95']:.0f}  p99 {rh['psl_p99']:.0f}  max {rh['psl_max']}")
        print(f"  线性探测成功: 实测 {lp['measured_success']:.2f}  理论 {lp['theoretical_success']:.2f}"
              f"  失败: 实测 {lp['measured_unsuccessful']:.2f}")
        print(f"  线性偏移: 均值 {lp['disp_mean']:.2f}  std {lp['disp_std']:.2f}  "
              f"p95 {lp['disp_p95']:.0f}  p99 {lp['disp_p99']:.0f}  max {lp['disp_max']}")
        print(f"  抢占次数/插入: {rh['swaps_per_insert']:.2f}")
        print()

    lat = measure_lookup_latency()
    print(f"=== 查找延迟（α={lat['actual_alpha']:.3f}, {lat['probe_keys']} 次查找）===")
    print(f"  罗宾汉 命中 {lat['robin_hood_hit_ns']:.0f} ns   未命中 {lat['robin_hood_miss_ns']:.0f} ns")
    print(f"  dict   命中 {lat['dict_hit_ns']:.0f} ns   未命中 {lat['dict_miss_ns']:.0f} ns")
    print("  （罗宾汉为纯 Python 实现，dict 为 C 实现，差值主要是解释器开销，非算法差异）")

    out = {
        "meta": {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "python": sys.version.split()[0],
            "note": "自制罗宾汉哈希表 vs 普通线性探测，含 PSL 分布与查找延迟",
        },
        "probe_measurements": results,
        "lookup_latency": lat,
    }
    out_path = os.path.join(os.path.dirname(__file__), "..", "results",
                            f"robin_hood_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n结果写入 {out_path}")


if __name__ == "__main__":
    main()
