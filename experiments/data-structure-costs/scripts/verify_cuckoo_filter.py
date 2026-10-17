# -*- coding: utf-8 -*-
"""
布谷鸟过滤器实测验证

用纯 Python 自制迷你布谷鸟过滤器，实测三件事：
1. 假阳性率随负载因子变化（固定指纹位数），验证 ε = 1 - (1 - 1/2^f)^(2b)
2. 假阳性率随指纹位数变化（固定负载因子）
3. 实际可达负载因子 vs 论文理论值（b=2→84%, b=4→95%, b=8→98%）

实现要点（partial-key cuckoo hashing, Fan et al. 2014）：
- 指纹 fp 取哈希值的低 f 位
- 桶索引 i1 = h1(x) mod m
- 第二桶 i2 = i1 XOR (hash(fp) mod m)，由指纹反推，省去第二个哈希函数
- 插入失败时踢出随机槽位中的指纹，递归重插，超过 max_kicks 判定为满
- 删除只精确移除目标指纹，不影响其他元素

结果写入 results/cuckoo_filter_<timestamp>.json
"""
import os
import sys
import json
import hashlib
import random
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from cuckoo_filter_model import (
    CuckooFilterSpec, compute,
    compute_fpr, compute_fpr_upper_bound,
    compute_num_buckets, compute_min_fingerprint_bits,
    THEORETICAL_LOAD_FACTOR,
)


def next_pow2(x: int) -> int:
    """向上取整到 2 的幂。cuckoo hashing 要求桶数为 2 的幂，否则 XOR 会越界。"""
    p = 1
    while p < x:
        p <<= 1
    return p


class CuckooFilter:
    """自制迷你布谷鸟过滤器。"""

    def __init__(self, num_buckets: int, bucket_size: int,
                 fingerprint_bits: int, max_kicks: int = 500, seed: int = 42):
        self.m = num_buckets
        self.b = bucket_size
        self.f = fingerprint_bits
        self.fp_mask = (1 << fingerprint_bits) - 1
        self.max_kicks = max_kicks
        # 桶数组：每桶 b 个槽位，None 表示空
        self.buckets = [[None] * bucket_size for _ in range(num_buckets)]
        self.num_items = 0
        self.insert_failures = 0
        self.rng = random.Random(seed)

    # ---- 哈希 ----
    def _fingerprint(self, item: str) -> int:
        """指纹：取哈希值低 f 位。fp=0 会退化成"只查一个桶"，按论文处理为 1。"""
        h = hashlib.md5(item.encode("utf-8")).digest()
        fp = int.from_bytes(h, "little") & self.fp_mask
        return fp if fp != 0 else 1

    def _index(self, item: str) -> int:
        """第一桶索引：h1(x) mod m。"""
        h = hashlib.sha1(item.encode("utf-8")).digest()
        return int.from_bytes(h[:8], "little") % self.m

    def _alt_index(self, index: int, fp: int) -> int:
        """第二桶索引：i1 XOR hash(fp) mod m（partial-key cuckoo hashing）。

        前提是 m 为 2 的幂：XOR 只翻转低位，结果必然仍在 [0, m) 内。
        由指纹反推第二位置，省去第二个独立哈希函数。
        """
        h = hashlib.md5(fp.to_bytes(8, "little")).digest()
        return index ^ (int.from_bytes(h[:8], "little") % self.m)

    # ---- 操作 ----
    def _bucket_has(self, index: int, fp: int) -> bool:
        return fp in self.buckets[index]

    def insert(self, item: str) -> bool:
        """插入元素。返回 True 成功，False 表示过滤器已满。"""
        fp = self._fingerprint(item)
        i1 = self._index(item)
        i2 = self._alt_index(i1, fp)

        # 已有相同指纹则视为已存在（不重复计数）
        if self._bucket_has(i1, fp) or self._bucket_has(i2, fp):
            return True

        idx = i1
        for _ in range(self.max_kicks):
            bucket = self.buckets[idx]
            empty = None
            for s in range(self.b):
                if bucket[s] is None:
                    empty = s
                    break
            if empty is not None:
                bucket[empty] = fp
                self.num_items += 1
                return True
            # 桶满：随机踢出一个，把被踢者换到自己桶里，自己继续走被踢者的另一条路
            victim = self.rng.randrange(self.b)
            fp, bucket[victim] = bucket[victim], fp
            idx = self._alt_index(idx, fp)

        self.insert_failures += 1
        return False

    def contains(self, item: str) -> bool:
        """查询元素。返回 True 表示"可能存在"。"""
        fp = self._fingerprint(item)
        i1 = self._index(item)
        i2 = self._alt_index(i1, fp)
        return self._bucket_has(i1, fp) or self._bucket_has(i2, fp)

    def delete(self, item: str) -> bool:
        """删除元素。返回 True 表示确实删掉了一个指纹。"""
        fp = self._fingerprint(item)
        i1 = self._index(item)
        i2 = self._alt_index(i1, fp)
        for idx in (i1, i2):
            bucket = self.buckets[idx]
            for s in range(self.b):
                if bucket[s] == fp:
                    bucket[s] = None
                    self.num_items -= 1
                    return True
        return False

    def occupied_slots(self) -> int:
        """已占用槽位数。"""
        return sum(1 for bkt in self.buckets for s in bkt if s is not None)

    def achieved_load(self) -> float:
        """实际负载因子 = 已占用槽数 / 总槽数。"""
        return self.occupied_slots() / (self.m * self.b)


def _query_budget(expected_fpr: float) -> int:
    """按期望假阳性率自适应查询预算，保证低 FPR 也有足够假阳性样本。"""
    return max(50_000, min(600_000, int(250 / max(expected_fpr, 1e-9))))


def measure_fpr_vs_load(num_buckets: int, fingerprint_bits: int,
                        load_list: list) -> list:
    """实测假阳性率随负载因子的变化。

    固定桶数（必须为 2 的幂），按目标负载反推插入数量：
    n = floor(α × m × b)。这样负载因子被精确控制，不会被取整到 2 的幂带偏。
    注意不能反过来先定 n 再定桶数——桶数取整到 2 的幂会把 0.75/0.9/0.95 全压到
    同一实际负载上，测不出负载因子的影响。
    """
    bucket_size = 4
    num_slots = num_buckets * bucket_size
    results = []
    for alpha in load_list:
        n = int(alpha * num_slots)
        cf = CuckooFilter(num_buckets, bucket_size, fingerprint_bits)

        t0 = time.time()
        for i in range(n):
            cf.insert(f"item_{i}")
        insert_secs = time.time() - t0

        theo = compute_fpr(fingerprint_bits, bucket_size, alpha)
        nq = _query_budget(theo)
        fp_hits = 0
        for i in range(nq):
            if cf.contains(f"nonexistent_{i}"):
                fp_hits += 1

        results.append({
            "target_load": alpha,
            "achieved_load": cf.achieved_load(),
            "n": n,
            "num_buckets": num_buckets,
            "num_slots": num_slots,
            "fingerprint_bits": fingerprint_bits,
            "num_queries": nq,
            "false_positives": fp_hits,
            "actual_fpr": fp_hits / nq,
            "theoretical_fpr": theo,
            "fpr_upper_bound": compute_fpr_upper_bound(fingerprint_bits, bucket_size),
            "insert_failures": cf.insert_failures,
            "insert_secs": round(insert_secs, 2),
        })
    return results


def measure_fpr_vs_bits(num_buckets: int, bits_list: list,
                        load_factor: float) -> list:
    """实测假阳性率随指纹位数的变化（固定桶数与负载因子）。"""
    bucket_size = 4
    num_slots = num_buckets * bucket_size
    n = int(load_factor * num_slots)
    results = []
    for f in bits_list:
        cf = CuckooFilter(num_buckets, bucket_size, f)
        for i in range(n):
            cf.insert(f"item_{i}")

        theo = compute_fpr(f, bucket_size, load_factor)
        nq = _query_budget(theo)
        fp_hits = 0
        for i in range(nq):
            if cf.contains(f"nonexistent_{i}"):
                fp_hits += 1

        results.append({
            "fingerprint_bits": f,
            "load_factor": load_factor,
            "n": n,
            "num_buckets": num_buckets,
            "num_queries": nq,
            "false_positives": fp_hits,
            "actual_fpr": fp_hits / nq,
            "theoretical_fpr": theo,
            "fpr_upper_bound": compute_fpr_upper_bound(f, bucket_size),
        })
    return results


def measure_achievable_load(n: int, bucket_size: int,
                            hard_cap: int = 300_000) -> dict:
    """实测实际可达负载因子：持续插入直到插入失败。"""
    # 给足桶数（按 99% 占用预配，再取整到 2 的幂），
    # 让插入失败由踢出循环触发而非容量不足
    num_buckets = next_pow2(compute_num_buckets(n, bucket_size, 0.99))
    cf = CuckooFilter(num_buckets, bucket_size, 12, max_kicks=500)
    inserted = 0
    for i in range(hard_cap):
        if not cf.insert(f"item_{i}"):
            break
        inserted += 1

    achieved = cf.achieved_load()
    theo = THEORETICAL_LOAD_FACTOR[bucket_size]
    return {
        "bucket_size": bucket_size,
        "num_buckets": num_buckets,
        "num_slots": num_buckets * bucket_size,
        "inserted": inserted,
        "achieved_load": achieved,
        "theoretical_load": theo,
        "insert_failures": cf.insert_failures,
    }


def measure_delete(n: int = 2000) -> dict:
    """验证删除语义：删除后假阳性率应回落。

    同时暴露布谷鸟过滤器删除的真实限制：只存指纹不存原值，两个不同元素若算出
    同一个指纹，删除其中一个会把另一个也抹掉，产生假阴性。这正是论文强调
    「删除的元素必须确认在集合中」的原因，也是半排序（semi-sorting）等变体要
    解决的问题。
    """
    num_buckets = next_pow2(compute_num_buckets(n, 4, 0.95))
    cf = CuckooFilter(num_buckets, 4, 8)
    for i in range(n):
        cf.insert(f"item_{i}")

    # 统计指纹冲突：有多少元素与其它元素共享指纹
    fps = [cf._fingerprint(f"item_{i}") for i in range(n)]
    distinct_fp = len(set(fps))

    nq = 50_000
    fpr_before = sum(1 for i in range(nq)
                     if cf.contains(f"nonexistent_{i}")) / nq

    # 删除一半元素
    deleted = 0
    for i in range(0, n, 2):
        if cf.delete(f"item_{i}"):
            deleted += 1

    fpr_after = sum(1 for i in range(nq)
                    if cf.contains(f"nonexistent_{i}")) / nq

    # 假阴性检查：剩余元素必须全部还能查到
    survivors = [i for i in range(1, n, 2)]
    false_negatives = sum(1 for i in survivors
                          if not cf.contains(f"item_{i}"))

    return {
        "n": n,
        "distinct_fingerprints": distinct_fp,
        "fingerprint_collisions": n - distinct_fp,
        "deleted": deleted,
        "survivors_checked": len(survivors),
        "fpr_before_delete": fpr_before,
        "fpr_after_delete": fpr_after,
        "false_negatives": false_negatives,
        "false_negative_rate": false_negatives / len(survivors),
    }


def main():
    print("=== 布谷鸟过滤器实测验证 ===\n")
    t_start = time.time()
    # 固定桶数 4096（2 的幂，cuckoo hashing 的硬要求），b=4，共 16384 槽
    num_buckets = 4096

    # 1. 假阳性率 vs 负载因子
    print("--- 假阳性率 vs 负载因子（f=12 bits, b=4, 4096 桶 / 16384 槽）---")
    load_results = measure_fpr_vs_load(num_buckets, 12, [0.5, 0.75, 0.9, 0.95])
    print(f"{'目标负载':>8}  {'实际负载':>8}  {'n':>7}  {'查询数':>8}  "
          f"{'实测 FPR':>10}  {'理论 FPR':>10}  {'上界':>9}")
    for r in load_results:
        print(f"{r['target_load']:>8.2f}  {r['achieved_load']:>8.3f}  "
              f"{r['n']:>7,}  {r['num_queries']:>8,}  {r['actual_fpr']:>10.5f}  "
              f"{r['theoretical_fpr']:>10.5f}  {r['fpr_upper_bound']:>9.5f}")

    # 2. 假阳性率 vs 指纹位数
    print("\n--- 假阳性率 vs 指纹位数（负载因子 0.95, b=4, 4096 桶）---")
    bits_results = measure_fpr_vs_bits(num_buckets, [4, 8, 12, 16], 0.95)
    print(f"{'f (bits)':>9}  {'n':>7}  {'查询数':>8}  "
          f"{'实测 FPR':>11}  {'理论 FPR':>11}  {'上界':>10}")
    for r in bits_results:
        print(f"{r['fingerprint_bits']:>9}  {r['n']:>7,}  {r['num_queries']:>8,}  "
              f"{r['actual_fpr']:>11.6f}  {r['theoretical_fpr']:>11.6f}  "
              f"{r['fpr_upper_bound']:>10.6f}")

    # 3. 实际可达负载因子（n 只用于推算初始桶数，实际由插入失败决定）
    print("\n--- 实际可达负载因子（持续插入直到失败，f=12 bits）---")
    load_achieved = [measure_achievable_load(num_buckets * 4, b) for b in (2, 4, 8)]
    print(f"{'b':>3}  {'桶数':>8}  {'槽数':>9}  {'实际负载':>9}  {'理论负载':>9}")
    for r in load_achieved:
        print(f"{r['bucket_size']:>3}  {r['num_buckets']:>8,}  {r['num_slots']:>9,}  "
              f"{r['achieved_load']:>9.3f}  {r['theoretical_load']:>9.2f}")

    # 4. 删除语义
    print("\n--- 删除语义验证（n=2000, f=8 bits）---")
    del_res = measure_delete(2000)
    print(f"指纹冲突 {del_res['fingerprint_collisions']} 个 "
          f"（{del_res['n']} 个元素仅 {del_res['distinct_fingerprints']} 个不同指纹）")
    print(f"删除前 FPR={del_res['fpr_before_delete']:.4f}  "
          f"删除 {del_res['deleted']} 个后 FPR={del_res['fpr_after_delete']:.4f}  "
          f"假阴性={del_res['false_negatives']}/{del_res['survivors_checked']}")

    result = {
        "meta": {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "python": sys.version.split()[0],
            "note": "自制迷你布谷鸟过滤器，partial-key cuckoo hashing",
            "num_buckets": num_buckets,
            "bucket_size": 4,
            "num_slots": num_buckets * 4,
            "total_secs": round(time.time() - t_start, 2),
        },
        "fpr_vs_load_factor": load_results,
        "fpr_vs_fingerprint_bits": bits_results,
        "achievable_load_factor": load_achieved,
        "delete_semantics": del_res,
    }

    out = os.path.join(os.path.dirname(__file__), "..", "results",
                       f"cuckoo_filter_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print(f"\n结果写入 {out}")
    print(f"总耗时 {result['meta']['total_secs']} 秒")


if __name__ == "__main__":
    main()
