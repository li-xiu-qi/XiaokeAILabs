# -*- coding: utf-8 -*-
"""
Join 算法实测验证

用纯 Python 自制三种 Join 算法，实测 CPU 时间：
1. Nested Loop Join：双重循环
2. Hash Join：构建哈希表 + 探测
3. Sort-Merge Join：排序 + 归并

方法：生成两个模拟表（随机整数 join 键），用三种算法 join，对比 CPU 时间。
结果写入 results/join_<timestamp>.json
"""
import os
import sys
import json
import time
import random
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from join_model import JoinSpec, nlj_metrics, hash_join_metrics, sort_merge_join_metrics, compute


def generate_tables(n_r: int, n_s: int, seed: int = 42) -> tuple:
    """生成两个表。R 和 S 有部分重叠的 join 键。"""
    random.seed(seed)
    # 用较小的键空间，确保有 join 结果
    key_space = max(n_r, n_s) // 2
    r = [(random.randint(0, key_space), random.random()) for _ in range(n_r)]
    s = [(random.randint(0, key_space), random.random()) for _ in range(n_s)]
    return r, s


def nested_loop_join(r: list, s: list) -> tuple:
    """Nested Loop Join。外层 R，内层 S。"""
    start = time.perf_counter()
    result = []
    for r_key, r_val in r:
        for s_key, s_val in s:
            if r_key == s_key:
                result.append((r_key, r_val, s_val))
    elapsed = time.perf_counter() - start
    return result, elapsed


def hash_join(r: list, s: list) -> tuple:
    """Hash Join。构建 S 的哈希表，探测 R。"""
    start = time.perf_counter()
    # 构建阶段
    hash_table = {}
    for s_key, s_val in s:
        if s_key not in hash_table:
            hash_table[s_key] = []
        hash_table[s_key].append(s_val)
    # 探测阶段
    result = []
    for r_key, r_val in r:
        if r_key in hash_table:
            for s_val in hash_table[r_key]:
                result.append((r_key, r_val, s_val))
    elapsed = time.perf_counter() - start
    return result, elapsed


def sort_merge_join(r: list, s: list) -> tuple:
    """Sort-Merge Join。先排序，再线性归并。"""
    start = time.perf_counter()
    # 排序阶段
    r_sorted = sorted(r, key=lambda x: x[0])
    s_sorted = sorted(s, key=lambda x: x[0])
    # 归并阶段
    result = []
    i = j = 0
    while i < len(r_sorted) and j < len(s_sorted):
        r_key, r_val = r_sorted[i]
        s_key, s_val = s_sorted[j]
        if r_key == s_key:
            # 收集所有匹配的 S
            j_start = j
            while j < len(s_sorted) and s_sorted[j][0] == r_key:
                result.append((r_key, r_val, s_sorted[j][1]))
                j += 1
            i += 1
            j = j_start  # 重置 j，因为下一个 R 可能也匹配同一个 S
        elif r_key < s_key:
            i += 1
        else:
            j += 1
    elapsed = time.perf_counter() - start
    return result, elapsed


def main():
    print("=== Join 算法实测验证 ===\n")

    # 用小参数快速验证
    spec = JoinSpec(
        n_r=5_000,
        n_s=2_000,
        row_size=100,
        page_size=4096,
        memory_pages=1000,
    )

    print(f"参数: R={spec.n_r:,} 行, S={spec.n_s:,} 行\n")

    # 生成表
    r, s = generate_tables(spec.n_r, spec.n_s)

    # NLJ
    print("--- Nested Loop Join ---")
    result_nlj, time_nlj = nested_loop_join(r, s)
    print(f"结果 {len(result_nlj):,} 行, CPU 时间 {time_nlj:.3f} s")
    nlj_theo = nlj_metrics(spec)
    print(f"理论: CPU {nlj_theo['cpu_comparisons']:,} 次比较, "
          f"I/O {nlj_theo['io_reads']:,} 次读")

    # Hash Join
    print("\n--- Hash Join ---")
    result_hj, time_hj = hash_join(r, s)
    print(f"结果 {len(result_hj):,} 行, CPU 时间 {time_hj:.3f} s")
    hj_theo = hash_join_metrics(spec)
    print(f"理论: CPU {hj_theo['cpu_hashes']:,} 次哈希, "
          f"I/O {hj_theo['io_reads']:,} 次读, "
          f"内存 {hj_theo['memory_bytes']/1024:.1f} KiB")

    # Sort-Merge Join
    print("\n--- Sort-Merge Join ---")
    result_smj, time_smj = sort_merge_join(r, s)
    print(f"结果 {len(result_smj):,} 行, CPU 时间 {time_smj:.3f} s")
    smj_theo = sort_merge_join_metrics(spec)
    print(f"理论: CPU {smj_theo['cpu_comparisons']:,} 次比较, "
          f"I/O {smj_theo['io_reads']:,} 次读")

    # 验证结果一致
    assert len(result_nlj) == len(result_hj) == len(result_smj), "三种 Join 结果行数应一致"
    print(f"\n✓ 三种 Join 结果行数一致: {len(result_nlj):,}")

    # 对比
    print("\n--- 性能对比 ---")
    print(f"NLJ:       {time_nlj:.3f} s  (基准)")
    print(f"Hash Join: {time_hj:.3f} s  ({time_nlj/time_hj:.1f}x 加速)")
    print(f"Sort-Merge:{time_smj:.3f} s  ({time_nlj/time_smj:.1f}x 加速)")

    # 汇总写入 JSON
    result = {
        "meta": {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "python": sys.version.split()[0],
            "note": "自制三种 Join 算法，小参数快速验证公式",
            "spec": {
                "n_r": spec.n_r,
                "n_s": spec.n_s,
                "row_size": spec.row_size,
                "page_size": spec.page_size,
                "memory_pages": spec.memory_pages,
            },
            "join_result_size": len(result_nlj),
        },
        "actual": {
            "nlj": {"result_size": len(result_nlj), "time_s": time_nlj},
            "hash_join": {"result_size": len(result_hj), "time_s": time_hj},
            "sort_merge": {"result_size": len(result_smj), "time_s": time_smj},
        },
        "theoretical": {
            "nlj": nlj_theo,
            "hash_join": hj_theo,
            "sort_merge": smj_theo,
        },
    }

    out = os.path.join(os.path.dirname(__file__), "..", "results",
                       f"join_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n结果写入 {out}")


if __name__ == "__main__":
    main()
