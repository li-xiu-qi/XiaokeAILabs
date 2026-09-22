# -*- coding: utf-8 -*-
"""
LSH（局部敏感哈希）实测验证

用随机投影 LSH（SimHash 风格，h(v) = sign(a·v)）自制迷你实现，实测三项：

1. S-curve 形状
   生成指定余弦相似度的向量对，统计实际成为候选对的比例，
   对比理论 S-curve P = 1 - (1 - p^k)^L，p = 1 - θ/π。
   这是 LSH 全部魔力的来源，直接验证 AND-OR 组合公式。

2. 端到端召回
   建库（n 个向量，每个种子带三个已知相似度的邻域向量），用暴力搜索作
   ground truth，逐相似度档位测召回率。直观展示 S-curve 的陡峭性。

3. 距离计算量与墙钟加速比
   统计 LSH 每次查询真正算距离的候选数，对比暴力扫描的 n 次。
   理论预测候选数 ≈ n^(1-ρ)。同时记录墙钟时间，观察两者差异。

方法：numpy 向量化算签名；band 查询用排序加 searchsorted 批量探测。
结果写入 results/lsh_<timestamp>.json
"""
import os
import sys
import json
import time
import math
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from lsh_model import (
    LSHSpec, compute, collision_prob, s_curve, compute_rho, compute_k, compute_L,
)


# ---------------------------------------------------------------- 工具函数

def make_controlled_pairs(rng, d, s, m):
    """生成 m 对余弦相似度恰为 s 的向量对。

    关键：噪声必须先投影到 q 的正交补空间再归一化，否则 v 的方向会被
    噪声的模长带偏，实测余弦相似度会远低于目标值。
    """
    q = rng.standard_normal((m, d))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    noise = rng.standard_normal((m, d))
    noise -= np.sum(noise * q, axis=1, keepdims=True) * q      # 正交化
    noise /= np.linalg.norm(noise, axis=1, keepdims=True)
    v = s * q + math.sqrt(max(0.0, 1.0 - s * s)) * noise
    return q, v


def compute_bits(X, A):
    """对 X 算随机投影签名，返回 (n, k*L) 的布尔位矩阵。"""
    return (X @ A) > 0.0


def bits_to_band_keys(bits, k, L):
    """把 (n, k*L) 的位矩阵按 band 切成 L 组 k 位，压成整数键 (n, L)。

    逐 band 用移位相加，避免 packbits 按字节对齐导致 band 边界错位
    （k 不是 8 的倍数时错位会把相邻 band 混进同一个键）。
    """
    n = bits.shape[0]
    keys = np.zeros((n, L), dtype=np.int64)
    for j in range(L):
        chunk = bits[:, j * k:(j + 1) * k].astype(np.int64)
        key = np.zeros(n, dtype=np.int64)
        for t in range(k):
            key |= chunk[:, t] << t
        keys[:, j] = key
    return keys


def build_band_indices(keys):
    """为每个 band 建有序索引。返回 [(sorted_keys, order), ...]。"""
    indices = []
    for j in range(keys.shape[1]):
        order = np.argsort(keys[:, j], kind="stable")
        indices.append((keys[order, j], order))
    return indices


def probe_candidates(band_indices, keys_q):
    """批量探测：对一批查询返回各自的候选下标集合。

    每个 band 用 searchsorted 在已排序的键数组上定位等值区间，
    把所有 (查询, 候选) 对向量化取出来，避免逐查询的 Python 循环。
    """
    nq = keys_q.shape[0]
    cand_sets = [set() for _ in range(nq)]
    for j, (sorted_keys, order) in enumerate(band_indices):
        qk = keys_q[:, j]
        lo = np.searchsorted(sorted_keys, qk, side="left")
        hi = np.searchsorted(sorted_keys, qk, side="right")
        counts = hi - lo
        total = int(counts.sum())
        if total == 0:
            continue
        q_ids = np.repeat(np.arange(nq), counts)
        starts = np.cumsum(counts) - counts
        within = np.arange(total) - np.repeat(starts, counts)
        positions = np.repeat(lo, counts) + within
        for qi, ci in zip(q_ids.tolist(), order[positions].tolist()):
            cand_sets[qi].add(ci)
    return cand_sets


# ---------------------------------------------------------------- 实验一：S-curve

def measure_s_curve(rng, d, k, L, targets, m_pairs, A):
    """实测不同目标相似度下的候选概率，对比理论 S-curve。"""
    out = []
    for s in targets:
        q, v = make_controlled_pairs(rng, d, s, m_pairs)
        s_meas = float(np.mean(np.sum(q * v, axis=1)))

        bits_q = compute_bits(q, A)
        bits_v = compute_bits(v, A)

        # 逐 band 判断 k 位签名是否完全相同，任一 band 全同即为候选
        hit = np.zeros(m_pairs, dtype=bool)
        for j in range(L):
            hit |= np.all(bits_q[:, j * k:(j + 1) * k]
                          == bits_v[:, j * k:(j + 1) * k], axis=1)
        empirical = float(np.mean(hit))

        out.append({
            "target_s": s,
            "measured_s": round(s_meas, 4),
            "empirical_candidate_rate": empirical,
            "theoretical_rate": s_curve(collision_prob(s_meas), k, L),
        })
    return out


# ---------------------------------------------------------------- 实验二：端到端召回

def build_corpus(rng, n_seeds, d, sim_levels):
    """建库：seeds + 每个 seed 的 sim_levels 个邻域向量。返回 (X, n_seeds)。

    噪声必须投影到 seed 的正交补空间再归一化，否则邻域向量的实际余弦
    相似度远低于目标值（未正交化时实测只有目标值的一小部分）。
    """
    seeds = rng.standard_normal((n_seeds, d))
    seeds /= np.linalg.norm(seeds, axis=1, keepdims=True)
    parts = [seeds]
    for s in sim_levels:
        noise = rng.standard_normal((n_seeds, d))
        noise -= np.sum(noise * seeds, axis=1, keepdims=True) * seeds
        noise /= np.linalg.norm(noise, axis=1, keepdims=True)
        v = s * seeds + math.sqrt(1.0 - s * s) * noise
        parts.append(v)
    return np.vstack(parts)


def run_end_to_end(rng, n_seeds, d, sim_levels, thresholds, n_queries):
    """跑一次端到端召回与耗时测量，返回结果字典。"""
    X = build_corpus(rng, n_seeds, d, sim_levels)
    n = X.shape[0]

    spec_model = LSHSpec(n=n, d=d, s1=0.8, s2=0.4)
    m_model = compute(spec_model)
    k, L = m_model.k, m_model.L
    A = rng.standard_normal((d, k * L))

    t0 = time.perf_counter()
    bits = compute_bits(X, A)
    keys = bits_to_band_keys(bits, k, L)
    band_indices = build_band_indices(keys)
    build_s = time.perf_counter() - t0

    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    q_seeds = rng.choice(n_seeds, size=min(n_queries, n_seeds), replace=False)

    # 每个查询在每个相似度档位上的真值邻居下标
    truth = {}
    for level, s in enumerate(sim_levels):
        offset = (level + 1) * n_seeds
        truth[s] = q_seeds + offset

    recall_rows = []
    lsh_times, bf_times = [], []
    cand_counts = []
    for i, qi in enumerate(q_seeds.tolist()):
        cos_all = Xn @ Xn[qi]

        t0 = time.perf_counter()
        cand = probe_candidates(band_indices, keys[qi:qi + 1])[0]
        cand.discard(qi)                       # 去掉自匹配
        cand_arr = np.fromiter(cand, dtype=np.int64, count=len(cand)) if cand \
            else np.empty(0, dtype=np.int64)
        if cand_arr.size:
            verified = cand_arr[cos_all[cand_arr] >= 0.5]
        else:
            verified = cand_arr
        lsh_times.append(time.perf_counter() - t0)
        cand_counts.append(len(cand))

        t0 = time.perf_counter()
        _ = cos_all >= 0.5
        bf_times.append(time.perf_counter() - t0)

        row = {"query_seed": qi, "candidates": len(cand)}
        for s in sim_levels:
            true_nb = truth[s][i]
            hit = int(np.count_nonzero(verified == true_nb))
            row[f"recall_at_{s}"] = hit
        recall_rows.append(row)

    per_level = []
    for s in sim_levels:
        found = sum(r[f"recall_at_{s}"] for r in recall_rows)
        total = len(recall_rows)
        per_level.append({
            "similarity_level": s,
            "true_neighbors": total,
            "found": found,
            "recall": found / total,
        })

    lsh_us = float(np.mean(lsh_times) * 1e6)
    bf_us = float(np.mean(bf_times) * 1e6)
    avg_cand = float(np.mean(cand_counts))

    return {
        "n": n,
        "k": k,
        "L": L,
        "rho": m_model.rho,
        "theoretical_speedup": m_model.scan_speedup,
        "theoretical_false_candidates": m_model.false_candidates_per_query,
        "build_s": build_s,
        "per_level_recall": per_level,
        "avg_candidates_examined": avg_cand,
        "candidate_to_scan_ratio": n / avg_cand if avg_cand else None,
        "avg_query_us_lsh": lsh_us,
        "avg_query_us_bruteforce": bf_us,
        "measured_speedup": bf_us / lsh_us,
    }


def main():
    rng = np.random.default_rng(42)
    d = 128
    print("=== LSH 实测验证 ===\n")

    # ---------------- 实验一：S-curve 形状 ----------------
    print("--- 实验一：S-curve 形状 ---")
    targets = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    combos = [(4, 16), (8, 32), (16, 64), (30, 970)]
    max_fns = max(k * L for k, L in combos)
    A_big = rng.standard_normal((d, max_fns))

    s_curve_results = {}
    for k, L in combos:
        m_pairs = 400 if (k * L) > 2000 else 2000
        res = measure_s_curve(rng, d, k, L, targets, m_pairs, A_big[:, :k * L])
        s_curve_results[f"k={k},L={L}"] = res
        r08 = next(r for r in res if r["target_s"] == 0.8)
        print(f"(k={k:<2}, L={L:<3})  共 {k*L:>5} 个哈希  "
              f"s=0.8 实测候选率={r08['empirical_candidate_rate']:.4f}  "
              f"理论={r08['theoretical_rate']:.4f}")

    detail_key = "k=16,L=64"
    print(f"\n{detail_key} 明细：")
    print(f"{'目标 s':>8} {'实测 s':>8} {'实测候选率':>12} {'理论候选率':>12}")
    for r in s_curve_results[detail_key]:
        print(f"{r['target_s']:>8.1f} {r['measured_s']:>8.4f} "
              f"{r['empirical_candidate_rate']:>12.4f} {r['theoretical_rate']:>12.4f}")

    # ---------------- 实验二：端到端召回与加速比 ----------------
    print("\n--- 实验二：端到端召回与耗时 ---")
    sim_levels = [0.9, 0.8, 0.7]
    n_queries = 400

    end_to_end = {}
    for n_seeds in [5_000, 20_000]:
        tag = f"n_seeds={n_seeds}"
        print(f"\n[{tag}]  每种子 3 个邻域向量，共 {n_seeds * 4} 个向量")
        res = run_end_to_end(rng, n_seeds, d, sim_levels, None, n_queries)
        end_to_end[tag] = res
        print(f"  模型参数 k={res['k']}, L={res['L']}, ρ={res['rho']:.4f}")
        print(f"  {'档位 s':>8} {'召回率':>8}")
        for row in res["per_level_recall"]:
            print(f"  {row['similarity_level']:>8.1f} {row['recall']:>8.4f}")
        print(f"  平均候选数={res['avg_candidates_examined']:.1f}  "
              f"(vs 暴力扫描 {res['n']} 次，比值 {res['candidate_to_scan_ratio']:.1f}x)")
        print(f"  墙钟：LSH {res['avg_query_us_lsh']:.1f} μs  "
              f"暴力 {res['avg_query_us_bruteforce']:.1f} μs  "
              f"加速比 {res['measured_speedup']:.2f}x")

    # ---------------- 汇总写 JSON ----------------
    result = {
        "meta": {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "note": "自制随机投影 LSH，实测 S-curve 与端到端召回",
            "d": d,
            "seed": 42,
        },
        "s_curve": s_curve_results,
        "end_to_end": end_to_end,
    }

    out = os.path.join(os.path.dirname(__file__), "..", "results",
                       f"lsh_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n结果写入 {out}")


if __name__ == "__main__":
    main()
