# -*- coding: utf-8 -*-
"""
MinHash 实测验证

自制 MinHash（用 k 组 (a, b) 乘性哈希模拟 k 个独立置换），实测三项：

1. Jaccard 估计精度
   生成指定 Jaccard 相似度的集合对，用 k 个 MinHash 估计 J，
   对比真值与理论标准差 σ = sqrt(J(1-J)/k)。

2. k 对精度的影响
   在 k = 64/128/256 上重复，验证 σ ∝ 1/sqrt(k)。

3. LSH band 的 S-curve
   统计不同 Jaccard 下成为候选对的比例，对比理论 1 - (1 - J^r)^b。

方法：universe 固定，用随机置换构造集合对；哈希值用质数取模的乘性哈希。
结果写入 results/minhash_<timestamp>.json
"""
import os
import sys
import json
import math
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from minhash_model import (
    MinHashSpec, compute, estimate_std, band_prob, compute_optimal_banding,
)


# ---------------------------------------------------------------- 工具函数

PRIME = (1 << 61) - 1          # 梅森质数，作取模基数


def make_hash_coeffs(rng, k):
    """生成 k 组 (a, b) 乘性哈希系数。a 必须非零。"""
    a = rng.integers(1, PRIME, size=k, dtype=np.int64)
    b = rng.integers(0, PRIME, size=k, dtype=np.int64)
    return a, b


def element_hashes(elem_ids, a, b):
    """计算每个元素的 k 个哈希值。elem_ids (m,) -> (m, k) int64。"""
    # 用 int64 乘法后取模，避免溢出：拆成 32 位乘
    ids = elem_ids.astype(np.int64).reshape(-1, 1)
    return (a * ids + b) % PRIME


def make_controlled_sets(rng, universe_size, set_size, j_target, m):
    """生成 m 对 Jaccard 相似度约为 j_target 的集合。

    由 J = c/(2a-c) 反解交集大小 c = 2aJ/(1+J)，
    每对用一个独立随机置换取前 a 个作 A，前 c 个作交集，
    再从 A 之后取 a-c 个补成 B（保证 B 的其余元素不在 A 中）。
    返回 (m, a) 的集合 A 下标与 (m, a) 的集合 B 下标。
    """
    a = set_size
    c = int(round(2.0 * a * j_target / (1.0 + j_target)))
    c = max(0, min(a, c))

    # m 个独立随机置换：对 (m, U) 随机矩阵 argsort
    rand_mat = rng.random((m, universe_size))
    perm = np.argsort(rand_mat, axis=1)

    set_a = perm[:, :a]                    # (m, a)
    overlap = perm[:, :c]                  # (m, c) ⊂ A
    rest = perm[:, a:a + (a - c)]          # (m, a-c) ⊂ U\A
    set_b = np.concatenate([overlap, rest], axis=1)
    return set_a, set_b, c


def minhash_signature(set_idx, elem_hash, k):
    """对一个集合算 k 维 MinHash 签名：逐置换取最小哈希值。"""
    # set_idx: (a,) 元素下标；elem_hash: (U, k)
    return elem_hash[set_idx].min(axis=0)


# ---------------------------------------------------------------- 实验一：估计精度

def measure_accuracy(rng, universe_size, set_size, k_list, j_targets, m_pairs):
    """实测不同 J 与 k 下的 MinHash 估计误差。"""
    a, b = make_hash_coeffs(rng, max(k_list))
    elem_hash = element_hashes(np.arange(universe_size), a, b)   # (U, k_max)

    out = {}
    for k in k_list:
        hk = elem_hash[:, :k]
        rows = []
        for j in j_targets:
            set_a, set_b, c = make_controlled_sets(
                rng, universe_size, set_size, j, m_pairs)
            # 真值 Jaccard
            true_j = c / (2 * set_size - c)

            est = np.empty(m_pairs)
            for i in range(m_pairs):
                sa = minhash_signature(set_a[i], hk, k)
                sb = minhash_signature(set_b[i], hk, k)
                est[i] = np.count_nonzero(sa == sb) / k

            rows.append({
                "target_j": round(j, 3),
                "true_j": round(true_j, 4),
                "mean_estimate": float(np.mean(est)),
                "std_estimate": float(np.std(est)),
                "theoretical_std": estimate_std(k, true_j),
            })
        out[f"k={k}"] = rows
    return out


# ---------------------------------------------------------------- 实验三：S-curve

def measure_band_s_curve(rng, universe_size, set_size, k, b, r, j_targets, m_pairs):
    """实测不同 J 下成为候选对的比例，对比理论 S-curve。"""
    a, bb = make_hash_coeffs(rng, k)
    elem_hash = element_hashes(np.arange(universe_size), a, bb)

    rows = []
    for j in j_targets:
        set_a, set_b, c = make_controlled_sets(
            rng, universe_size, set_size, j, m_pairs)
        true_j = c / (2 * set_size - c)

        n_cand = 0
        for i in range(m_pairs):
            sa = minhash_signature(set_a[i], elem_hash, k)
            sb = minhash_signature(set_b[i], elem_hash, k)
            same = (sa == sb)
            # 任一带内 r 行全同即为候选
            hit = False
            for band in range(b):
                if np.all(same[band * r:(band + 1) * r]):
                    hit = True
                    break
            n_cand += int(hit)

        empirical = n_cand / m_pairs
        rows.append({
            "target_j": round(j, 3),
            "true_j": round(true_j, 4),
            "empirical_candidate_rate": empirical,
            "theoretical_rate": band_prob(true_j, r, b),
        })
    return rows


def main():
    rng = np.random.default_rng(123)
    print("=== MinHash 实测验证 ===\n")

    universe_size = 20_000
    set_size = 200
    j_targets = [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9]
    k_list = [64, 128, 256]
    m_pairs = 400

    # ---------------- 实验一：估计精度 ----------------
    print("--- 实验一：Jaccard 估计精度 ---")
    acc = measure_accuracy(rng, universe_size, set_size, k_list, j_targets, m_pairs)

    print(f"\nuniverse={universe_size}  set_size={set_size}  m_pairs={m_pairs}")
    print(f"{'目标J':>7} {'真值J':>7} | {'k=64 实测':>14} {'理论σ':>8} | "
          f"{'k=128 实测':>14} {'理论σ':>8} | {'k=256 实测':>14} {'理论σ':>8}")
    for i, j in enumerate(j_targets):
        r64 = acc["k=64"][i]
        r128 = acc["k=128"][i]
        r256 = acc["k=256"][i]
        print(f"{j:>7.2f} {r64['true_j']:>7.4f} | "
              f"{r64['mean_estimate']:>8.4f}±{r64['std_estimate']:<5.4f} {r64['theoretical_std']:>8.4f} | "
              f"{r128['mean_estimate']:>8.4f}±{r128['std_estimate']:<5.4f} {r128['theoretical_std']:>8.4f} | "
              f"{r256['mean_estimate']:>8.4f}±{r256['std_estimate']:<5.4f} {r256['theoretical_std']:>8.4f}")

    # 无偏性检查：估计均值应接近真值
    print("\n无偏性检查（k=128，均值 vs 真值）：")
    for i, j in enumerate(j_targets):
        r = acc["k=128"][i]
        bias = r["mean_estimate"] - r["true_j"]
        print(f"  J={r['true_j']:.4f}  估计均值={r['mean_estimate']:.4f}  偏差={bias:+.4f}")

    # ---------------- 实验二：k 对精度的影响 ----------------
    print("\n--- 实验二：σ 随 k 的变化（J=0.5 附近）---")
    j_mid = 0.5
    k_scaling = []
    for k in [32, 64, 128, 256, 512]:
        res = measure_accuracy(rng, universe_size, set_size, [k], [j_mid], m_pairs)
        r = res[f"k={k}"][0]
        k_scaling.append({
            "k": k,
            "std_estimate": r["std_estimate"],
            "theoretical_std": r["theoretical_std"],
            "sigma_sqrt_k": r["std_estimate"] * math.sqrt(k),
        })
        print(f"k={k:>3}  实测σ={r['std_estimate']:.5f}  理论σ={r['theoretical_std']:.5f}  "
              f"σ×√k={r['std_estimate']*math.sqrt(k):.5f}")

    # ---------------- 实验三：LSH band S-curve ----------------
    print("\n--- 实验三：LSH band S-curve ---")
    spec = MinHashSpec(n=1_000_000, avg_set_size=set_size, k=128, j_threshold=0.8)
    m = compute(spec)
    k, b, r = m.k, m.b, m.r
    print(f"k={k}  自动标定 b={b}, r={r}  (拐点 J=0.8)")

    curve = measure_band_s_curve(rng, universe_size, set_size, k, b, r,
                                 [0.2, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
                                 m_pairs=600)
    print(f"\n{'目标J':>7} {'真值J':>7} {'实测候选率':>12} {'理论候选率':>12}")
    for row in curve:
        print(f"{row['target_j']:>7.2f} {row['true_j']:>7.4f} "
              f"{row['empirical_candidate_rate']:>12.4f} {row['theoretical_rate']:>12.4f}")

    # 阈值以上召回
    rows_hi = [row for row in curve if row["true_j"] >= 0.8]
    if rows_hi:
        avg_recall = sum(row["empirical_candidate_rate"] for row in rows_hi) / len(rows_hi)
        print(f"\nJ ≥ 0.8 的平均候选率（召回）={avg_recall:.4f}  "
              f"理论拐点值 1-1/e={1 - 1/math.e:.4f}")

    # ---------------- 汇总写 JSON ----------------
    result = {
        "meta": {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "note": "自制 MinHash，实测 Jaccard 估计精度与 LSH band S-curve",
            "universe_size": universe_size,
            "set_size": set_size,
            "m_pairs": m_pairs,
            "seed": 123,
        },
        "accuracy": acc,
        "k_scaling": k_scaling,
        "band_s_curve": curve,
        "model_banding": {"k": k, "b": b, "r": r, "j_threshold": 0.8},
    }

    out = os.path.join(os.path.dirname(__file__), "..", "results",
                       f"minhash_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n结果写入 {out}")


if __name__ == "__main__":
    main()
