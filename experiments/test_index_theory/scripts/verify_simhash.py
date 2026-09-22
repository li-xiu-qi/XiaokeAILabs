# -*- coding: utf-8 -*-
"""
SimHash 实测验证

自制 SimHash（随机超平面投影，符号位作哈希），实测三项：

1. θ/π 关系
   生成指定余弦相似度的向量对，计算 SimHash 海明距离，
   对比理论 E[h] = d × θ/π 与 σ = sqrt(d × (θ/π)(1 - θ/π))。
   这是 SimHash 全部性质的来源。

2. 位数对稳定性的影响
   在 d = 64/128/256 位上重复测量，验证相对标准差随位数下降。

3. 桶半径召回
   给定相似度阈值 s0 与半径 r，实测实际召回率，
   对比模型的正态近似 P(Binomial(d, θ0/π) ≤ r)。

方法：numpy 向量化算签名与海明距离。
结果写入 results/simhash_<timestamp>.json
"""
import os
import sys
import json
import math
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from simhash_model import (
    SimHashSpec, compute, expected_hamming, hamming_std, recall_at_radius,
)


def make_controlled_pairs(rng, d, s, m):
    """生成 m 对余弦相似度恰为 s 的向量对。

    噪声必须先投影到 q 的正交补空间再归一化，否则 v 的方向被噪声模长带偏，
    实测余弦相似度会远低于目标值（未正交化时实测只有目标值的一小部分）。
    """
    q = rng.standard_normal((m, d))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    noise = rng.standard_normal((m, d))
    noise -= np.sum(noise * q, axis=1, keepdims=True) * q      # 正交化
    noise /= np.linalg.norm(noise, axis=1, keepdims=True)
    v = s * q + math.sqrt(max(0.0, 1.0 - s * s)) * noise
    return q, v


def simhash_sign(X, A):
    """SimHash 签名：X @ A 的符号位。返回 (n, num_bits) bool。"""
    return (X @ A) > 0.0


def hamming_distances(sig_q, sig_v):
    """逐对海明距离。返回 (m,) int。"""
    return np.count_nonzero(sig_q != sig_v, axis=1)


def main():
    rng = np.random.default_rng(7)
    print("=== SimHash 实测验证 ===\n")

    # ---------------- 实验一：θ/π 关系 ----------------
    print("--- 实验一：θ/π 关系（不同相似度下的海明距离）---")
    targets = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    m_pairs = 4000
    bit_configs = [64, 128, 256]

    theta_pi_results = {}
    for num_bits in bit_configs:
        A = rng.standard_normal((128, num_bits))
        rows = []
        for s in targets:
            q, v = make_controlled_pairs(rng, 128, s, m_pairs)
            cos = np.sum(q * v, axis=1)
            s_meas = float(np.mean(cos))

            sig_q = simhash_sign(q, A)
            sig_v = simhash_sign(v, A)
            h = hamming_distances(sig_q, sig_v)

            theta = math.acos(max(-1.0, min(1.0, s_meas)))
            rows.append({
                "target_s": s,
                "measured_s": round(s_meas, 4),
                "mean_hamming": float(np.mean(h)),
                "std_hamming": float(np.std(h)),
                "theo_mean": expected_hamming(num_bits, theta),
                "theo_std": hamming_std(num_bits, theta),
            })
        theta_pi_results[f"bits={num_bits}"] = rows

    # 打印 bits=128 的详细对比
    print("\nbits=128 的实测 vs 理论：")
    print(f"{'目标 s':>8} {'实测 s':>8} {'实测均值':>10} {'理论均值':>10} "
          f"{'实测σ':>8} {'理论σ':>8}")
    for r in theta_pi_results["bits=128"]:
        print(f"{r['target_s']:>8.2f} {r['measured_s']:>8.4f} "
              f"{r['mean_hamming']:>10.3f} {r['theo_mean']:>10.3f} "
              f"{r['std_hamming']:>8.3f} {r['theo_std']:>8.3f}")

    # ---------------- 实验二：位数对相对标准差的影响 ----------------
    print("\n--- 实验二：位数对相对标准差的影响（s=0.8）---")
    s_test = 0.8
    q, v = make_controlled_pairs(rng, 128, s_test, 6000)
    bit_stability = []
    for num_bits in bit_configs:
        A = rng.standard_normal((128, num_bits))
        sig_q = simhash_sign(q, A)
        sig_v = simhash_sign(v, A)
        h = hamming_distances(sig_q, sig_v)
        rel_std = float(np.std(h)) / float(np.mean(h))
        bit_stability.append({
            "num_bits": num_bits,
            "mean_hamming": float(np.mean(h)),
            "std_hamming": float(np.std(h)),
            "relative_std": rel_std,
        })
        print(f"bits={num_bits:>3}  均值={np.mean(h):>7.3f}  σ={np.std(h):>6.3f}  "
              f"相对σ={rel_std:.4f}")

    # ---------------- 实验三：桶半径召回 ----------------
    print("\n--- 实验三：桶半径召回（自动半径 r = round(d × θ0/π)）---")
    num_bits = 64
    A = rng.standard_normal((128, num_bits))
    spec = SimHashSpec(n=1_000_000, d=128, num_bits=num_bits, s0=0.8)
    m = compute(spec)

    # 公平对比：对每个目标相似度 s，用「该 s 对应的自动半径」测召回，
    # 再与模型 recall_at_radius(bits, arccos(s), r_auto) 对比。
    # 不能用固定阈值 0.8 的半径去测 s=0.95 的对——那些对的期望海明距离
    # 本来就小得多，召回必然偏高，那是混入了更高相似度的样本。
    probe_targets = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]
    radius_rows = []
    for s in probe_targets:
        theta = math.acos(s)
        r_auto = max(1, round(expected_hamming(num_bits, theta)))
        tp = fn = 0
        for _ in range(6):                  # 多轮抽样稳定估计
            q2, v2 = make_controlled_pairs(rng, 128, s, 2500)
            sig_q = simhash_sign(q2, A)
            sig_v = simhash_sign(v2, A)
            h = hamming_distances(sig_q, sig_v)
            in_radius = h <= r_auto
            cos = np.sum(q2 * v2, axis=1)
            is_true = cos >= s - 1e-9      # 真值：实测余弦 ≥ 目标 s
            tp += int(np.count_nonzero(in_radius & is_true))
            fn += int(np.count_nonzero(~in_radius & is_true))
        recall = tp / max(1, tp + fn)
        theo = recall_at_radius(num_bits, theta, r_auto)
        radius_rows.append({
            "target_s": s,
            "auto_radius": r_auto,
            "expected_hamming": expected_hamming(num_bits, theta),
            "recall": recall,
            "theoretical_recall": theo,
        })
        print(f"s={s:.2f}  自动半径 r={r_auto:>2}  期望海明={expected_hamming(num_bits, theta):>5.2f}  "
              f"实测召回={recall:.4f}  理论={theo:.4f}")

    # 固定阈值 s0=0.8、扫半径的对照（说明半径对召回的杠杆作用）
    print(f"\n固定 s0=0.8 扫半径（期望海明距离 "
          f"{expected_hamming(num_bits, math.acos(0.8)):.2f}）：")
    scan_rows = []
    for r in [8, 10, 12, 13, 14, 16, 19]:
        tp = fn = 0
        for s in [0.85, 0.90, 0.95, 0.99]:
            q2, v2 = make_controlled_pairs(rng, 128, s, 2500)
            sig_q = simhash_sign(q2, A)
            sig_v = simhash_sign(v2, A)
            h = hamming_distances(sig_q, sig_v)
            in_radius = h <= r
            cos = np.sum(q2 * v2, axis=1)
            is_true = cos >= 0.8
            tp += int(np.count_nonzero(in_radius & is_true))
            fn += int(np.count_nonzero(~in_radius & is_true))
        recall = tp / max(1, tp + fn)
        scan_rows.append({"radius": r, "recall": recall})
        print(f"  r={r:>2}  实测召回={recall:.4f}")

    print(f"\n模型自动半径 r={m.r}（期望海明距离 {m.expected_hamming:.2f}）")

    # ---------------- 汇总写 JSON ----------------
    result = {
        "meta": {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "note": "自制 SimHash，实测 θ/π 关系与桶半径召回",
            "d": 128,
            "seed": 7,
        },
        "theta_pi_relation": theta_pi_results,
        "bit_stability": bit_stability,
        "radius_recall": radius_rows,
        "radius_scan": scan_rows,
        "model_default_radius": m.r,
    }

    out = os.path.join(os.path.dirname(__file__), "..", "results",
                       f"simhash_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n结果写入 {out}")


if __name__ == "__main__":
    main()
