# -*- coding: utf-8 -*-
"""最后一处未验证变量：落盘的 E_true 缓存与现算是否一致，以及它是否就是那个
跨脚本差异的来源。

已确认的事实清单：
  - 同批文本、同 batch size 连续编码两次，逐位一致（0.00000）
  - 先编码别的文本再编码目标批，结果不变（0.00000）
  - batch 8 对 batch 32，余弦差中位 0.037、最大 0.357
  - 换扰动生成顺序，gap 摆动仅 0.00065
  - 图侧同批两次编码差为 0，新编对缓存差 0.0002

未追的一处：probe_determinism 显示「新编码 batch32 对落盘缓存」余弦差中位 0.00000
但最大 0.14902。中位为 0 而最大值 0.15 意味着偏差稀疏但剧烈，这种形态没被解释过。
而脚本读数恰好按「用缓存 E_true」与「现算 E_true」分成两簇：
  现算 E_true：probe_batch_control -0.005 上下、diag_shuffle_inversion v2 -0.00512
  用缓存 E_true：probe_sample_dependence -0.03705、probe_generation_order -0.01939

本脚本在同一进程内：
  1 现算 1500 条真句向量（text-only、batch 32）
  2 与落盘缓存逐样本比余弦差，看偏差的分布形态（稀疏还是系统）
  3 分别用「缓存 E_true」与「现算 E_true」算 shuffle gap，看是否分开两簇
若第 3 步能把 -0.005 与 -0.019 分开，则缓存即差异源，且要解释缓存为何与现算不同。

用法：~/xinfer-env/bin/python scripts/probe_cache_vs_fresh.py
"""
import io, json, os, sys

import numpy as np

sys.path.insert(0, "/home/ke/crossmodal_lab/scripts")
import order_hardneg as OH
import random as _random
from mm_lab import CrossModalLab

FLICKR_CSV = "/home/ke/crossmodal_lab/datasets/flickr30k/flickr_annotations_30k.csv"
OUT = "/home/ke/crossmodal_lab/results/shuffle_inversion"
TAG = "jina-embeddings-v4"


def load(n):
    rows = []
    with io.open(FLICKR_CSV, encoding="utf-8") as f:
        import csv
        for row in csv.DictReader(f):
            try:
                caps = json.loads(row["raw"])
            except Exception:
                continue
            caps = [c.strip() for c in caps if c and c.strip()]
            if caps:
                rows.append((row["filename"], caps[0]))
    rows.sort()
    return rows[:n]


def cosmat(A, B):
    An = A / np.linalg.norm(A, axis=1, keepdims=True)
    Bn = B / np.linalg.norm(B, axis=1, keepdims=True)
    return An @ Bn.T


def gap_of(E_img, E_t, E_v):
    g = np.array([cosmat(E_img[i:i + 1], E_t[i:i + 1])[0, 0]
                  - cosmat(E_img[i:i + 1], E_v[i:i + 1])[0, 0]
                  for i in range(len(E_img))])
    return float(np.median(g)), float((g > 0).mean()), g


def main():
    rows = load(1500)
    caps = [c for _, c in rows]
    E_img_c = np.load(os.path.join(OUT, f"cache_{TAG}_img.npy"))
    E_true_c = np.load(os.path.join(OUT, f"cache_{TAG}_true.npy"))
    print(f"样本 {len(rows)}  缓存 图 {E_img_c.shape} 文本 {E_true_c.shape}", flush=True)

    OH.rng = _random.Random(20260927)
    variants = [OH.shuffle_all(c) for c in caps]

    lab = CrossModalLab(f"jinaai/{TAG}")

    # 1/2 现算真句与乱序句（text-only，batch 32）
    E_true_f = lab.encode_text(caps, batch_size=32)
    E_v = lab.encode_text(variants, batch_size=32)
    lab.close()

    # 缓存与现算的逐样本余弦差
    d = np.array([cosmat(E_true_c[i:i + 1], E_true_f[i:i + 1])[0, 0]
                  for i in range(len(caps))])
    diff = 1 - d
    print(f"\n缓存 vs 现算 余弦差：中位 {np.median(diff):.6f}  "
          f"均值 {diff.mean():.6f}  最大 {diff.max():.6f}", flush=True)
    for thr in (1e-4, 1e-3, 1e-2):
        print(f"  余弦差 > {thr:g} 的样本数：{int((diff > thr).sum())}/{len(diff)}",
              flush=True)

    # 偏差最大的样本长什么样
    idx = np.argsort(-diff)[:5]
    print("\n偏差最大的 5 条：", flush=True)
    for i in idx:
        print(f"  [{i}] diff={diff[i]:.5f} len={len(caps[i].split())}词 "
              f"| {caps[i][:70]}", flush=True)

    # 3 两种 E_true 分别算 gap
    for name, Et in (("缓存 E_true", E_true_c), ("现算 E_true", E_true_f)):
        med, pos, _ = gap_of(E_img_c, Et, E_v)
        print(f"\n用{name}算 shuffle gap：中位 {med:+.5f}  正比例 {pos:.3f}",
              flush=True)

    # 4 只用偏差小的样本子集再算一次，看是否收敛
    keep = diff < 1e-4
    print(f"\n仅用缓存与现算一致的 {int(keep.sum())} 条：", flush=True)
    for name, Et in (("缓存 E_true", E_true_c), ("现算 E_true", E_true_f)):
        med, pos, _ = gap_of(E_img_c[keep], Et[keep], E_v[keep])
        print(f"  {name}：中位 {med:+.5f}  正比例 {pos:.3f}", flush=True)


if __name__ == "__main__":
    main()
