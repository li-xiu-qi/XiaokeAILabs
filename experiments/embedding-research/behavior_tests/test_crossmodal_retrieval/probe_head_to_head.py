# -*- coding: utf-8 -*-
"""单进程头对头：逐行循环 vs 整矩阵对角线，验证两者等价。

两个发现，一个有用一个记录失败形态。

有用的是对角线与逐行两种写法等价（差 8.88e-16，浮点噪声级），所以主脚本的整矩阵
对角线取法没有 bug，-0.00512 与探针的 -0.037 之差不是算法错误。同时也确认了这些
向量确实已归一化（范数 1.0000），raw dot 与余弦逐位相同。

记录失败形态：本脚本第一版漏 import json，json.loads 的 NameError 被
except Exception: continue 吞掉，1500 行样本全部静默跳过，报的是
"operands could not be broadcast (0,0) (5,2048)"。

最终未解决问题：三种 loop 写法都给 -0.00512，与 probe_sample_dependence 的
-0.03705 差异来源仍未定位，已知与算法、batch、生成顺序都无关。

用法：~/xinfer-env/bin/python scripts/probe_head_to_head.py
"""
import io, json, os, sys

import numpy as np

sys.path.insert(0, "/home/ke/crossmodal_lab/scripts")
import order_hardneg as OH
import random as _random
from mm_lab import CrossModalLab

FLICKR_CSV = "/home/ke/crossmodal_lab/datasets/flickr30k/flickr_annotations_30k.csv"
OUT = "/home/ke/crossmodal_lab/results/shuffle_inversion"


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


def main():
    rows = load(1500)
    caps = [c for _, c in rows]
    E_img = np.load(os.path.join(OUT, "cache_jina-embeddings-v4_img.npy"))
    E_true = np.load(os.path.join(OUT, "cache_jina-embeddings-v4_true.npy"))
    print(f"缓存形状 图 {E_img.shape} 文本 {E_true.shape}")

    # 验证缓存与 rows 对齐：True 文本向量应与该行 caption 重新编码的结果一致
    OH.rng = _random.Random(20260927)
    variants = [OH.shuffle_all(c) for c in caps]
    lab = CrossModalLab("jinaai/jina-embeddings-v4")
    chk = lab.encode_text(caps[:5], batch_size=8)
    print("前 5 条缓存文本向量与重算向量的最大偏差：",
          float(np.max(np.abs(chk - E_true[:5]))))

    E_v = lab.encode_text(variants, batch_size=32)
    lab.close()

    # 方法 1：探针的逐行循环
    g1 = np.array([cosmat(E_img[i:i + 1], E_true[i:i + 1])[0, 0]
                   - cosmat(E_img[i:i + 1], E_v[i:i + 1])[0, 0]
                   for i in range(len(rows))])

    # 方法 2：主脚本的整矩阵对角线
    S_var = cosmat(E_img, E_v)
    S_true = cosmat(E_img, E_true)
    g2 = np.array([S_true[i, i] - S_var[i, i] for i in range(len(rows))])

    # 方法 3：主脚本的写法，但 E_true 用 E_true[idx] 显式重建
    idx = list(range(len(rows)))
    E_t = np.array([E_true[i] for i in idx])
    S_true3 = cosmat(E_img, E_t)
    g3 = np.array([S_true3[i, i] - S_var[i, i] for i in range(len(idx))])

    for name, g in (("方法1 逐行", g1), ("方法2 对角线", g2), ("方法3 显式重建", g3)):
        print(f"{name:14s} 中位 {np.median(g):+.5f} 正比例 {(g > 0).mean():.3f} "
              f"与方法1最大差 {float(np.max(np.abs(g - g1))):.2e}")

    # 关键：检查 S_true 的对角线是否等于逐行结果
    print("S_true 对角线 vs 逐行前 5 项：", np.round(S_true[:5, :5].diagonal(), 5),
          np.round(g1[:5] + np.array([S_var[i, i] for i in range(5)]), 5))
    print("E_img 前 3 行范数：", np.round(np.linalg.norm(E_img[:3], axis=1), 4))
    print("E_true 前 3 行范数：", np.round(np.linalg.norm(E_true[:3], axis=1), 4))


if __name__ == "__main__":
    main()
