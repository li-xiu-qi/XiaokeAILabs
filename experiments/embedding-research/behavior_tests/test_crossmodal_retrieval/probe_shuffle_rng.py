# -*- coding: utf-8 -*-
"""拆开两个混淆因素：样本集 vs shuffle 变体的 rng 状态。

n=1000 上 jina-v4 order-shuffle 余弦 gap 中位 -0.02079（正比例 36.5%），
n=1500 上是 -0.00512（正比例 47.5%）。两者差在三倍以上。两处差异：

  A 样本集：前 1000 条 vs 前 1500 条（都是排序后取前 n）
  B rng 状态：diag_hardneg_cosine 先跑 swap 消耗 1000 次抽样再生成 shuffle；
    diag_shuffle_inversion 与 pick_examples 都是 shuffle 先生成

本脚本固定 A（只用前 1000 条，复用已落盘的 1500 缓存），只改 B，读 gap 中位数。
若两种 rng 状态下中位数接近，说明 rng 不是原因，差异在样本集；
若中位数相差大，说明「反向偏好」对抽种敏感，不是模型的稳定属性。

用法：~/xinfer-env/bin/python probe_shuffle_rng.py
"""
import io, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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


def gap_stats(E_img, E_true, E_v):
    g = np.array([cosmat(E_img[i:i + 1], E_true[i:i + 1])[0, 0]
                  - cosmat(E_img[i:i + 1], E_v[i:i + 1])[0, 0]
                  for i in range(len(E_img))])
    return float(np.median(g)), float((g > 0).mean())


def main():
    rows = load(1000)          # 固定样本集：排序后前 1000 条
    caps = [c for _, c in rows]
    E_img = np.load(os.path.join(OUT, "cache_jina-embeddings-v4_img.npy"))[:1000]
    E_true = np.load(os.path.join(OUT, "cache_jina-embeddings-v4_true.npy"))[:1000]
    lab = CrossModalLab("jinaai/jina-embeddings-v4")

    # B1: rng 全新，shuffle 先生成（复刻 diag_shuffle_inversion / pick_examples）
    OH.rng = _random.Random(20260927)
    v1 = [OH.shuffle_all(c) for c in caps]
    E_v1 = lab.encode_text(v1, batch_size=32)

    # B2: rng 先被 swap 消耗 1000 次抽样，再生成 shuffle（复刻 diag_hardneg_cosine）
    OH.rng = _random.Random(20260927)
    for c in caps:
        OH.swap_two_content_words(c)
    v2 = [OH.shuffle_all(c) for c in caps]
    E_v2 = lab.encode_text(v2, batch_size=32)

    # B3: 换一个完全不同的种子，看是否还在同一量级
    OH.rng = _random.Random(777)
    v3 = [OH.shuffle_all(c) for c in caps]
    E_v3 = lab.encode_text(v3, batch_size=32)
    lab.close()

    print(f"样本 {len(rows)} 张图")
    for label, v, E_v in (("B1 rng全新·shuffle先生成", v1, E_v1),
                          ("B2 swap消耗后再生shuffle", v2, E_v2),
                          ("B3 种子777", v3, E_v3)):
        diff = sum(1 for a, b in zip(v1, v) if a != b)
        med, pos = gap_stats(E_img, E_true, E_v)
        print(f"{label}: 与B1变体不同 {diff}/{len(v)} 条, "
              f"gap 中位 {med:+.5f}, 正比例 {pos:.3f}")


if __name__ == "__main__":
    main()
