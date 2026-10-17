# -*- coding: utf-8 -*-
"""定位「乱序句反向偏好」住在哪个样本段。

n=1000（排序后前 1000 条）jina-v4 shuffle gap 中位 -0.0395，正比例 25.5%；
n=1500（排序后前 1500 条）中位 -0.0051，正比例 47.5%。前 1000 是前 1500 的
子集，加进来的 500 条必须把中位抬高了 0.034。Flickr 的 annotations CSV 按
filename 排序，所以「前 n 条」不是随机样本，而是字母表上的连续区间。

本脚本对同一条 1500 条序列逐样本算 shuffle gap，按索引段与随机抽样分别汇总。

方法论结论仍然有效，已核实：五种样本定义（前 1000、1000-1500、前 1500、随机
200、随机 1000）给出的中位数全部落在 -0.031 到 -0.038、正比例 25% 到 30%，高度
一致。所以这个效应不依赖样本段，CSV 按 filename 排序造成的连续区间不是原因。

但绝对值不可引用。后来 probe_determinism.py 定位到真正的驱动因素是编码时的
batch 配置：同批文本 batch 8 对 batch 32，余弦差中位 0.037、最大 0.357。本脚本
的读数只是某一个 batch 设置下的结果，同一样本集换 batch 会变成 -0.0395，与主
脚本的 -0.0051 相差八倍。所以这里能说的是「段间无差异」，不能说「效应有多大」。

用法：~/xinfer-env/bin/python probe_sample_dependence.py
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


def stats(g, tag):
    g = np.asarray(g)
    print(f"{tag:28s} n={len(g):5d}  gap 中位 {np.median(g):+.5f}  "
          f"正比例 {(g > 0).mean():.3f}  均值 {g.mean():+.5f}")


def main():
    rows = load(1500)
    caps = [c for _, c in rows]
    E_img = np.load(os.path.join(OUT, "cache_jina-embeddings-v4_img.npy"))
    E_true = np.load(os.path.join(OUT, "cache_jina-embeddings-v4_true.npy"))
    print(f"首条文件名 {rows[0][0]}  第1000条 {rows[999][0]}  第1500条 {rows[1499][0]}")

    OH.rng = _random.Random(20260927)
    variants = [OH.shuffle_all(c) for c in caps]
    lab = CrossModalLab("jinaai/jina-embeddings-v4")
    E_v = lab.encode_text(variants, batch_size=32)
    lab.close()

    g = np.array([cosmat(E_img[i:i + 1], E_true[i:i + 1])[0, 0]
                  - cosmat(E_img[i:i + 1], E_v[i:i + 1])[0, 0]
                  for i in range(len(rows))])

    stats(g[:1000], "前 1000 条（旧 D 组）")
    stats(g[1000:1500], "第 1000-1500 条")
    stats(g, "前 1500 条")
    stats(g[:500], "前 500 条")
    stats(g[500:1000], "第 500-1000 条")

    r = _random.Random(0)
    for n in (200, 1000):
        idx = r.sample(range(1500), n)
        stats(g[idx], f"全序列随机抽 {n} 条")

    # 文件名分布，看前 1000 是否集中的某些前缀
    pref = {}
    for nm, _ in rows[:1000]:
        pref[nm[:3]] = pref.get(nm[:3], 0) + 1
    print(f"前 1000 条文件名前 3 位分布（前 8 种）："
          f"{sorted(pref.items(), key=lambda x: -x[1])[:8]}")
    pref2 = {}
    for nm, _ in rows[1000:1500]:
        pref2[nm[:3]] = pref2.get(nm[:3], 0) + 1
    print(f"第 1000-1500 条文件名前 3 位分布（前 8 种）："
          f"{sorted(pref2.items(), key=lambda x: -x[1])[:8]}")


if __name__ == "__main__":
    main()
