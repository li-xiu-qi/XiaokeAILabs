# -*- coding: utf-8 -*-
"""判定：同一批文本，编码顺序是否影响结果。

现象：同一批 shuffle 变体、同一份图/文本缓存，probe_sample_dependence 给 shuffle
gap 中位 -0.03705，probe_head_to_head 给 -0.00512。两者唯一差别是后者在编码变体
之前先编码了 5 条 caption。若模型对编码顺序敏感，则 D 组与 shuffle 两组读数都不
可比，因为效应量（0.037）与顺序噪声同量级。

四个检查：
  1 同批文本连续编码两次，余弦是否一致（纯确定性）
  2 先编码一批别的文本，再编码目标批，余弦是否变化（顺序/状态泄漏）
  3 batch_size 8 与 32 对同一批文本的结果差异
  4 图向量是否同样受影响
用法：~/xinfer-env/bin/python probe_determinism.py
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


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def report(tag, A, B):
    d = np.array([cos(A[i], B[i]) for i in range(len(A))])
    print(f"{tag:38s} 余弦差 中位 {np.median(1 - d):+.5f} 最大 {np.max(1 - d):+.5f}")


def main():
    rows = load(1500)
    caps = [c for _, c in rows]
    lab = CrossModalLab("jinaai/jina-embeddings-v4")

    # 1 纯确定性：同批文本编码两次
    sub = caps[:200]
    A = lab.encode_text(sub, batch_size=32)
    B = lab.encode_text(sub, batch_size=32)
    report("1 同批编码两次", A, B)

    # 2 顺序效应：先编码别的，再编码目标批
    C = lab.encode_text(caps[200:400], batch_size=32)
    D = lab.encode_text(sub, batch_size=32)
    report("2 先编别的再编目标批", A, D)

    # 3 batch_size 效应
    E = lab.encode_text(sub, batch_size=8)
    report("3 batch 8 对 batch 32", A, E)
    F = lab.encode_text(sub, batch_size=64)
    report("3 batch 64 对 batch 32", A, F)

    # 4 与落盘缓存比
    E_true = np.load(os.path.join(OUT, "cache_jina-embeddings-v4_true.npy"))
    report("4 新编码 batch32 对落盘缓存", A, E_true[:200])

    # 5 图向量顺序效应
    paths = [os.path.join("/home/ke/crossmodal_lab/datasets/flickr30k/images/"
                          "flickr30k-images", nm) for nm, _ in rows[:200]]
    IA = lab.encode_image(paths, batch_size=8)
    IB = lab.encode_image(paths, batch_size=8)
    report("5 图同批编码两次", IA, IB)
    E_img = np.load(os.path.join(OUT, "cache_jina-embeddings-v4_img.npy"))
    report("5 图新编码对落盘缓存", IA, E_img[:200])
    lab.close()


if __name__ == "__main__":
    main()
