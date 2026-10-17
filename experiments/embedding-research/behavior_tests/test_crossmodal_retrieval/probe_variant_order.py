# -*- coding: utf-8 -*-
"""判别：shuffle 变体是否随生成顺序变化，以及 gap 中位数为何在两个脚本间不一致。

主脚本 diag_shuffle_inversion 按 shuffle -> swap -> drop 顺序生成，共用
order_hardneg 的模块级 rng（seed 20260927）。pick_examples 只生成 shuffle。
若两者 shuffle 变体字符串不同，则 rng 消耗顺序是成因；若相同，则 gap 计算
本身有差。用法：~/xinfer-env/bin/python probe_variant_order.py
"""
import io, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mm_lab import CrossModalLab
import order_hardneg as OH
from diag_shuffle_inversion import drop_two_content_words

FLICKR_CSV = "/home/ke/crossmodal_lab/datasets/flickr30k/flickr_annotations_30k.csv"
OUT = "/home/ke/crossmodal_lab/results/shuffle_inversion"
rng = __import__("random").Random(20260929)


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

    # 顺序 A：只生成 shuffle（复刻 pick_examples）
    OH.rng = __import__("random").Random(20260927)
    varA = [OH.shuffle_all(c) for c in caps]

    # 顺序 B：shuffle -> swap -> drop（复刻主脚本）
    OH.rng = __import__("random").Random(20260927)
    varB_sh, varB_sw, varB_dr = [], [], []
    for c in caps:
        v = OH.shuffle_all(c)
        if v and v != c:
            varB_sh.append(v)
    for c in caps:
        v = OH.swap_two_content_words(c)
        if v and v != c:
            varB_sw.append(v)
    for c in caps:
        v = drop_two_content_words(c)
        if v and v != c:
            varB_dr.append(v)

    same = sum(1 for a, b in zip(varA, varB_sh) if a == b)
    print(f"顺序A shuffle 变体 {len(varA)} 条，顺序B shuffle 变体 {len(varB_sh)} 条")
    print(f"前 1500 条中字符串相同的有 {same} 条")
    print(f"样例A: {varA[0][:70]}")
    print(f"样例B: {varB_sh[0][:70]}")

    # 用同一批图向量与真句向量，分别算两种顺序的 shuffle gap
    E_img = np.load(os.path.join(OUT, "cache_jina-embeddings-v4_img.npy"))
    E_true = np.load(os.path.join(OUT, "cache_jina-embeddings-v4_true.npy"))
    lab = CrossModalLab("jinaai/jina-embeddings-v4")

    for name, variants in (("A_仅shuffle", varA), ("B_顺序shuffle", varB_sh)):
        if not variants:
            continue
        E_v = lab.encode_text(variants, batch_size=32)
        g = []
        for i in range(min(len(variants), len(E_img))):
            st = cosmat(E_img[i:i + 1], E_true[i:i + 1])[0, 0]
            sf = cosmat(E_img[i:i + 1], E_v[i:i + 1])[0, 0]
            g.append(st - sf)
        g = np.array(g)
        print(f"{name}: n={len(g)} gap 中位 {np.median(g):+.5f} 为正比例 {(g > 0).mean():.3f}")
    lab.close()


if __name__ == "__main__":
    main()
