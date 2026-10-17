# -*- coding: utf-8 -*-
"""缓存 vs 现算 的差异是整体旋转还是局部扰动。

已确认：
  - 同一进程内 T1/T2/T3（纯文本、图像后文本、再编一次）两两余弦差全为 0
  - 300 条里 279 条与缓存差 <0.001，21 条差到 0.25
  - 坏样本词数中位与全体相同（17），模长比 1.0000
  - 图像编码不污染后续文本编码

所以偏差不是调用次序、不是长度、不是模长。剩下两种形态：
  A 整体旋转：整套缓存相对现算有一个共同的方向偏移，则任意两条的
    相对余弦差应该相近，且缓存内部结构与现算内部结构一致
  B 局部扰动：只有部分样本被换掉，缓存内部结构在这些样本上与现算不一致

判据：算 缓存内部两两余弦 与 现算内部两两余弦 的差。若 A，差应接近 0
（因为共同旋转保持相对关系）；若 B，在这些样本对上会显著偏离。

另外直接查一个我一直没验的可能：缓存的第 i 行是否其实是另一条 caption
的编码（错位）。用缓存第 i 行 对 现算全库做最近邻，看它匹配的是第几条。

用法：~/xinfer-env/bin/python scripts/probe_cache_geometry.py
"""
import io, json, os, sys

import numpy as np

sys.path.insert(0, "/home/ke/crossmodal_lab/scripts")
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


def main():
    rows = load(1500)
    caps = [c for _, c in rows]
    E_c = np.load(os.path.join(OUT, f"cache_{TAG}_true.npy"))

    lab = CrossModalLab(f"jinaai/{TAG}")
    E_f = lab.encode_text(caps, batch_size=32)
    lab.close()

    n = len(caps)
    print(f"样本 {n}", flush=True)

    # 逐样本差
    d = 1 - cosmat(E_c, E_f).diagonal()
    bad = np.where(d > 1e-3)[0]
    print(f"\n逐样本余弦差：中位 {np.median(d):.6f} 最大 {d.max():.6f} "
          f">0.001 的 {len(bad)}/{n}", flush=True)

    # 缓存第 i 行在现算全库里的最近邻（查错位）
    S = cosmat(E_c[:200], E_f)
    nn = np.argmax(S, axis=1)
    nnd = 1 - S[np.arange(len(nn)), nn]
    hit_self = int((nn == np.arange(len(nn))).sum())
    print(f"\n前 200 条：缓存行对现算全库最近邻 = 自身的有 {hit_self}/200",
          flush=True)
    print(f"  最近邻余弦差 中位 {np.median(nnd):.6f} 最大 {nnd.max():.6f}",
          flush=True)
    mism = np.where(nn != np.arange(len(nn)))[0][:8]
    for i in mism:
        print(f"  [{i}] 最近邻是第 {nn[i]} 条 "
              f"(差 {nnd[i]:.4f})：{caps[nn[i]][:50]}", flush=True)

    # 整体旋转 vs 局部扰动：缓存内部结构 vs 现算内部结构
    rng = np.random.default_rng(0)
    pairs = rng.integers(0, n, (4000, 2))
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    Sc, Sf = cosmat(E_c, E_c), cosmat(E_f, E_f)
    dc = Sc[pairs[:, 0], pairs[:, 1]]
    df = Sf[pairs[:, 0], pairs[:, 1]]
    dd = np.abs(dc - df)
    print(f"\n随机 4000 对样本的内部两两余弦差（|缓存-现算|）："
          f"中位 {np.median(dd):.6f} 最大 {dd.max():.6f}", flush=True)
    print("  → 中位接近 0 说明整体结构一致（支持整体旋转），"
          "显著非 0 说明内部结构被改（支持局部扰动）", flush=True)

    # 投影均值差：若整体旋转，应有共同偏移方向
    mu_c = E_c.mean(axis=0)
    mu_f = E_f.mean(axis=0)
    print(f"\n均值向量余弦 "
          f"{float(mu_c @ mu_f / (np.linalg.norm(mu_c) * np.linalg.norm(mu_f))):.6f}",
          flush=True)
    print(f"均值模长 cached {np.linalg.norm(mu_c):.4f} fresh "
          f"{np.linalg.norm(mu_f):.4f}", flush=True)

    # 坏样本与其他样本的关系：坏样本的缓存向量到现算均值的距离
    good = d <= 1e-3
    print(f"\n坏样本 {int((~good).sum())} 条，好样本 {int(good.sum())} 条",
          flush=True)
    if bad.size:
        An = E_f / np.linalg.norm(E_f, axis=1, keepdims=True)
        mu = An[good].mean(axis=0)
        mu = mu / np.linalg.norm(mu)
        for name, E in (("缓存", E_c), ("现算", E_f)):
            En = E / np.linalg.norm(E, axis=1, keepdims=True)
            g = En[good] @ mu
            b = En[bad] @ mu
            print(f"  {name}：到好样本均值余弦 好样本中位 "
                  f"{float(np.median(g)):.4f}  坏样本中位 "
                  f"{float(np.median(b)):.4f}", flush=True)


if __name__ == "__main__":
    main()
