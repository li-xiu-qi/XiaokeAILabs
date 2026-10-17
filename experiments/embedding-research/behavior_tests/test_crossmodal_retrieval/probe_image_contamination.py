# -*- coding: utf-8 -*-
"""机制验证：先编码图像，会不会改变同一实例上后续的文本编码。

probe_cache_vs_fresh 已确认：diag_shuffle_inversion 落盘的 E_true 缓存与纯文本
进程现算的真句向量，1500 条全部不一致（余弦差中位 0.090、最大 0.420）。用缓存算
jina-v4 的 shuffle gap 得 -0.03960，用现算得 -0.01449，相差 0.025。

唯一未被检验过的差异是那次的调用次序：缓存那次是「先生成三批扰动文本 → 加载
模型 → 编码 1500 张图（311 秒）→ 编码真句」，而纯文本探针是「加载模型 → 编码
文本」。probe_determinism 测过「先编别的文本再编目标批」不影响结果，但从未测过
「先编码图像再编文本」。

本脚本在同一进程内做三格对照：
  T1  只编码文本
  T2  先编码若干图像，再编码同一批文本
  T3  先编码文本，再编码图像，最后再编码同一批文本（看是否被图像调用改变）
三格两两比对，直接判定图像编码是否污染后续文本编码。

用法：~/xinfer-env/bin/python scripts/probe_image_contamination.py
"""
import io, json, os, sys

import numpy as np

sys.path.insert(0, "/home/ke/crossmodal_lab/scripts")
from mm_lab import CrossModalLab

FLICKR_CSV = "/home/ke/crossmodal_lab/datasets/flickr30k/flickr_annotations_30k.csv"
IMG_ROOT = ("/home/ke/crossmodal_lab/datasets/flickr30k/images/"
            "flickr30k-images")


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


def cmp(tag, A, B):
    d = 1 - np.array([cosmat(A[i:i + 1], B[i:i + 1])[0, 0] for i in range(len(A))])
    print(f"{tag:34s} 中位 {np.median(d):.6f}  均值 {d.mean():.6f}  "
          f"最大 {d.max():.6f}  >0.001 的条数 {int((d > 1e-3).sum())}/{len(d)}",
          flush=True)


def main():
    rows = load(300)
    caps = [c for _, c in rows]
    paths = [os.path.join(IMG_ROOT, nm) for nm, _ in rows]

    lab = CrossModalLab("jinaai/jina-embeddings-v4")

    # T1：纯文本
    T1 = lab.encode_text(caps, batch_size=32)

    # T2：先编码图像，再编码同一批文本
    I = lab.encode_image(paths, batch_size=8)
    T2 = lab.encode_text(caps, batch_size=32)

    # T3：再编码一次文本（此时图像已编过两轮）
    T3 = lab.encode_text(caps, batch_size=32)

    lab.close()

    cmp("T1 纯文本 vs T2 图像后文本", T1, T2)
    cmp("T2 图像后 vs T3 再编一次", T2, T3)
    cmp("T1 纯文本 vs T3 再编一次", T1, T3)

    E_true_c = np.load("/home/ke/crossmodal_lab/results/shuffle_inversion/"
                       "cache_jina-embeddings-v4_true.npy")
    E_img_c = np.load("/home/ke/crossmodal_lab/results/shuffle_inversion/"
                      "cache_jina-embeddings-v4_img.npy")
    print("", flush=True)
    cmp("T1 纯文本 vs 落盘缓存", T1, E_true_c[:len(caps)])
    cmp("T2 图像后 vs 落盘缓存", T2, E_true_c[:len(caps)])
    cmp("本次图像 vs 落盘图像缓存", I, E_img_c[:len(caps)])
    print(f"\n本次图向量 NaN 行数 {int(np.isnan(I).any(axis=1).sum())}/"
          f"{len(I)}，缓存图向量 NaN 行数 "
          f"{int(np.isnan(E_img_c[:len(I)]).any(axis=1).sum())}", flush=True)

    # 稀疏大偏差的形态要归因到具体机制，不能只报有 7% 偏了。
    d = 1 - np.array([cosmat(T1[i:i + 1], E_true_c[i:i + 1])[0, 0]
                      for i in range(len(caps))])
    bad = np.where(d > 1e-3)[0]
    print("", flush=True)
    print("偏差 > 0.001 的明细 共 %d 条" % len(bad), flush=True)
    for i in bad:
        toks = caps[i].split()
        print("  [%d] diff=%.5f %d词 norm_cached=%.4f norm_fresh=%.4f | %s"
              % (i, d[i], len(toks), np.linalg.norm(E_true_c[i]),
                 np.linalg.norm(T1[i]), caps[i][:64]), flush=True)
    if len(bad):
        lens = [len(caps[i].split()) for i in bad]
        alll = [len(c.split()) for c in caps]
        print("  坏样本词数 %s" % sorted(lens), flush=True)
        print("  全部词数中位 %d，坏样本词数中位 %d"
              % (int(np.median(alll)), int(np.median(lens))), flush=True)
        print("  坏样本索引 %s" % bad.tolist(), flush=True)
        print("  坏样本索引 mod 32 %s" % [int(i) % 32 for i in bad],
              flush=True)
        r = [np.linalg.norm(E_true_c[i]) / max(np.linalg.norm(T1[i]), 1e-9)
             for i in bad]
        print("  坏样本 norm 比 cached/fresh 中位 %.4f"
              % float(np.median(r)), flush=True)


if __name__ == "__main__":
    main()
