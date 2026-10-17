# -*- coding: utf-8 -*-
"""batch 配置混淆的受控实验：gap 是否随编码设置摆动，摆动多大。

背景。同一样本集、同一批 shuffle 变体、同一份图向量缓存，两条路径给出 jina-v4
的 shuffle gap 中位 -0.00512 与 -0.03954，相差八倍。probe_determinism.py 已定位
到原因是编码时的 batch 配置（同批文本 batch 8 对 batch 32，余弦差中位 0.037、最大
0.357）。本实验把这一项当成唯一的自变量扫一遍，看 gap 的摆动幅度。

四种编码方式：

  sep   true 与 variant 分两次独立编码（复刻现有流水线）
  mix   true 与 variant 交错放进同一次调用，两侧批量组成完全相同
        这是消除该变量的改法：差别只可能来自文本内容，不可能来自批次构成
  pair  [true_i, variant_i] 作为两条一批，每对处在完全相同的批次上下文里
        最严格，代价是要单独编码，只能做子集
  img   图向量在各 batch 设置下重编码，验证图侧是否同样受影响

判据：若 sep 跨 batch 设置的 gap 摆动幅度大于 mix 与 pair 的摆动幅度，则混淆
成立，sep 的读数不可用；若 mix 与 pair 的 gap 在 batch 设置间稳定，则该值是可信
的现象幅度。

用法：
  ~/xinfer-env/bin/python scripts/probe_batch_control.py \
      --model jinaai/jina-embeddings-v4 --n 1500
  ~/xinfer-env/bin/python scripts/probe_batch_control.py \
      --model BAAI/BGE-VL-base --n 1500 --pairs 400
"""
import argparse, io, json, os, sys

import numpy as np

sys.path.insert(0, "/home/ke/crossmodal_lab/scripts")
import order_hardneg as OH
import random as _random
from mm_lab import CrossModalLab

FLICKR_CSV = "/home/ke/crossmodal_lab/datasets/flickr30k/flickr_annotations_30k.csv"
OUT = "/home/ke/crossmodal_lab/results/batch_control"
BATCHES = [8, 16, 32, 64, 128]


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
    return float(np.median(g)), float((g > 0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="jinaai/jina-embeddings-v4")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--pairs", type=int, default=400,
                    help="pair 方式的子集大小")
    args = ap.parse_args()

    rows = load(args.n)
    caps = [c for _, c in rows]
    tag = args.model.split("/")[-1].replace(".", "_")
    E_img = np.load(f"/home/ke/crossmodal_lab/results/shuffle_inversion/"
                    f"cache_{tag}_img.npy")
    print(f"样本 {len(rows)} 图缓存 {E_img.shape}", flush=True)

    OH.rng = _random.Random(20260927)
    variants = [OH.shuffle_all(c) for c in caps]

    lab = CrossModalLab(args.model)
    res = {"model": args.model, "n": len(rows), "batches": {}, "pair": {},
           "image_check": {}}

    # --- sep：分两次独立编码，扫 batch ---
    for bs in BATCHES:
        ET = lab.encode_text(caps, batch_size=bs)
        EV = lab.encode_text(variants, batch_size=bs)
        med, pos = gap_of(E_img, ET, EV)
        res["batches"][bs] = {"gap_median": round(med, 5),
                              "positive_rate": round(pos, 4),
                              "mean_cos_true": round(float(np.mean(
                                  np.sum(E_img / np.linalg.norm(E_img, axis=1,
                                                                keepdims=True)
                                         * ET / np.linalg.norm(ET, axis=1,
                                                               keepdims=True),
                                         axis=1))), 5),
                              "mean_cos_var": round(float(np.mean(
                                  np.sum(E_img / np.linalg.norm(E_img, axis=1,
                                                                keepdims=True)
                                         * EV / np.linalg.norm(EV, axis=1,
                                                               keepdims=True),
                                         axis=1))), 5)}
        print(f"sep  batch {bs:4d}  gap 中位 {med:+.5f}  正比例 {pos:.3f}"
              f"  图-真均值 {res['batches'][bs]['mean_cos_true']:.4f}"
              f"  图-变体均值 {res['batches'][bs]['mean_cos_var']:.4f}",
              flush=True)

    # --- mix：交错进同一次调用，两侧批量组成相同 ---
    inter = [x for pair in zip(caps, variants) for x in pair]
    for bs in (32, 64):
        E = lab.encode_text(inter, batch_size=bs)
        ET, EV = E[0::2], E[1::2]
        med, pos = gap_of(E_img, ET, EV)
        key = f"mix_{bs}"
        res["pair"][key] = {"gap_median": round(med, 5),
                            "positive_rate": round(pos, 4)}
        print(f"mix  batch {bs:4d}  gap 中位 {med:+.5f}  正比例 {pos:.3f}",
              flush=True)

    # --- pair：[true_i, variant_i] 两条一批，最严格 ---
    m = min(args.pairs, len(rows))
    pairs = []
    for i in range(m):
        pairs.extend([caps[i], variants[i]])
    E = lab.encode_text(pairs, batch_size=m * 2)
    ET, EV = E[0::2], E[1::2]
    med, pos = gap_of(E_img[:m], ET, EV)
    res["pair"]["pair_one_call"] = {"gap_median": round(med, 5),
                                    "positive_rate": round(pos, 4),
                                    "n": m}
    print(f"pair 一次调用 n={m}  gap 中位 {med:+.5f}  正比例 {pos:.3f}",
          flush=True)

    # --- 图侧重编码检查 ---
    img_root = ("/home/ke/crossmodal_lab/datasets/flickr30k/images/"
                "flickr30k-images")
    paths = [os.path.join(img_root, nm) for nm, _ in rows[:200]]
    for bs in (8, 32):
        IE = lab.encode_image(paths, batch_size=bs)
        d = np.array([cosmat(IE[i:i + 1], E_img[i:i + 1])[0, 0]
                      for i in range(len(IE))])
        res["image_check"][bs] = {
            "cos_diff_median": round(float(np.median(1 - d)), 6),
            "cos_diff_max": round(float(np.max(1 - d)), 6)}
        print(f"图 batch {bs:4d} 对缓存 余弦差 中位 "
              f"{np.median(1 - d):.6f} 最大 {np.max(1 - d):.6f}", flush=True)
    lab.close()

    gaps = [v["gap_median"] for v in res["batches"].values()]
    res["sep_swing"] = round(max(gaps) - min(gaps), 5)
    print(f"\nsep 跨 batch 摆动幅度 {res['sep_swing']:.5f}", flush=True)

    os.makedirs(OUT, exist_ok=True)
    outp = os.path.join(OUT, f"batch_control_{tag}.json")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"落盘 -> {outp}", flush=True)


if __name__ == "__main__":
    main()
