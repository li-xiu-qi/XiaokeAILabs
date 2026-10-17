# -*- coding: utf-8 -*-
"""从可用子集里挑余弦口径的乱序句示例。⚠️ 读数不可信，勿引用其输出。

两个问题。一是 rng：本脚本的导入链比 diag_shuffle_inversion 多一层，不显式重置
order_hardneg 的模块级 rng 就生成变体，得到的集合与主实验不同，曾据此报出
-0.03734（与独立复现的 -0.03705 纯属巧合，路径不同）。二是索引：E_v 一度按
cands 内位置索引而 E_img 按原始行号索引，只要有行被过滤就错位。两处都已修，
但修完之后暴露了更根本的问题，见 probe_determinism.py。

为什么读数不可信：本脚本测的 shuffle gap 中位数 -0.00512 与另一路径的
-0.03954 相差八倍，差异来源是编码时的 batch 配置（同批文本 batch 8 对 batch 32
余弦差中位 0.037、最大 0.357），与被测效应同量级。也就是说这里的 gap 数字随
batch 设置在 -0.005 到 -0.039 之间摆动，无法归因给模型。

要定位这类现象，先读 probe_determinism.py，把 batch 固定并换多档取稳定性。
保留此文件记录前两次失败的具体形态。
"""
import argparse, io, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mm_lab import CrossModalLab
import order_hardneg as OH
# 必须显式重置。shuffle_all / swap_two_content_words 用 order_hardneg 的模块级
# rng，本脚本的导入链比 diag_shuffle_inversion 多一层（还 import 了后者），
# 不重置时 rng 起点不同，产出的变体集合与主实验不一致。
import random as _random
OH.rng = _random.Random(20260927)
from order_hardneg import shuffle_all, swap_two_content_words
from diag_shuffle_inversion import drop_two_content_words

FLICKR_CSV = "/home/ke/crossmodal_lab/datasets/flickr30k/flickr_annotations_30k.csv"
FLICKR_IMG = "/home/ke/crossmodal_lab/datasets/flickr30k/images/flickr30k-images"
OUT = "/home/ke/crossmodal_lab/results/shuffle_inversion"

KINDS = {"shuffle": shuffle_all, "swap": swap_two_content_words,
         "drop": drop_two_content_words}


def load(n):
    """与 diag_shuffle_inversion.main 完全一致：取排序后前 n 条，不用
    run_experiments.load_flickr（它按 seed 随机采样，会与缓存错位）。"""
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="jinaai/jina-embeddings-v4")
    ap.add_argument("--kind", default="shuffle")
    args = ap.parse_args()

    rows = load(1500)
    tag = args.model.split("/")[-1].replace(".", "_")
    E_img = np.load(os.path.join(OUT, f"cache_{tag}_img.npy"))
    E_true = np.load(os.path.join(OUT, f"cache_{tag}_true.npy"))

    fn = KINDS[args.kind]
    cands = []
    for i, (nm, cap) in enumerate(rows):
        v = fn(cap)
        if v and v != cap:
            cands.append((i, nm, cap, v))
    print(f"样本 {len(cands)}")

    lab = CrossModalLab(args.model)
    E_v_all = lab.encode_text([c[3] for c in cands], batch_size=32)
    lab.close()
    vok = ~np.isnan(E_v_all).any(axis=1)
    if not vok.all():
        print(f"文本侧剔除 {int((~vok).sum())} 条 NaN")
    # E_v 按原始行号索引（与 E_img/E_true 同一坐标系），不用 cands 内位置
    E_v = {}
    for c, e, k in zip(cands, E_v_all, vok):
        if k:
            E_v[c[0]] = e
    keep = [c for c in cands if c[0] in E_v]

    gaps = []
    for (i, nm, cap, v) in keep:
        s_t = cosmat(E_img[i:i + 1], E_true[i:i + 1])[0, 0]
        s_f = cosmat(E_img[i:i + 1], E_v[i].reshape(1, -1))[0, 0]
        gaps.append(s_t - s_f)
    gaps = np.array(gaps)
    order = np.argsort(gaps)

    picks = [order[len(order) // 2], order[0], order[-1]]
    seen = set()
    for pos, p in enumerate(picks):
        if p in seen:
            continue
        seen.add(p)
        i, nm, cap, v = keep[p]
        st = cosmat(E_img[i:i + 1], E_true[i:i + 1])[0, 0]
        sf = cosmat(E_img[i:i + 1], E_v[i].reshape(1, -1))[0, 0]
        rec = {"model": args.model, "kind": args.kind, "file": nm,
               "label": ["中位附近", "最负（偏袒乱序）", "最正（惩罚乱序）"][pos],
               "true": cap, "false": v,
               "sim_true": round(float(st), 4), "sim_false": round(float(sf), 4),
               "gap": round(float(gaps[p]), 4)}
        print(json.dumps(rec, ensure_ascii=False))

    print(f"\n本子集余弦 gap 中位 {np.median(gaps):+.5f}  为正比例 {(gaps > 0).mean():.3f}")
    outp = os.path.join(OUT, f"examples_{args.model.split('/')[-1]}_{args.kind}.json")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump({"gap_median": float(np.median(gaps)),
                   "positive_rate": float((gaps > 0).mean())}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
