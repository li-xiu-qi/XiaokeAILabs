# -*- coding: utf-8 -*-
"""验证最后一处变量：扰动生成的调用顺序。

已确认的事实：
  - order_hardneg 的 shuffle_all / swap_two_content_words 共用同一个模块级 rng
  - 同一样本集、同一种子 20260927，只是生成顺序不同，产出的乱序句集合完全不同
    （换序后 1500 条无一相同）
  - 主脚本 diag_shuffle_inversion 按 shuffle -> swap -> drop 顺序生成，shuffle
    拿到的是全新 rng 状态
  - 探针脚本把 shuffle 排在 swap 消耗之后，拿到的是被消耗过的状态
  - 两者给出的 jina-v4 shuffle gap 中位差 0.034（-0.005 对 -0.039）

本脚本在同一进程、同一份缓存、同一编码设置下，只切换生成顺序，看 gap 是否随之
改变。若改变，则「随机扰动集合的身份」本身就是一个能改变读数一个量级的变量，
必须固定并报告；若不改变，则差异另有来源。

注意：CSV 解析必须显式 import json。第一次写这个脚本时漏了，json.loads 抛的
NameError 被 except Exception: continue 吞掉，1500 行样本全部静默跳过，脚本打印
"样本 0" 却照常往下跑。这是同一个模式第二次犯错，见 probe_head_to_head.py。

用法：~/xinfer-env/bin/python scripts/probe_generation_order.py --model jinaai/jina-embeddings-v4
"""
import argparse, io, json, os, sys

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


def gap(E_img, E_t, E_v):
    g = np.array([cosmat(E_img[i:i + 1], E_t[i:i + 1])[0, 0]
                  - cosmat(E_img[i:i + 1], E_v[i:i + 1])[0, 0]
                  for i in range(len(E_img))])
    return float(np.median(g)), float((g > 0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="jinaai/jina-embeddings-v4")
    ap.add_argument("--n", type=int, default=1500)
    args = ap.parse_args()

    rows = load(args.n)
    caps = [c for _, c in rows]
    tag = args.model.split("/")[-1].replace(".", "_")
    E_img = np.load(os.path.join(
        OUT, f"cache_{tag}_img.npy"))
    print(f"样本 {len(rows)} 图缓存 {E_img.shape}", flush=True)

    lab = CrossModalLab(args.model)

    variants = {}
    # O1：shuffle 先生成（主脚本顺序，rng 全新）
    OH.rng = _random.Random(20260927)
    variants["O1_shuffle_first"] = [OH.shuffle_all(c) for c in caps]
    # O2：swap 先消耗，shuffle 后生成（探针脚本的顺序）
    OH.rng = _random.Random(20260927)
    for c in caps:
        OH.swap_two_content_words(c)
    variants["O2_shuffle_after_swap"] = [OH.shuffle_all(c) for c in caps]
    # O3：shuffle 自身先消耗一轮，再正式生成。同一函数、同一批样本，
    # 只是 rng 状态被自己推进过，用来排除「O2 的差异来自 swap 这个动作」
    OH.rng = _random.Random(20260927)
    for c in caps:
        OH.shuffle_all(c)
    variants["O3_after_self_consume"] = [OH.shuffle_all(c) for c in caps]

    E_true = np.load(os.path.join(OUT, f"cache_{tag}_true.npy"))
    meds = []
    for name, v in variants.items():
        E_v = lab.encode_text(v, batch_size=32)
        med, pos = gap(E_img, E_true, E_v)
        meds.append(med)
        print(f"{name:26s} 变体唯一 {len(set(v))}/{len(v)}  "
              f"gap 中位 {med:+.5f}  正比例 {pos:.3f}", flush=True)
    print(f"\n仅换生成顺序，gap 中位摆动 {max(meds) - min(meds):.5f}",
          flush=True)
    lab.close()


if __name__ == "__main__":
    main()
