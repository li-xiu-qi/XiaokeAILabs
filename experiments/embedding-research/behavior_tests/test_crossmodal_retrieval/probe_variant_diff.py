# -*- coding: utf-8 -*-
"""钉死 probe_sample_dependence 的 -0.037 与 probe_batch_control 的 -0.005 之差。

probe_batch_control 已在四种编码方式、五个 batch 档位下确认 jina-v4 的 shuffle
gap 稳定在 -0.004 到 -0.0099。而 probe_sample_dependence 报的是 -0.031 到 -0.038，
比它大一个量级。两者的样本集、rng 种子、模型、缓存都相同，唯一没交叉验证的是
生成的变体字符串是否为同一批。

本脚本把几条路径生成的变体逐条比对：
  A probe_batch_control 的路径：load 之后重置 rng，直接生成
  B probe_sample_dependence 的路径：load 之后重置 rng，直接生成（看起来同 A）
  C 主脚本 diag_shuffle_inversion 的路径：module 级 rng，三种 kind 依次生成

三者的 rng 都是同一个模块对象，差别只在调用时机与前置消耗。若变体字符串一致而
gap 不同，则问题在别处（缓存、编码）；若变体不一致，则 -0.037 那批来自不同的
扰动集合，读数不可比。

用法：~/xinfer-env/bin/python scripts/probe_variant_diff.py
"""
import io, json, os, random, sys

import numpy as np

sys.path.insert(0, "/home/ke/crossmodal_lab/scripts")
FLICKR_CSV = "/home/ke/crossmodal_lab/datasets/flickr30k/flickr_annotations_30k.csv"


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


rows = load(1500)
caps = [c for _, c in rows]
print(f"样本 {len(rows)} 首行 {rows[0][0]} 末行 {rows[-1][0]}")

import order_hardneg as OH

# A：先重置再生成
OH.rng = random.Random(20260927)
A = [OH.shuffle_all(c) for c in caps]

# B：复刻 probe_sample_dependence。它 main() 里也是先重置再生成
OH.rng = random.Random(20260927)
B = [OH.shuffle_all(c) for c in caps]

# C：模拟 diag_shuffle_inversion，module 级 rng 先被 swap 消耗再生成 shuffle
OH.rng = random.Random(20260927)
for c in caps:
    OH.swap_two_content_words(c)
C = [OH.shuffle_all(c) for c in caps]

for name, v in (("A 直接生成", A), ("B 同 A 路径", B), ("C swap 消耗后再生成", C)):
    print(f"{name}: None 数 {sum(1 for x in v if x is None)}")
    print(f"   前 2 条：{v[0][:60]} | {v[1][:60]}")

print(f"A vs B 相同 {sum(1 for a, b in zip(A, B) if a == b)}/{len(A)}")
print(f"A vs C 相同 {sum(1 for a, b in zip(A, C) if a == b)}/{len(A)}")
