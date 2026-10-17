# -*- coding: utf-8 -*-
"""差分：为什么同样的 OH.rng 重置，探针与主脚本给出不同的 shuffle 变体。

不编码模型，只打字符串。三条路径：
  P1 只 import order_hardneg，main 里重置 rng 后生成（复刻 probe_sample_dependence）
  P2 import mm_lab -> order_hardneg，模块级重置 rng，再 import diag_shuffle_inversion
  后生成（复刻修好的 pick_examples）
  P3 import run_experiments -> order_hardneg，不重置，直接生成（复刻主脚本）
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

# P1
import order_hardneg as OH1
OH1.rng = random.Random(20260927)
v1 = [OH1.shuffle_all(c) for c in caps]

# P2：模拟不同导入次序
import importlib
OH2 = importlib.import_module("order_hardneg")
import mm_lab  # noqa
OH2.rng = random.Random(20260927)
import diag_shuffle_inversion  # noqa
OH3 = importlib.import_module("order_hardneg")
print("P2 导入 diag_shuffle_inversion 后 OH.rng 是否被换对象：",
      OH3.rng is not OH2.rng)
OH3.rng = random.Random(20260927)
v2 = [OH3.shuffle_all(c) for c in caps]

# P3：主脚本路径，不重置
import run_experiments as R  # noqa
OH4 = importlib.import_module("order_hardneg")
v3 = [OH4.shuffle_all(c) for c in caps]

for name, v in (("P1 探测针路径", v1), ("P2 主脚本路径", v2), ("P3 未重置", v3)):
    print(f"{name}: 前 3 条变体")
    for s in v[:3]:
        print(f"    {s[:90]}")

print("P1 vs P2 相同条数:", sum(1 for a, b in zip(v1, v2) if a == b))
print("P1 vs P3 相同条数:", sum(1 for a, b in zip(v1, v3) if a == b))
print("P2 vs P3 相同条数:", sum(1 for a, b in zip(v2, v3) if a == b))
