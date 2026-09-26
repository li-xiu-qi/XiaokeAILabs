# -*- coding: utf-8 -*-
"""补采 T2T 分布，供画图用。

diag_cohesion.py 只打印汇总值，没落盘，画不出分布。
本脚本取 Flickr30k 前 N 张图，编码每图的 5 条 caption，保存两类相似度：

  t2t_rel   同一张图内两条不同 caption 的相似度（同模态相关）
  t2t_unrel 不同图之间 caption 的相似度（同模态无关基线）

T2I 的 rel / unrel 已在 results/<模型名>/a_retrieval.csv，本脚本不重复采。

用法：collect_t2t.py [N] [模型id]
"""
import csv as _csv
import json
import os
import sys
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mm_lab import CrossModalLab
import run_experiments as R

N = int(sys.argv[1]) if len(sys.argv) > 1 else 500
MODEL = sys.argv[2] if len(sys.argv) > 2 else "jinaai/jina-embeddings-v4"
RES = R.results_dir_for(MODEL)
os.makedirs(RES, exist_ok=True)
OUT = os.path.join(RES, "t2t_hist.txt")

img_root = R.ensure_flickr_images()
csv_path = os.path.join(R.DATA, "flickr30k", "flickr_annotations_30k.csv")

recs = []
with open(csv_path, encoding="utf-8") as f:
    for row in _csv.DictReader(f):
        try:
            caps = json.loads(row["raw"])
        except Exception:
            continue
        if len(caps) >= 2:
            recs.append((row["filename"], caps))
recs.sort()
recs = recs[:N]
print(f"样本 {len(recs)} 张图", flush=True)

lab = CrossModalLab(MODEL)

# 每张图取前 5 条 caption，逐图编码，只取成功编码的
texts, owners = [], []
for idx, (fn, caps) in enumerate(recs):
    for c in caps[:5]:
        texts.append(c)
        owners.append(idx)

te = lab.encode_text(texts, batch_size=32)
ok = ~np.isnan(te).any(axis=1)
te = te[ok]
owners = np.array([o for o, k in zip(owners, ok) if k])
print(f"caption {len(texts)} 条，有效 {len(te)} 条", flush=True)

sims = te @ te.T
n = len(te)
same = owners[:, None] == owners[None, :]
iu = np.triu_indices(n, k=1)

rel = sims[iu][same[iu]]
unrel = sims[iu][~same[iu]]

print(f"t2t_rel   n={len(rel)} median={np.median(rel):.4f} mean={rel.mean():.4f}", flush=True)
print(f"t2t_unrel n={len(unrel)} median={np.median(unrel):.4f} mean={unrel.mean():.4f}", flush=True)
print(f"t2t gap   median={np.median(rel) - np.median(unrel):.4f}", flush=True)

os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
# 无关对有几百万个，直接落盘会是几十 MB。这里只写 50 桶直方图，
# 画图只需要分布形状，不需要逐对数值。
NB = 50
counts = {"rel": [0] * NB, "unrel": [0] * NB}
for kind, arr in (("rel", rel), ("unrel", unrel)):
    for v in arr:
        b = min(int(v * NB), NB - 1)
        if b >= 0:
            counts[kind][b] += 1
with open(OUT, "w", encoding="utf-8") as f:
    for kind in ("rel", "unrel"):
        for b, c in enumerate(counts[kind]):
            f.write(f"{kind},{b} {c}\n")
print(f"写出 {OUT}", flush=True)
