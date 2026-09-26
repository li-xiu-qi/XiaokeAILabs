# -*- coding: utf-8 -*-
"""诊断：跨模态 R@1 低是模型特性还是编码 bug。

三个探针，都用 Flickr30k 前 N 张图：
  T2T  同一张图的 5 条 caption 互相相似度（同模态内聚性，应高）
  I2I  同一张图与自身（应 1.0）与不同图（同模态无关基线）
  T2I  图对它的一条 caption vs 对其他 caption（跨模态）

若 T2T 高而 T2I 低，说明跨模态对齐弱，是模型特性。
若 T2T 也低，说明文本编码有问题，实验读数全部作废。
"""
import json, os, sys, zipfile
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mm_lab import CrossModalLab
import run_experiments as R

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200

img_root = R.ensure_flickr_images()
csv_path = os.path.join(R.DATA, "flickr30k", "flickr_annotations_30k.csv")
import csv as _csv
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

lab = CrossModalLab()
try:
    ie = lab.encode_image([os.path.join(img_root, n) for n, _ in recs], batch_size=8)
    ok = ~np.isnan(ie).any(axis=1)
    recs = [r for r, k in zip(recs, ok) if k]
    ie = ie[ok]
    all_caps = [c for _, caps in recs for c in caps]
    te = lab.encode_text(all_caps, batch_size=32)
    caps_per = [len(caps) for _, caps in recs]

    # T2T：同图 caption 之间 vs 跨图 caption 之间
    same, diff = [], []
    starts = np.cumsum([0] + caps_per)
    for i in range(len(recs)):
        s, e = starts[i], starts[i + 1]
        for a in range(s, e):
            for b in range(a + 1, e):
                same.append(float(lab.cosine(te[a:a+1], te[b:b+1])[0]))
    rng = np.random.default_rng(0)
    for _ in range(len(same)):
        a, b = rng.integers(0, len(te), 2)
        if starts[np.searchsorted(starts, a, 'right') - 1] != starts[np.searchsorted(starts, b, 'right') - 1]:
            diff.append(float(lab.cosine(te[a:a+1], te[b:b+1])[0]))
    # I2I：同图（自比）与跨图
    i_same = [float(lab.cosine(ie[i:i+1], ie[i:i+1])[0]) for i in range(len(recs))]
    i_diff = []
    for _ in range(len(recs)):
        a, b = rng.integers(0, len(ie), 2)
        if a != b:
            i_diff.append(float(lab.cosine(ie[a:a+1], ie[b:b+1])[0]))
    # T2I：图对同图首条 caption vs 对其他图的首条 caption
    t2i_same = [float(lab.cosine(ie[i:i+1], te[starts[i]:starts[i]+1])[0]) for i in range(len(recs))]
    t2i_diff = []
    for _ in range(len(recs)):
        a, b = rng.integers(0, len(recs), 2)
        if a != b:
            t2i_diff.append(float(lab.cosine(ie[a:a+1], te[starts[b]:starts[b]+1])[0]))

    out = {
        "n_images": len(recs),
        "T2T_same_image_median": float(np.median(same)),
        "T2T_diff_image_median": float(np.median(diff)),
        "T2T_gap": float(np.median(same) - np.median(diff)),
        "I2I_self_median": float(np.median(i_same)),
        "I2I_diff_median": float(np.median(i_diff)),
        "T2I_same_median": float(np.median(t2i_same)),
        "T2I_diff_median": float(np.median(t2i_diff)),
        "T2I_gap": float(np.median(t2i_same) - np.median(t2i_diff)),
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
finally:
    lab.close()
