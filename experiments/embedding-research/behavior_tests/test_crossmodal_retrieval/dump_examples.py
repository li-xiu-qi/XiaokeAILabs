# -*- coding: utf-8 -*-
"""余弦口径下的乱序句示例。⚠️ 本脚本不可用，勿运行。

CASES 里三个图文件在 DGX 上不存在，csv 中也没有对应行，encode_image 全返
NaN，输出全是 NaN。这些图来自旧脚本 order_hardneg 的一次运行，无从复核。

要取余弦口径的示例，改用 probe_sample_dependence.py：它从当前实际可用的
1500 张里逐样本算 gap，并按索引段与随机抽样分别汇总，不只给示例。

保留此文件仅为记录 CASES 的出处，不是可执行代码。
"""
import io, json, os, random, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mm_lab import CrossModalLab
from order_hardneg import shuffle_all, swap_two_content_words

FLICKR_CSV = "/home/ke/crossmodal_lab/datasets/flickr30k/flickr_annotations_30k.csv"
FLICKR_IMG = "/home/ke/crossmodal_lab/datasets/flickr30k/images/flickr30k-images"
OUT = "/home/ke/crossmodal_lab/results/shuffle_inversion"

rng = random.Random(20260929)

CASES = [
    ("2740249418.jpg",
     "Several men in hard hats are operating a giant pulley system.",
     "giant operating are men system. Several hard in a pulley hats"),
    ("2533764758.jpg",
     "Two young children ride on a small bicycle driven by a young woman.",
     "children Two on ride small a young bicycle woman young by driven a"),
    ("3362840713.jpg",
     "A group of people stand on a stage in front of a crowd.",
     "people group of A stand on stage in a front crowd of"),
]


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
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
                rows.append((row["filename"], caps))

    out = {"cosine_examples": []}
    for model in ["jinaai/jina-embeddings-v4", "BAAI/BGE-VL-base"]:
        lab = CrossModalLab(model)
        for fn, true_s, false_s in CASES:
            e_t = lab.encode_text([true_s])[0]
            e_f = lab.encode_text([false_s])[0]
            img = os.path.join(FLICKR_IMG, fn)
            e_i = lab.encode_image([img])[0]
            rec = {
                "model": model, "file": fn,
                "true": true_s, "false": false_s,
                "sim_true": round(cos(e_i, e_t), 4),
                "sim_false": round(cos(e_i, e_f), 4),
                "gap": round(cos(e_i, e_t) - cos(e_i, e_f), 4),
                "text_sim_tf": round(cos(e_t, e_f), 4),
            }
            out["cosine_examples"].append(rec)
            print(json.dumps(rec, ensure_ascii=False))
        lab.close()

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "cosine_examples.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
