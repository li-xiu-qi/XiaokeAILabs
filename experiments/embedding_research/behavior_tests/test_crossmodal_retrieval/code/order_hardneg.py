# -*- coding: utf-8 -*-
"""D 组：Flickr30k-Order，闭合 ARO 三类硬负样本的第三类。

relation 与 attribution 已由官方构造好 true/false，Order 这一类要自己生成
词序扰动。ARO 论文的做法是交换名词、动词短语等语言学元素，需要词性标注，
而目标机无外网装不了相关库。这里用两种不依赖词性的扰动做对照：

  swap     交换两个实词的位置。近似论文的 noun/verb phrase swapping，
           句子保持语法正确，是更严格的测试。
  shuffle  把全部词随机打乱。对应论文的 shuffle-all 策略，句子语法崩坏，
           用来检验乱码是否被模型轻易排除。

若 shuffle 的 gap 明显大于 swap，说明模型对语法崩坏敏感，那么
relation 与 attribution 上测到的无能就不能简单归因为「读词袋」，
它至少还能靠语法异常排除一类。

读数与 B 组对齐：图侧 gap = sim(图, true) - sim(图, false)，
另记 sim(true, false) 用于确认负样本确实词面高度重合。

用法：order_hardneg.py [图数] [模型id]
"""

import csv as _csv
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mm_lab import CrossModalLab
import run_experiments as R

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
MODEL = sys.argv[2] if len(sys.argv) > 2 else "jinaai/jina-embeddings-v4"
RES = R.results_dir_for(MODEL)
os.makedirs(RES, exist_ok=True)

rng = random.Random(20260927)


def swap_two_content_words(cap: str):
    """交换两个实词。实词用长度近似（>=4 字母），不依赖词性标注。"""
    toks = cap.split()
    idx = [i for i, t in enumerate(toks)
           if len(t.strip(".,!?;:'\"")) >= 4]
    if len(idx) < 2:
        return None
    for _ in range(20):
        a, b = rng.sample(idx, 2)
        ta, tb = toks[a].lower().strip(".,!?;:'\""), toks[b].lower().strip(".,!?;:'\"")
        if ta == tb:
            continue
        out = list(toks)
        out[a], out[b] = out[b], out[a]
        return " ".join(out)
    return None


def shuffle_all(cap: str):
    toks = cap.split()
    if len(toks) < 4:
        return None
    out = list(toks)
    rng.shuffle(out)
    if out == toks:
        return None
    return " ".join(out)


def build(n):
    csv_path = os.path.join(R.DATA, "flickr30k", "flickr_annotations_30k.csv")
    recs = []
    with open(csv_path, encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            try:
                caps = json.loads(row["raw"])
            except Exception:
                continue
            if caps:
                recs.append((row["filename"], caps[0]))
    recs.sort()
    return recs[:n]


def main():
    img_root = R.ensure_flickr_images()
    recs = build(N)
    R.log(f"样本 {len(recs)} 张图，模型 {MODEL}")

    lab = CrossModalLab(MODEL)
    try:
        rows = []
        for kind, fn in (("swap", swap_two_content_words),
                         ("shuffle", shuffle_all)):
            items = []
            for name, cap in recs:
                neg = fn(cap)
                if neg and neg != cap:
                    items.append((name, cap, neg))
            R.log(f"{kind}: {len(items)} 条")

            names = [i[0] for i in items]
            trues = [i[1] for i in items]
            falses = [i[2] for i in items]

            ie = lab.encode_image([os.path.join(img_root, n) for n in names],
                                  batch_size=8)
            ok = ~np.isnan(ie).any(axis=1)
            ie = ie[ok]
            te_t = lab.encode_text([t for t, k in zip(trues, ok) if k], batch_size=32)
            te_f = lab.encode_text([t for t, k in zip(falses, ok) if k], batch_size=32)
            names = [n for n, k in zip(names, ok) if k]

            # 逐行算，避免整矩阵占内存
            for i in range(len(names)):
                s_t = float(np.dot(te_t[i], ie[i]))
                s_f = float(np.dot(te_f[i], ie[i]))
                tf = float(np.dot(te_t[i], te_f[i]))
                rows.append({
                    "kind": kind, "image": names[i],
                    "true": trues[i][:120], "false": falses[i][:120],
                    "sim_true": s_t, "sim_false": s_f,
                    "img_gap": s_t - s_f, "text_sim_tf": tf,
                })
            del ie, te_t, te_f

        by = {}
        for kind in ("swap", "shuffle"):
            g = np.array([r["img_gap"] for r in rows if r["kind"] == kind])
            t = np.array([r["text_sim_tf"] for r in rows if r["kind"] == kind])
            by[kind] = {
                "n": int(len(g)),
                "img_gap_median": float(np.median(g)) if len(g) else None,
                "img_gap_positive_rate": float((g > 0).mean()) if len(g) else None,
                "text_sim_tf_median": float(np.median(t)) if len(t) else None,
            }
        out = {"model": MODEL, "by_kind": by}

        with open(os.path.join(RES, "d_order.csv"), "w", newline="",
                  encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        with open(os.path.join(RES, "d_summary.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        R.log("D 组完成：" + json.dumps(out, ensure_ascii=False))
    finally:
        lab.close()


if __name__ == "__main__":
    main()
