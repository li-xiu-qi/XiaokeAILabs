# -*- coding: utf-8 -*-
"""C 组：描述中介。同一条查询，走图的原生向量与走图的文字描述，两条路差多少。

动机来自素材检索的实际取舍：命中一帧画面时，可以用帧的图像向量做候选，
也可以用这帧的文字描述向量做候选。两条路的分数不可比（跨模态与同模态的
绝对分布不同），要比的是排序质量与失败模式。

对每张图取第一条人工 caption 作查询，候选池固定为同一批图，算两条路：

  图向量路由    查询文本 对 500 张图的图像向量
  描述向量路由  查询文本 对 500 条描述文本的文本向量

查询取每图第二条人工 caption，描述候选取每图第一条。两条路因此都有明确的
正确候选，且查询不在候选里。若用同一条 caption 既作查询又作候选，排除自身
后就不存在正确候选，检索任务不成立（第一版就栽在这）。

读数四类：
  R@1 / R@5 / R@10          两条路各自的排序质量
  margin                    相关减无关的中位数
  互补率                    一条进前五另一条没进的比例，回答「能不能互相救」

口径与 A 组对齐：同一批图、同一条查询、同一个候选池大小，这样两条路之间
以及与 A 组的 T2I 读数都可比。

用法：route_compare.py [图数] [模型id]
"""

import csv as _csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mm_lab import CrossModalLab
import run_experiments as R

N = int(sys.argv[1]) if len(sys.argv) > 1 else 500
MODEL = sys.argv[2] if len(sys.argv) > 2 else "jinaai/jina-embeddings-v4"
RES = R.results_dir_for(MODEL)
os.makedirs(RES, exist_ok=True)

K = 10


def load_pairs(n):
    """读 Flickr30k 标注，返回 (图名, 查询, 描述候选)。

    查询用第二条 caption，描述候选用第一条。两者是同一张图的两种人工描述，
    语义等价而措辞不同，正是「换个说法」的真实形态。
    """
    csv_path = os.path.join(R.DATA, "flickr30k", "flickr_annotations_30k.csv")
    recs = []
    with open(csv_path, encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            try:
                caps = json.loads(row["raw"])
            except Exception:
                continue
            if len(caps) >= 2:
                recs.append((row["filename"], caps[1], caps[0]))
    recs.sort()
    return recs[:n]


def main():
    img_root = R.ensure_flickr_images()
    recs = load_pairs(N)
    names = [r[0] for r in recs]
    queries = [r[1] for r in recs]
    descs = [r[2] for r in recs]
    R.log(f"样本 {len(recs)} 张图，模型 {MODEL}")

    lab = CrossModalLab(MODEL)
    try:
        ie = lab.encode_image([os.path.join(img_root, n) for n in names],
                              batch_size=8)
        ok = ~np.isnan(ie).any(axis=1)
        names = [n for n, k in zip(names, ok) if k]
        queries = [q for q, k in zip(queries, ok) if k]
        descs = [d for d, k in zip(descs, ok) if k]
        ie = ie[ok]
        qe = lab.encode_text(queries, batch_size=32)
        de = lab.encode_text(descs, batch_size=32)
        n = len(descs)

        s_ti = qe @ ie.T      # 查询文本 对 图
        s_tt = qe @ de.T      # 查询文本 对 描述候选

        # 查询与候选都不是同一条文本，无需排除自身
        rows = []
        for i in range(n):
            r_ti = int(np.where(np.argsort(-s_ti[i]) == i)[0][0]) + 1
            r_tt = int(np.where(np.argsort(-s_tt[i]) == i)[0][0]) + 1
            rows.append({
                "image": names[i],
                "rank_image_route": r_ti,
                "rank_text_route": r_tt,
                "rel_ti": float(s_ti[i, i]),
                "unrel_ti": float(np.median(np.delete(s_ti[i], i))),
                "rel_tt": float(s_tt[i, i]),
                "unrel_tt": float(np.median(np.delete(s_tt[i], i))),
            })

        def summarize(rk, rel, unrel):
            ranks = np.array([r[rk] for r in rows], dtype=float)
            rv = np.array([r[rel] for r in rows], dtype=float)
            uv = np.array([r[unrel] for r in rows], dtype=float)
            return {
                "R@1": float((ranks <= 1).mean()),
                "R@5": float((ranks <= 5).mean()),
                "R@10": float((ranks <= K).mean()),
                "median_rank": float(np.median(ranks)),
                "rel_median": float(np.median(rv)),
                "unrel_median": float(np.median(uv)),
                "margin_median": float(np.median(rv) - np.median(uv)),
            }

        out = {
            "model": MODEL,
            "n": n,
            "image_route": summarize("rank_image_route", "rel_ti", "unrel_ti"),
            "text_route": summarize("rank_text_route", "rel_tt", "unrel_tt"),
        }

        ri = np.array([r["rank_image_route"] for r in rows])
        rt = np.array([r["rank_text_route"] for r in rows])
        out["complement"] = {
            "both_top1": int(((ri == 1) & (rt == 1)).sum()),
            "image_only_top5": int(((ri <= 5) & (rt > 5)).sum()),
            "text_only_top5": int(((rt <= 5) & (ri > 5)).sum()),
            "neither_top5": int(((ri > 5) & (rt > 5)).sum()),
        }

        with open(os.path.join(RES, "route_compare.csv"), "w", newline="",
                  encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        with open(os.path.join(RES, "route_summary.json"), "w",
                  encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        R.log("C 组完成：" + json.dumps(out, ensure_ascii=False))
    finally:
        lab.close()


if __name__ == "__main__":
    main()
