# -*- coding: utf-8 -*-
"""跨模态检索机制实验：A 组检索基线与 B 组 ARO 硬负样本。

A 组（模态 gap 与检索基线）
  用 Flickr30k。每图取第一条人工 caption 作查询，算 R@1/5/10 与 margin。
  margin 是相关对（图对自身 caption）减无关对（图对其他图 caption）的中位数。
  绝对相似度只作参考，跨模态下各向异性会让它虚高，结论以 margin 与排序指标为准。

B 组（ARO 硬负样本，隔离「换个说法」的跨模态代价）
  VG-Relation 与 VG-Attribution 已由官方构造好 true 与 false caption，两者只在
  结构上有别（关系顺序、属性归因），词面高度重合。对每条样本算两个差：
    图侧 gap = sim(图, true) - sim(图, false)
    文本侧 gap = sim(true, false) - 底噪
  两者对照回答：模型在纯文本上能分清新旧说法的程度，与在图侧能分的程度差多少。
  这条连接文本内实验的结论（换个说法掉 0.1 到 0.5），预期跨模态掉得更多。

用法（DGX Spark）：
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ~/xinfer-env/bin/python run_experiments.py --group a --limit 2000
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ~/xinfer-env/bin/python run_experiments.py --group b --limit 3000
  ... --group ab          两组都跑
  ... --model BAAI/BGE-VL-base   换模型跑对照，结果落 results/<模型名>/，不覆盖默认模型

产物落在 results/<模型名>/：
  a_retrieval.csv      每条查询的 rank 与相关/无关 margin
  a_summary.json       R@k 与 margin 汇总
  b_aro.csv            每条 ARO 样本的图侧与文本侧 gap
  b_summary.json       按关系、属性两类分别汇总
"""

import argparse
import csv
import json
import os
import random
import re
import sys
import time
import zipfile
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mm_lab import CrossModalLab  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "datasets")
RESULTS = os.path.join(ROOT, "results")


def results_dir_for(model_id: str) -> str:
    """按模型分结果目录。

    2026-09-26 加模型间对照时补：不分子目录的话第二个模型会把第一个的
    a_retrieval.csv 与 b_aro.csv 直接覆盖，且覆盖前没有任何提示。
    """
    return os.path.join(ROOT, "results", model_id.split("/")[-1])


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ensure_flickr_images():
    """解压 Flickr30k 图像 zip，返回图像目录。已解压则跳过。"""
    zpath = os.path.join(DATA, "flickr30k", "flickr30k-images.zip")
    outdir = os.path.join(DATA, "flickr30k", "images")
    if os.path.isdir(outdir) and len(os.listdir(outdir)) > 100:
        return outdir
    log(f"解压 {os.path.basename(zpath)} ...")
    os.makedirs(outdir, exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(outdir)
    # zip 内可能还有一层 flickr30k-images/
    inner = os.path.join(outdir, "flickr30k-images")
    if os.path.isdir(inner):
        return inner
    return outdir


def load_flickr(limit=None, seed=0, split=None):
    """读标注 CSV，返回 [(image_name, caption)]，每图取第一条 caption。

    CSV 列为 raw, sentids, split, filename, img_id。raw 是 5 条 caption 的
    JSON 数组字符串，需二次解析。split 为 Karpathy 划分，默认不限。
    """
    csv_path = os.path.join(DATA, "flickr30k", "flickr_annotations_30k.csv")
    items = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if split and row.get("split") != split:
                continue
            try:
                caps = json.loads(row["raw"])
            except Exception:
                continue
            if caps:
                items.append((row["filename"], caps[0].strip()))
    items.sort()
    if limit and len(items) > limit:
        rng = random.Random(seed)
        items = rng.sample(items, limit)
    return items


def load_aro(kind, limit=None, seed=0):
    """读 ARO 标注。kind 为 relation 或 attribution。"""
    fname = {"relation": "vgr_annotations.json",
             "attribution": "vga_annotations.json"}[kind]
    path = os.path.join(DATA, "aro", fname)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if limit and len(data) > limit:
        rng = random.Random(seed)
        data = rng.sample(data, limit)
    return data


def find_aro_images():
    """定位 VG 图像目录。zip 解压后目录名不定，按含 jpg 数量找。"""
    base = os.path.join(DATA, "aro")
    zpath = os.path.join(base, "vgr_vga_images.zip")
    if not os.path.isdir(os.path.join(base, "images")):
        if os.path.exists(zpath):
            log("解压 ARO VG 图像 ...")
            with zipfile.ZipFile(zpath) as z:
                z.extractall(base)
        else:
            raise FileNotFoundError(f"缺少 {zpath}")
    for dirpath, _, files in os.walk(base):
        jpgs = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if len(jpgs) > 100:
            return dirpath
    raise FileNotFoundError("未找到含图像的目录")


def aro_image_path(img_root, image_id):
    """VG 图像文件名规则。先试 image_id.jpg，再试通配。"""
    for ext in (".jpg", ".jpeg", ".png"):
        p = os.path.join(img_root, f"{image_id}{ext}")
        if os.path.exists(p):
            return p
    return None


# ------------------------------------------------------------------ A 组

def run_group_a(lab, limit):
    log("A 组：Flickr30k 检索基线")
    img_root = ensure_flickr_images()
    items = load_flickr(limit=limit)
    log(f"样本 {len(items)} 对")

    paths = [os.path.join(img_root, name) for name, _ in items]
    caps = [c for _, c in items]

    log("编码图像 ...")
    ie = lab.encode_image(paths, batch_size=8)
    ok_img = ~np.isnan(ie).any(axis=1)
    log(f"图像可用 {int(ok_img.sum())}/{len(paths)}")

    log("编码查询文本 ...")
    te = lab.encode_text(caps, batch_size=32)

    keep = ok_img
    ie, te, caps, names = ie[keep], te[keep], [c for c, k in zip(caps, keep) if k], \
        [n for n, k in zip([x[0] for x in items], keep) if k]

    # 相似度矩阵。按块算，避免一次性占太多内存。
    log("算相似度矩阵 ...")
    n = len(ie)
    sims = np.zeros((n, n), dtype=np.float32)
    block = 512
    for s in range(0, n, block):
        e = min(s + block, n)
        a = ie[s:e]
        na = np.linalg.norm(a, axis=1, keepdims=True)
        b = te.T
        nb = np.linalg.norm(te, axis=1, keepdims=True).T
        sims[s:e] = (a @ b) / np.clip(na * nb, 1e-12, None)

    # 每行是一张图，对该图第一条 caption 的检索排名
    ranks = []
    rel_sims, unrel_sims = [], []
    for i in range(n):
        order = np.argsort(-sims[i])
        rank = int(np.where(order == i)[0][0]) + 1
        ranks.append(rank)
        rel_sims.append(float(sims[i, i]))
        row = np.delete(sims[i], i)
        unrel_sims.append(float(np.median(row)))

    ranks = np.array(ranks)
    out = {
        "n": int(n),
        "R@1": float((ranks <= 1).mean()),
        "R@5": float((ranks <= 5).mean()),
        "R@10": float((ranks <= 10).mean()),
        "median_rank": float(np.median(ranks)),
        "rel_sim_median": float(np.median(rel_sims)),
        "unrel_sim_median": float(np.median(unrel_sims)),
        "margin_median": float(np.median(np.array(rel_sims) - np.array(unrel_sims))),
        "img_img_unrel_median": None,  # 图对图的无关基线，B 组之后补
    }

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "a_retrieval.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "caption", "rank", "rel_sim", "unrel_sim_median"])
        for name, cap, rk, rs, us in zip(names, caps, ranks, rel_sims, unrel_sims):
            w.writerow([name, cap[:80], rk, f"{rs:.6f}", f"{us:.6f}"])
    with open(os.path.join(RESULTS, "a_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log("A 组完成：" + json.dumps(out, ensure_ascii=False))
    return out


# ------------------------------------------------------------------ B 组

def run_group_b(lab, limit):
    log("B 组：ARO 硬负样本")
    img_root = find_aro_images()
    rows = []
    for kind in ("relation", "attribution"):
        data = load_aro(kind, limit=limit)
        log(f"{kind}: {len(data)} 条")

        # 去重图像，只编码一次
        uniq = sorted({str(d["image_id"]) for d in data})
        paths, valid = [], {}
        for iid in uniq:
            p = aro_image_path(img_root, iid)
            if p:
                valid[iid] = p
        log(f"  去重图像 {len(uniq)}，找到 {len(valid)}")
        if not valid:
            log(f"  {kind} 无可用图像，跳过。请检查 {img_root} 的文件名规则")
            continue

        keys = list(valid)
        ie = lab.encode_image([valid[k] for k in keys], batch_size=8)
        ok = ~np.isnan(ie).any(axis=1)
        emb = {k: ie[i] for i, k in enumerate(keys) if ok[i]}

        true_caps = [d["true_caption"] for d in data]
        false_caps = [d["false_caption"] for d in data]
        te_t = lab.encode_text(true_caps, batch_size=32)
        te_f = lab.encode_text(false_caps, batch_size=32)

        for i, d in enumerate(data):
            iid = str(d["image_id"])
            if iid not in emb:
                continue
            v = emb[iid][None, :]
            s_true = float(lab.cosine(v, te_t[i][None, :])[0])
            s_false = float(lab.cosine(v, te_f[i][None, :])[0])
            t_gap = float(lab.cosine(te_t[i][None, :], te_f[i][None, :])[0])
            rows.append({"kind": kind, "image_id": iid,
                         "true_caption": d["true_caption"][:90],
                         "false_caption": d["false_caption"][:90],
                         "sim_true": s_true, "sim_false": s_false,
                         "img_gap": s_true - s_false,
                         "text_sim_tf": t_gap})

    if not rows:
        log("B 组无数据")
        return {}

    img_gap = np.array([r["img_gap"] for r in rows])
    txt_sim = np.array([r["text_sim_tf"] for r in rows])
    out = {
        "n": len(rows),
        "img_gap_median": float(np.median(img_gap)),
        "img_gap_positive_rate": float((img_gap > 0).mean()),
        "text_sim_tf_median": float(np.median(txt_sim)),
    }
    by_kind = {}
    for kind in sorted({r["kind"] for r in rows}):
        g = np.array([r["img_gap"] for r in rows if r["kind"] == kind])
        t = np.array([r["text_sim_tf"] for r in rows if r["kind"] == kind])
        by_kind[kind] = {
            "n": int(len(g)),
            "img_gap_median": float(np.median(g)),
            "img_gap_positive_rate": float((g > 0).mean()),
            "text_sim_tf_median": float(np.median(t)),
        }
    out["by_kind"] = by_kind

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "b_aro.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(os.path.join(RESULTS, "b_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log("B 组完成：" + json.dumps(out, ensure_ascii=False))
    return out


def main():
    global RESULTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="ab", choices=["a", "b", "ab"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default="jinaai/jina-embeddings-v4",
                    help="跨模态 embedding 模型 id")
    a = ap.parse_args()

    RESULTS = results_dir_for(a.model)
    log(f"模型 {a.model}")
    log(f"结果目录 {RESULTS}")

    lab = CrossModalLab(a.model)
    try:
        if a.group in ("a", "ab"):
            run_group_a(lab, a.limit)
        if a.group in ("b", "ab"):
            run_group_b(lab, a.limit)
    finally:
        lab.close()


if __name__ == "__main__":
    main()
