# -*- coding: utf-8 -*-
"""D 组整表余弦口径重算。

order_hardneg.py 用 np.dot 算相似度，是未归一化的 raw embedding（|a||b|cos），
模长效应混在读数里。重算范围不止 order-shuffle 一行：relation 与 attribution
的正比例和中位数同样受影响（raw dot 的符号取决于两个向量的模长比，不只方向）。

本脚本对 ARO 三类硬负样本 + Flickr order 两类扰动，统一用余弦重算，
每个读数同时输出 raw dot 值以便对照差异来源。

用法：
  ~/xinfer-env/bin/python diag_hardneg_cosine.py --model jinaai/jina-embeddings-v4
  ~/xinfer-env/bin/python diag_hardneg_cosine.py --model BAAI/BGE-VL-base
"""
import argparse, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_experiments as R
from mm_lab import CrossModalLab
from order_hardneg import shuffle_all, swap_two_content_words
from diag_shuffle_inversion import drop_two_content_words

OUT = "/home/ke/crossmodal_lab/results/hardneg_cosine"


def cos(A, B):
    """A, B 为 1-D 向量，返回余弦。"""
    return float(np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B)))


def batch_cos(EI, ET):
    """EI (n,d) 对 ET (n,d) 逐行余弦。"""
    a = EI / np.linalg.norm(EI, axis=1, keepdims=True)
    b = ET / np.linalg.norm(ET, axis=1, keepdims=True)
    return np.sum(a * b, axis=1)


def load_flickr(n):
    rows = []
    import csv
    with open(R.os.path.join(R.DATA, "flickr30k", "flickr_annotations_30k.csv"),
              encoding="utf-8") as f:
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


def encode_pairs(lab, image_root, image_refs, cap_t, cap_f, batch=16):
    """按图像引用编码图像，逐对编码两条 caption。

    image_refs 已是完整路径。VG 的 image_id 不带扩展名，Flickr 的 filename 带
    .jpg，两种由调用方各自拼好路径，这里不猜。
    返回 (EI, ET, EF) 与有效掩码。"""
    EI = lab.encode_image(list(image_refs), batch_size=batch)
    ok = ~np.isnan(EI).any(axis=1)
    keep = [j for j in range(len(image_refs)) if ok[j]]
    print(f"  图像有效 {len(keep)}/{len(image_refs)}")
    if not keep:
        print("  !! 全部图像无效，检查路径拼接")
        return None
    EI = EI[keep]
    ct = [cap_t[j] for j in keep]
    cf = [cap_f[j] for j in keep]
    ET = lab.encode_text(ct, batch_size=32)
    EF = lab.encode_text(cf, batch_size=32)
    # 文本侧也要滤 NaN
    tok = ~np.isnan(ET).any(axis=1) & ~np.isnan(EF).any(axis=1)
    print(f"  文本有效 {int(tok.sum())}/{len(keep)}")
    if int(tok.sum()) < 50:
        return None
    return EI[tok], ET[tok], EF[tok]


def summarize(name, gaps, gaps_raw, text_sim):
    g = np.array([x for x in gaps if x is not None])
    gr = np.array([x for x in gaps_raw if x is not None])
    ts = np.array([x for x in text_sim if x is not None])
    return {
        "kind": name,
        "n": int(len(g)),
        "gap_median_cos": round(float(np.median(g)), 5),
        "gap_positive_rate_cos": round(float((g > 0).mean()), 4),
        "gap_median_rawdot": round(float(np.median(gr)), 5),
        "gap_positive_rate_rawdot": round(float((gr > 0).mean()), 4),
        "norm_true_median": None,
        "text_sim_median": round(float(np.median(ts)), 4) if len(ts) else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="jinaai/jina-embeddings-v4")
    ap.add_argument("--n", type=int, default=1000)
    args = ap.parse_args()

    R.log(f"模型 {args.model}")
    lab = CrossModalLab(args.model)
    img_root = R.find_aro_images()
    results = []

    # ---- ARO relation / attribution ----
    for kind in ("relation", "attribution"):
        data = R.load_aro(kind, limit=args.n)
        ct = [d["true_caption"] for d in data]
        cf = [d["false_caption"] for d in data]
        paths = []
        for d in data:
            p = R.aro_image_path(img_root, d["image_id"])
            paths.append(p if p else "/nonexistent.jpg")
        enc = encode_pairs(lab, img_root, paths, ct, cf)
        if enc is None:
            continue
        EI, ET, EF = enc
        gaps = [batch_cos(EI[i:i + 1], ET[i:i + 1])[0]
                - batch_cos(EI[i:i + 1], EF[i:i + 1])[0] for i in range(len(EI))]
        gaps_raw = [float(np.dot(EI[i], ET[i])) - float(np.dot(EI[i], EF[i]))
                    for i in range(len(EI))]
        ts = [batch_cos(ET[i:i + 1], EF[i:i + 1])[0] for i in range(len(EI))]
        rec = summarize(kind, gaps, gaps_raw, ts)
        results.append(rec)
        print(json.dumps(rec, ensure_ascii=False))

    # ---- Flickr order-swap / order-shuffle / order-drop ----
    rows = load_flickr(args.n)
    flickr_root = os.path.join(R.DATA, "flickr30k", "images", "flickr30k-images")
    for name, fn in (("order-swap", swap_two_content_words),
                     ("order-shuffle", shuffle_all),
                     ("order-drop", drop_two_content_words)):
        caps_t, caps_f, ids = [], [], []
        for nm, cap in rows:
            v = fn(cap)
            if v and v != cap:
                ids.append(nm)
                caps_t.append(cap)
                caps_f.append(v)
        enc = encode_pairs(lab, flickr_root,
                          [os.path.join(flickr_root, nm) for nm in ids],
                          caps_t, caps_f)
        if enc is None:
            continue
        EI, ET, EF = enc
        gaps = [batch_cos(EI[i:i + 1], ET[i:i + 1])[0]
                - batch_cos(EI[i:i + 1], EF[i:i + 1])[0] for i in range(len(EI))]
        gaps_raw = [float(np.dot(EI[i], ET[i])) - float(np.dot(EI[i], EF[i]))
                    for i in range(len(EI))]
        ts = [batch_cos(ET[i:i + 1], EF[i:i + 1])[0] for i in range(len(EI))]
        rec = summarize(name, gaps, gaps_raw, ts)
        results.append(rec)
        print(json.dumps(rec, ensure_ascii=False))

    lab.close()
    os.makedirs(OUT, exist_ok=True)
    outp = os.path.join(OUT, f"hardneg_cosine_{args.model.split('/')[-1]}.json")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "n": args.n, "results": results},
                  f, indent=2, ensure_ascii=False)
    R.log(f"落盘 -> {outp}")


if __name__ == "__main__":
    main()
