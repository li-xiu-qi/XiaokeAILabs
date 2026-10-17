# -*- coding: utf-8 -*-
"""乱序句反向偏好：shuffle 后的句子是否对任何图都能拿更高分。

D 组发现 jina-v4 给乱序 caption 的相似度中位数比真 caption 高 0.0151（gap 为负，
60% 样本成立）。本实验不追问「乱序时检索差不差」，只追问这个反向偏好本身：
  1. 能否在更大样本、更多扰动类型上复现
  2. 它是否只对自己的图成立，还是对任意图都成立（检验均值回落假设）
  3. 若是全局现象，能否用「乱序句 embedding 更靠近语料均值」解释

三个判据，全部可用现有 embedding 一次算完，不需要重编码模型以外的部分：
  own_gap      对自己图：sim(图, 真) - sim(图, 乱序)
  other_gap    对随机其他图：同一差值取平均。若 own_gap 为负而 other_gap 更负，
               说明乱序句的优势只对自己那张图，反向偏好是特异的，不是全局均值效应。
               若两者都为负且接近，说明乱序句对所有图都更吃香，支持均值回落。
  mean_cos     乱序句与真句子各自到本批文本均值的余弦。乱序句更高 => 支持假设。

扰动除 shuffle 外另加两类对照，用来排除「词被打乱」以外的解释：
  drop         随机删掉两个实词（仍是打乱的子集，长度变短）
  swap         交换两个实词（语法仍正确，D 组已见弱反向）
用法：~/xinfer-env/bin/python diag_shuffle_inversion.py --model jinaai/jina-embeddings-v4 --n 1500
"""
import argparse, io, json, os, random, sys, time, warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/ke/crossmodal_lab/scripts")
import run_experiments as R
from order_hardneg import shuffle_all, swap_two_content_words
from run_experiments import log   # log 定义在 run_experiments，之前漏导入导致直接 NameError

FLICKR_CSV = "/home/ke/crossmodal_lab/datasets/flickr30k/flickr_annotations_30k.csv"
FLICKR_IMG = "/home/ke/crossmodal_lab/datasets/flickr30k/images/flickr30k-images"
OUT = "/home/ke/crossmodal_lab/results/shuffle_inversion"

rng = random.Random(20260929)


def drop_two_content_words(cap: str):
    """随机删两个实词（长度近似 >=4）。用来分离「乱序」与「变短」。"""
    toks = cap.split()
    idx = [i for i, t in enumerate(toks) if len(t.strip(".,!?;:'\"")) >= 4]
    if len(idx) < 3:
        return None
    a, b = rng.sample(idx, 2)
    return " ".join(t for i, t in enumerate(toks) if i not in (a, b))


def load(n):
    rows = []
    with open(FLICKR_CSV, encoding="utf-8") as f:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="jinaai/jina-embeddings-v4")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--n-other", type=int, default=50, help="每句抽多少张其他图做对照")
    args = ap.parse_args()

    from mm_lab import CrossModalLab

    rows = load(args.n)
    log(f"样本 {len(rows)} 张图")
    names = [r[0] for r in rows]
    trues = [r[1] for r in rows]

    kinds = {"shuffle": shuffle_all, "swap": swap_two_content_words,
             "drop": drop_two_content_words}
    variants = {}
    for kind, fn in kinds.items():
        out, keep = [], []
        for i, (nm, cap) in enumerate(rows):
            v = fn(cap)
            if v and v != cap:
                out.append(v)
                keep.append(i)
        variants[kind] = (out, np.array(keep))
        log(f"  {kind}: 成功构造 {len(out)} 条")

    # 缓存：1500 张图编码要 300 秒，换模型不能重来
    os.makedirs(OUT, exist_ok=True)
    tag = args.model.split("/")[-1].replace(".", "_")
    p_img = os.path.join(OUT, f"cache_{tag}_img.npy")
    p_txt = os.path.join(OUT, f"cache_{tag}_true.npy")
    if os.path.exists(p_img) and os.path.exists(p_txt):
        E_img, E_true = np.load(p_img), np.load(p_txt)
        ok = np.ones(len(names), dtype=bool)
        log(f"命中缓存 图 {E_img.shape} 文本 {E_true.shape}")
    else:
        lab = CrossModalLab(args.model)
        t0 = time.time()
        E_img = lab.encode_image([os.path.join(FLICKR_IMG, n) for n in names], batch_size=8)
        ok = ~np.isnan(E_img).any(axis=1)
        E_img = E_img[ok]
        E_true = lab.encode_text([t for t, k in zip(trues, ok) if k], batch_size=32)
        lab.close()
        np.save(p_img, E_img)
        np.save(p_txt, E_true)
        log(f"图向量 {E_img.shape} 用 {time.time()-t0:.0f}s，已缓存")
    valid = np.where(ok)[0]

    # 只保留扰动也成功的交集
    common = None
    for kind, (_, keep) in variants.items():
        s = set(keep.tolist())
        common = s if common is None else (common & s)
    idx = sorted(common & set(valid.tolist()))
    log(f"三种扰动都成功的共同样本 {len(idx)}")
    if not idx:
        raise SystemExit("无共同样本")

    res = {"model": args.model, "n": len(idx), "per_kind": {}}
    text_mean = E_true[idx].mean(axis=0)   # 扰动前算，idx 后续会被逐 kind 裁

    for kind, (texts, keep) in variants.items():
        pos = {g: j for j, g in enumerate(keep.tolist())}
        sel = [pos[i] for i in idx]
        # 该 kind 的文本向量单独编码
        lab2 = CrossModalLab(args.model)
        E_v = lab2.encode_text([texts[j] for j in sel], batch_size=32)
        lab2.close()
        # 文本侧同样要滤 NaN。shuffle 与 drop 会产出让编码器返回 NaN 的句子
        # （极短或触发 remote code 边界），不过滤整组读数都是 NaN。
        vok = ~np.isnan(E_v).any(axis=1)
        if not vok.all():
            log(f"  {kind}: 文本侧剔除 {int((~vok).sum())} 条 NaN")
        if int(vok.sum()) < 50:
            log(f"  {kind}: 有效样本不足 50，跳过")
            continue
        E_v = E_v[vok]
        sel = [j for j, k in zip(sel, vok) if k]
        idx = [i for i, k in zip(idx, vok) if k]   # 同步裁，否则对角线与 E_true 错位
        E_t = E_true[:0].reshape(0, E_true.shape[1])
        E_t = np.array([E_true[i] for i in idx])

        S_img_var = cosmat(E_img, E_v)     # (n_img, n_var)
        S_img_true = cosmat(E_img, E_t)  # (n_img, n_true)

        # 对角线：自己的图对自己的句子
        own = np.array([S_img_true[i, i] - S_img_var[i, i] for i in range(len(idx))])

        # 其他图对照：每句抽 n_other 张非自己的图
        rng2 = np.random.default_rng(0)
        others = []
        for i in range(len(idx)):
            pool = [j for j in range(len(idx)) if j != i]
            pick = rng2.choice(pool, min(args.n_other, len(pool)), replace=False)
            # 真句对随机图 减 乱序句对同一批随机图
            t = np.mean([S_img_true[p, i] for p in pick])
            v = np.mean([S_img_var[p, i] for p in pick])
            others.append(t - v)
        others = np.array(others)

        # 与语料均值的距离
        def mc(E):
            En = E / np.linalg.norm(E, axis=1, keepdims=True)
            mn = text_mean / np.linalg.norm(text_mean)
            return En @ mn

        res["per_kind"][kind] = {
            "own_gap_median": round(float(np.median(own)), 5),
            "own_gap_positive_rate": round(float((own > 0).mean()), 4),
            "other_gap_median": round(float(np.median(others)), 5),
            "other_gap_positive_rate": round(float((others > 0).mean()), 4),
            "mean_cos_var": round(float(np.mean(mc(E_v))), 5),
            "mean_cos_true": round(float(np.mean(mc(E_t))), 5),
            "norm_true_median": round(float(np.median(np.linalg.norm(E_t, axis=1))), 4),
            "norm_var_median": round(float(np.median(np.linalg.norm(E_v, axis=1))), 4),
            "text_sim_median": round(float(np.median(np.array(
                [cosmat(E_t[i:i+1], E_v[i:i+1])[0, 0] for i in range(len(idx))]))), 4),
        }
        k = res["per_kind"][kind]
        print(f"\n== {kind} (n={len(idx)}) ==")
        print(f"  对自己图 gap 中位 {k['own_gap_median']:+.5f}  为正比例 {k['own_gap_positive_rate']:.3f}")
        print(f"  对其他图 gap 中位 {k['other_gap_median']:+.5f}  为正比例 {k['other_gap_positive_rate']:.3f}")
        print(f"  到语料均值余弦  真 {k['mean_cos_true']:.4f}  变体 {k['mean_cos_var']:.4f}")
        print(f"  向量模长中位    真 {k['norm_true_median']:.4f}  变体 {k['norm_var_median']:.4f}")
        print(f"  真句与变体余弦中位 {k['text_sim_median']:.4f}")

    os.makedirs(OUT, exist_ok=True)
    outp = os.path.join(OUT, f"shuffle_inversion_{args.model.split('/')[-1]}.json")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"\n落盘 -> {outp}")


if __name__ == "__main__":
    main()
