# -*- coding: utf-8 -*-
"""三项几何诊断复算：各向异性、hubness、RRF。

本脚本的唯一目的是让 REPORT.md「结果九」的三个读数可复现。那三项此前只
存在于终端输出，被转录进报告但没有脚本，等于不可复现。本脚本从 DGX 上
已有的三个相似度矩阵直接重算，不重新编码，也不需要 GPU。

三个矩阵的来源与被测对象：
  f_i2t_simmat.npy   v5-omni，图对文，500 图 × 500 caption，正确项在对角
  f_a2t_simmat.npy   v5-omni，音频对文，500 音频 × 500 描述，正确项在对角
  g_clap_simmat.npy  CLAP，音频对文，500 音频 × 500 描述，正确项在对角

三项的定义与报告口径严格一致：
  anisotropy 在相似度矩阵本身上算（不是 embedding 矩阵）。满秩上界是
              min(n_query, n_cand) = 500。有效秩用奇异值谱的指数熵
              exp(-Σ p log p)，p = λ_i / Σλ。第一奇异值占比 = λ_1 / Σλ。
  hubness     查询侧 top-k 的候选 in-degree。自匹配（对角）不计入候选，
              否则每条候选必得自己那一票，分布被系统性拉平。每查询贡献
              k 票，故 deg_mean 恒为 k。基尼系数度量这些票集中在少数候选的程度。
  rrf         两路 rank 矩阵按 RRF 融合，score = Σ 1/(k0 + rank + 1)，
              rank 为 0-based。这里融合的是同一几何结构的两个不同质量路由，
              是诊断性合成，不是真实检索场景。

用法：~/xinfer-env/bin/python diag_geometry_simmat.py [--check]
  --check  除打印外，把结果与报告中的历史读数逐项比对并标出偏差
"""
import argparse, json, os
import numpy as np

BASE = "/home/ke/crossmodal_lab/results"
OUT = os.path.join(BASE, "geometry_diag")

MATS = {
    "omni_i2t": os.path.join(BASE, "jina-embeddings-v5-omni-small/f_i2t_simmat.npy"),
    "omni_a2t": os.path.join(BASE, "jina-embeddings-v5-omni-small/f_a2t_simmat.npy"),
    "clap_a2t": os.path.join(BASE, "clap_general/g_clap_simmat.npy"),
}

# 报告中已录的历史读数，用于 --check 比对
HIST = {
    "omni_i2t": {"eff_rank": 85.1, "rank_ratio": 0.170, "top_sv": 0.157, "neg_ratio": 0.089,
                 "deg_max": 55, "deg_ratio": 5.5, "gini": 0.374, "top1pct_share": 0.040,
                 "r1": 0.902, "median_rank": 1.0},
    "omni_a2t": {"eff_rank": 7.3, "rank_ratio": 0.015, "top_sv": 0.655, "neg_ratio": 0.000,
                 "deg_max": 115, "deg_ratio": 11.5, "gini": 0.741, "top1pct_share": 0.101,
                 "r1": 0.010, "median_rank": 195.0},
    "clap_a2t": {"eff_rank": 7.1, "rank_ratio": 0.014, "top_sv": 0.605, "neg_ratio": 0.006,
                 "deg_max": 124, "deg_ratio": 12.4, "gini": 0.752, "top1pct_share": 0.103,
                 "r1": 0.020, "median_rank": 104.0},
}


def eff_rank(S):
    """指数熵有效秩。满秩上界 = min(S.shape)。"""
    s = np.linalg.svd(S, compute_uv=False)
    lam = s ** 2
    lam = lam[lam > 1e-12]
    p = lam / lam.sum()
    er = float(np.exp(-(p * np.log(p)).sum()))
    return er, float(er / min(S.shape))


def top_sv(S):
    s = np.linalg.svd(S, compute_uv=False)
    lam = s ** 2
    return float(lam[0] / lam.sum())


def neg_ratio(S):
    """非对角元素的负值占比。对角线恒为正，必须排除否则拉高正比例。"""
    n = S.shape[0]
    off = S + np.where(np.eye(n, dtype=bool), 1e9, 0)
    return float((off < 0).mean())


def hubness(S, k=10):
    """候选 in-degree 分布。排除自匹配：对角线置 -inf 后再取 top-k。

    不排除时每条候选必得自己那一票，deg_max 与基尼都会被系统性压低
    （实测 omni_i2t  deg_max 50 对 55，基尼 0.339 对 0.374）。
    """
    n = S.shape[0]
    off = S.copy()
    np.fill_diagonal(off, -np.inf)
    idx = np.argsort(-off, axis=1)[:, :k]
    deg = np.bincount(idx.ravel(), minlength=n).astype(float)
    d = np.sort(deg)
    m = len(d)
    gini = float((2 * np.arange(1, m + 1) - m - 1).dot(d) / (m * d.sum()))
    n_top = max(1, int(round(n * 0.01)))
    return {"deg_max": float(deg.max()), "deg_mean": float(deg.mean()),
            "deg_ratio": float(deg.max() / deg.mean()),
            "gini": gini,
            "top1pct_share": float(np.sort(deg)[-n_top:].sum() / deg.sum()),
            "top5": [{"cand": int(i), "deg": int(deg[i])} for i in np.argsort(-deg)[:5]]}


def retrieval(S):
    """对角线即正确项。

    unrel_median 是每查询的中位数再取中位（median of per-query medians），
    不是全局中位。两种算法的实测差异不可忽略：omni_a2t 全局 0.2398 对
    逐查询 0.2511，clap_a2t 0.3265 对 0.3353。报告的 margin 一律用逐查询口径。
    """
    n = S.shape[0]
    r = np.argsort(-S, axis=1).argsort(axis=1)[np.arange(n), np.arange(n)] + 1
    off = S + np.where(np.eye(n, dtype=bool), 1e9, 0)
    per_q = np.array([np.median(off[i][off[i] < 1e8]) for i in range(n)])
    return {"r@1": float((r <= 1).mean()), "r@5": float((r <= 5).mean()),
            "r@10": float((r <= 10).mean()), "median_rank": float(np.median(r)),
            "rel_median": float(np.median(S.diagonal())),
            "unrel_median": float(np.median(per_q)),
            "margin": float(np.median(S.diagonal()) - np.median(per_q))}


def rrf_fuse(mats, k0=60):
    """等权 RRF。mats 按同一位置(query i, cand i) 融合。"""
    n = mats[0].shape[0]
    ranks = [np.argsort(-S, axis=1).argsort(axis=1) for S in mats]
    score = sum(1.0 / (k0 + R.astype(np.float64) + 1.0) for R in ranks)
    order = np.argsort(-score, axis=1)
    r = order.argsort(axis=1)[np.arange(n), np.arange(n)] + 1
    return {"r@1": float((r <= 1).mean()), "r@5": float((r <= 5).mean()),
            "r@10": float((r <= 10).mean()), "median_rank": float(np.median(r)),
            "k0": k0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--n", type=int, default=None, help="截断到前 n 行做快速自检")
    args = ap.parse_args()

    S = {}
    for tag, p in MATS.items():
        if not os.path.exists(p):
            print(f"[缺] {tag}: {p} 不存在")
            continue
        m = np.load(p)
        if args.n:
            m = m[:args.n, :args.n]
        S[tag] = m
        print(f"[载] {tag} shape={m.shape}")

    res = {"anisotropy": {}, "hubness": {}, "retrieval": {}}
    for tag, m in S.items():
        er, rr = eff_rank(m)
        res["anisotropy"][tag] = {"eff_rank": round(er, 1), "rank_ratio": round(rr, 3),
                                  "top_sv_frac": round(top_sv(m), 3),
                                  "neg_ratio": round(neg_ratio(m), 3)}
        res["hubness"][tag] = {k: (round(v, 3) if isinstance(v, float) else v)
                               for k, v in hubness(m).items() if k != "top5"}
        res["retrieval"][tag] = {k: (round(v, 4) if isinstance(v, float) else v)
                                 for k, v in retrieval(m).items()}
        print(f"\n== {tag} ==")
        print(f"  有效秩 {er:.1f}  秩占比 {rr:.1%}  第一奇异值 {top_sv(m):.1%}  负相似度 {neg_ratio(m):.1%}")
        h = hubness(m)
        print(f"  in-degree 最大 {h['deg_max']:.0f}  最大/平均 {h['deg_ratio']:.1f}  基尼 {h['gini']:.3f}  前1%占比 {h['top1pct_share']:.1%}")
        rv = retrieval(m)
        print(f"  R@1 {rv['r@1']:.3f}  R@5 {rv['r@5']:.3f}  中位排名 {rv['median_rank']:.0f}  "
              f"相关中位 {rv['rel_median']:.4f}  无关中位 {rv['unrel_median']:.4f}  margin {rv['rel_median']-rv['unrel_median']:.4f}")

    if "omni_i2t" in S and "omni_a2t" in S:
        res["rrf"] = rrf_fuse([S["omni_i2t"], S["omni_a2t"]])
        print(f"\n== RRF 融合 omni_i2t + omni_a2t (k0=60) ==")
        print(f"  R@1 {res['rrf']['r@1']:.3f}  R@5 {res['rrf']['r@5']:.3f}  中位排名 {res['rrf']['median_rank']:.0f}")
        print(f"  两路单独 R@1: 图 {res['retrieval']['omni_i2t']['r@1']:.3f}  音频 {res['retrieval']['omni_a2t']['r@1']:.3f}")

    if args.check:
        print("\n== 与报告历史读数比对 ==")
        bad = 0
        for tag, h in HIST.items():
            if tag not in S:
                continue
            for key, want in h.items():
                got = None
                for grp in ("anisotropy", "hubness", "retrieval"):
                    if key in res[grp].get(tag, {}):
                        got = res[grp][tag][key]
                alias = {"deg_ratio": "deg_ratio", "top1pct_share": "top1pct_share",
                         "r1": "r@1"}
                key2 = alias.get(key, key)
                for grp in ("anisotropy", "hubness", "retrieval"):
                    if key2 in res[grp].get(tag, {}):
                        got = res[grp][tag][key2]
                if got is None:
                    continue
                tol = 0.02 if want >= 1 else max(0.002, want * 0.15)
                ok = abs(got - want) <= tol
                flag = "OK  " if ok else "偏差"
                if not ok:
                    bad += 1
                print(f"  {flag} {tag}.{key}: 复算 {got}  报告 {want}")
        print(f"\n偏差项 {bad} 个")

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "geometry_diag.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"\n落盘 -> {OUT}/geometry_diag.json")


if __name__ == "__main__":
    main()
