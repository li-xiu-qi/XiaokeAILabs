# -*- coding: utf-8 -*-
"""各向异性实验：用文献标准定义算，并出图。

背景：REPORT「结果九」原有的有效秩读数（85.1 / 7.3 / 7.1）用六种口径都复现不出，
且方向与 embedding 矩阵实测相反。本篇不复原旧数，改用文献里通用的标准定义重做，
回答真正的问题：音频侧与图文侧哪个更各向异性，差多少。

四个定义，全部来自常见用法，不自己造：
  cone_norm      锥集中度。所有向量已 L2 归一化，故 ||mean||² / n 即方差落在均值
                 方向上的比例。全同向时 1，均匀铺开时趋 0。
  mean_cos        随机两两余弦的均值。各向同性时趋 0，各向异性时显著为正。
  cos_p95         随机两两余弦的 95 分位。看尾部，mean 会被小值拉平。
  eff_rank_ratio  去均值后的指数熵有效秩占 min(n,d) 比例，铺满程度。
  top_sv_frac     去均值后第一奇异值占总方差比例。

可视化四联：随机对余弦分布直方图（各向异性的直观形态）、锥集中度条形、
奇异谱、PCA 二维投影散点。

用法：~/xinfer-env/bin/python diag_anisotropy_v2.py [--n 500]
"""
import argparse, io, json, os, warnings
import numpy as np

warnings.filterwarnings("ignore")

CACHE = "/home/ke/crossmodal_lab/results/diag_cache"
OUT = "/home/ke/crossmodal_lab/results/geometry_diag"

FLICKR_CSV = "/home/ke/crossmodal_lab/datasets/flickr30k/flickr_annotations_30k.csv"
FSD = ("/home/ke/crossmodal_lab/datasets/fsd50k/parquet/"
       "train-00000-of-00031-31accf869b307ce3.parquet")
SR_OMNI = 16000


def log(m):
    print(m, flush=True)


def load_fsd_desc(n):
    import pyarrow.parquet as pq
    t = pq.read_table(FSD)
    desc = t.column("description").to_pylist()
    out = []
    for i in range(min(n, len(desc))):
        d = (desc[i] or "").strip()
        if d:
            out.append(d)
    return out[:n]


def enc_omni_text_fsd(n):
    """v5-omni 对 FSD50K 描述的文本向量，补上与音频同数据集同模型的文本侧。"""
    import sys
    sys.path.insert(0, "/home/ke/crossmodal_lab/scripts")
    from mm_lab import CrossModalLab
    p = os.path.join(CACHE, "omni_txt_fsd.npy")
    if os.path.exists(p):
        return np.load(p)
    lab = CrossModalLab("jinaai/jina-embeddings-v5-omni-small")
    descs = load_fsd_desc(n)
    E = lab.encode_text(descs, batch_size=32)
    lab.close()
    np.save(p, E)
    log(f"  编码 omni 文本 {E.shape}")
    return E


# ---------- 指标 ----------

def cone_norm(X):
    """锥集中度：||mean||^2 / n。向量已归一化，这就是均值方向上的方差占比。"""
    m = X.mean(axis=0)
    return float(m @ m)


def pair_cos_stats(X, rng_seed=0, max_n=500):
    """随机对余弦的均值与 95 分位。

    只用互不相同的行，避免同文本重复（FSD50K 前 500 条描述去重后仅 150 条，
    'electric guitar' 一条出现 88 次）把分布顶出假峰值。
    """
    X = np.unique(X, axis=0)
    n = X.shape[0]
    if n > max_n:
        rng = np.random.default_rng(rng_seed)
        X = X[rng.choice(n, max_n, replace=False)]
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    S = Xn @ Xn.T
    iu = np.triu_indices(S.shape[0], 1)
    v = S[iu]
    return float(v.mean()), float(np.percentile(v, 95)), v


def dedup_note(X):
    u = np.unique(X, axis=0)
    return X.shape[0], u.shape[0]


def eff_rank_ratio(X):
    Xc = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    lam = s ** 2
    lam = lam[lam > 1e-12]
    p = lam / lam.sum()
    er = float(np.exp(-(p * np.log(p)).sum()))
    return er / min(X.shape), float(lam[0] / lam.sum())


# ---------- 主流程 ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--fig", default="anisotropy.png")
    args = ap.parse_args()

    log("准备 embedding")
    mats = {
        "omni图(Flickr)": np.load(f"{CACHE}/omni_img.npy"),
        "omni音(FSD50K)": np.load(f"{CACHE}/omni_aud.npy"),
        "omni文(FSD50K)": enc_omni_text_fsd(args.n),
        "clap音(FSD50K)": np.load(f"{CACHE}/clap_audio.npy"),
        "clap文(FSD50K)": np.load(f"{CACHE}/clap_text.npy"),
    }

    res, dists = {}, {}
    print(f"\n{'矩阵':18s}{'去重前':>7s}{'去重后':>7s}{'锥集中度':>10s}{'mean_cos':>10s}{'cos_p95':>10s}{'eff_rank占比':>13s}{'首奇异':>9s}")
    for tag, X in mats.items():
        n0, n1 = dedup_note(X)
        cn = cone_norm(X)
        mc, p95, v = pair_cos_stats(X)
        er, ts = eff_rank_ratio(X)
        dists[tag] = v
        res[tag] = {"n_raw": n0, "n_unique": n1, "cone_norm": round(cn, 4),
                    "mean_cos": round(mc, 4), "cos_p95": round(p95, 4),
                    "eff_rank_ratio": round(er, 4), "top_sv_frac": round(ts, 4)}
        print(f"{tag:18s}{n0:>7d}{n1:>7d}{cn:>10.4f}{mc:>10.4f}{p95:>10.4f}{er:>13.4f}{ts:>9.4f}")

    # ---------- 出图 ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    # 找一个已有的中文字体，避免豆腐块
    cand = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "WenQuanYi Zen Hei",
            "Source Han Sans SC", "Arial Unicode MS"]
    have = {f.name for f in font_manager.fontManager.ttflist}
    zh = next((c for c in cand if c in have), None)
    if zh:
        plt.rcParams["font.sans-serif"] = [zh]
    plt.rcParams["axes.unicode_minus"] = False

    tags = list(mats)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    colors = plt.cm.tab10(np.linspace(0, 0.5, len(tags)))

    ax = axes[0, 0]
    for (tag, v), c in zip(dists.items(), colors):
        ax.hist(v, bins=80, range=(-1, 1), histtype="step", lw=1.6,
                color=c, label=tag, density=True)
    ax.axvline(0, color="k", lw=0.6, ls=":")
    ax.set_xlabel("随机两两余弦")
    ax.set_ylabel("密度")
    ax.set_title("随机对余弦分布：越往右越集中（各向异性）")
    ax.legend(fontsize=7.5, loc="upper left")

    ax = axes[0, 1]
    vals = [res[t]["cone_norm"] for t in tags]
    labs = [f"{t}\n(去重后 {res[t]['n_unique']})" for t in tags]
    ax.barh(range(len(tags)), vals, color=colors)
    ax.set_yticks(range(len(tags)))
    ax.set_yticklabels(labs, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("锥集中度  ||mean||²/n")
    ax.set_title("方差落在均值方向上的比例")
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=8)

    ax = axes[1, 0]
    for tag, c in zip(tags, colors):
        Xc = mats[tag] - mats[tag].mean(axis=0, keepdims=True)
        s = np.linalg.svd(Xc, compute_uv=False)
        lam = s ** 2
        lam = lam / lam.sum()
        cum = np.cumsum(lam)[:80]
        ax.plot(range(1, len(cum) + 1), cum, lw=1.5, color=c, label=tag)
    ax.set_yscale("log")
    ax.set_xlabel("奇异方向序号")
    ax.set_ylabel("累计方差占比（log）")
    ax.set_title("奇异谱：曲线越陡越集中")
    ax.legend(fontsize=7.5)

    ax = axes[1, 1]
    for tag, c in zip(tags, colors):
        Xc = mats[tag] - mats[tag].mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        P = (U[:, :2] * S[:2]) @ Vt[:2]
        ax.scatter(P[:, 0], P[:, 1], s=6, alpha=0.55, color=c, label=tag)
    ax.axhline(0, color="k", lw=0.4)
    ax.axvline(0, color="k", lw=0.4)
    ax.set_xlabel("第一主方向")
    ax.set_ylabel("第二主方向")
    ax.set_title("去均值后 PCA 二维投影")
    ax.legend(fontsize=7.5, markerscale=2)

    fig.suptitle("各向异性：五个 embedding 矩阵的几何形态", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(args.fig, dpi=110)
    log(f"图已存 {args.fig}")

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "anisotropy_v2.json"), "w", encoding="utf-8") as f:
        json.dump({"metrics": res, "fig": args.fig}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
