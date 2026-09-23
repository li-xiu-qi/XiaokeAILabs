# -*- coding: utf-8 -*-
"""分析：Δsim 统计、模型间方向矩阵、跨语言基准、显著性检验、图表。

用法：
  python analyze.py [raw_sims.csv 路径]
"""

import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402

# 单侧条件折叠：rel_one = mean(rel_one_a, rel_one_b)，按 (model, lang_pair, theme)
FOLD: Dict[str, List[str]] = {
    "rel_one": ["rel_one_a", "rel_one_b"],
    "irr_one": ["irr_one_a", "irr_one_b"],
    "rel_both": ["rel_both"],
    "irr_both": ["irr_both"],
}


def load_raw(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def fold_conditions(df: pd.DataFrame) -> pd.DataFrame:
    """把单侧 a/b 变体折叠成单一条件，base 原样。"""
    parts = [df[df["condition"] == "base"].copy()]
    for new_cond, members in FOLD.items():
        sub = df[df["condition"].isin(members)]
        collapsed = (
            sub.groupby(["model", "lang", "lang_pair", "theme_id"])["sim"]
            .mean()
            .reset_index()
        )
        collapsed["condition"] = new_cond
        parts.append(collapsed)
    return pd.concat(parts, ignore_index=True)


def add_delta(df: pd.DataFrame) -> pd.DataFrame:
    """Δsim = sim(条件) - sim(base)，按 (model, lang_pair, theme) 对齐。"""
    base = (
        df[df["condition"] == "base"][["model", "lang_pair", "theme_id", "sim"]]
        .rename(columns={"sim": "sim_base"})
    )
    merged = df.merge(base, on=["model", "lang_pair", "theme_id"], how="inner")
    out = merged[merged["condition"] != "base"].copy()
    out["delta"] = out["sim"] - out["sim_base"]
    return out


def summarize(delta_df: pd.DataFrame) -> pd.DataFrame:
    """每 (model, lang_pair, condition) 的 Δ 统计与 Wilcoxon 符号秩检验。"""
    rows = []
    for (model, lp, cond), g in delta_df.groupby(["model", "lang_pair", "condition"]):
        d = g["delta"].values
        try:
            p = stats.wilcoxon(d, alternative="two-sided").pvalue
        except Exception:
            p = np.nan
        rows.append({
            "model": model, "lang_pair": lp, "condition": cond,
            "n": len(d),
            "median_delta": float(np.median(d)),
            "mean_delta": float(np.mean(d)),
            "frac_down": float(np.mean(d < 0)),
            "wilcoxon_p": float(p),
        })
    return pd.DataFrame(rows)


def base_bench(df: pd.DataFrame) -> pd.DataFrame:
    """base 条件下各 (model, lang_pair) 的相似度基准。"""
    b = df[df["condition"] == "base"]
    return b.groupby(["model", "lang", "lang_pair"])["sim"].agg(
        n="count", median="median", mean="mean", std="std"
    ).reset_index()


def direction_matrix(summary: pd.DataFrame, lang_pair: str = "zh-zh") -> pd.DataFrame:
    """模型 × 条件 的 Δ 中位数矩阵。"""
    s = summary[summary["lang_pair"] == lang_pair]
    return s.pivot(index="model", columns="condition", values="median_delta").reindex(
        columns=config.ANALYSIS_CONDITIONS[1:]
    )


def make_figures(summary: pd.DataFrame, delta_df: pd.DataFrame, bench: pd.DataFrame, figdir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "AR PL UMing CN", "SimHei", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    os.makedirs(figdir, exist_ok=True)

    order = config.ANALYSIS_CONDITIONS[1:]

    # 图1：方向矩阵热力图（zh-zh）
    dm = direction_matrix(summary, "zh-zh")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    vmax = float(np.nanmax(np.abs(dm.values)))
    im = ax.imshow(dm.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(order)), order, rotation=20, ha="right")
    ax.set_yticks(range(len(dm.index)), dm.index)
    for i in range(dm.shape[0]):
        for j in range(dm.shape[1]):
            ax.text(j, i, f"{dm.values[i, j]:+.3f}", ha="center", va="center", fontsize=9)
    ax.set_title("zh-zh: 加内容后的相似度变化 Δ 中位数（负 = 变像，正 = 变不像）")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig1_direction_matrix_zh.png"), dpi=150)
    plt.close(fig)

    # 图2：每条件 Δ 的原始分布箱线（zh-zh）
    zh = delta_df[delta_df["lang_pair"] == "zh-zh"]
    models = list(summary["model"].unique())
    fig, axes = plt.subplots(1, len(order), figsize=(3.0 * len(order), 4.2), sharey=True)
    for ax, cond in zip(np.atleast_1d(axes), order):
        data = [zh[(zh["model"] == m) & (zh["condition"] == cond)]["delta"].values for m in models]
        ax.boxplot(data, tick_labels=models, showmeans=True)
        ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.set_title(cond, fontsize=10)
        ax.tick_params(axis="x", rotation=35, labelsize=7.5)
    axes[0].set_ylabel("Δsim")
    fig.suptitle("zh-zh: 各条件下 20 个主题的 Δ 分布（每模型一箱）")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig2_delta_distribution_zh.png"), dpi=150)
    plt.close(fig)

    # 图3：跨语言 base 基准
    fig, ax = plt.subplots(figsize=(8, 4.5))
    lps = config.LANG_PAIRS
    x = np.arange(len(models))
    width = 0.8 / len(lps)
    for k, lp in enumerate(lps):
        vals = []
        for m in models:
            v = bench[(bench["model"] == m) & (bench["lang_pair"] == lp)]["median"].values
            vals.append(v[0] if len(v) else np.nan)
        ax.bar(x + (k - (len(lps) - 1) / 2) * width, vals, width, label=lp)
    ax.set_xticks(x, models, rotation=20, ha="right")
    ax.set_ylabel("base sim median")
    ax.set_title("各模型 base 条件下的相似度基准（按语言对）")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig3_crosslingual_base.png"), dpi=150)
    plt.close(fig)
    print(f"figures -> {figdir}")


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(config.RESULTS_DIR, "raw_sims.csv")
    df = load_raw(path)
    folded = fold_conditions(df)
    delta = add_delta(folded)
    summary = summarize(delta)
    bench = base_bench(folded)
    summary.to_csv(os.path.join(config.RESULTS_DIR, "summary_delta.csv"), index=False, encoding="utf-8-sig")
    bench.to_csv(os.path.join(config.RESULTS_DIR, "base_bench.csv"), index=False, encoding="utf-8-sig")
    delta.to_csv(os.path.join(config.RESULTS_DIR, "delta_long.csv"), index=False, encoding="utf-8-sig")
    make_figures(summary, delta, bench, config.FIGURES_DIR)
    print(direction_matrix(summary, "zh-zh").round(4).to_string())
    print()
    print(bench.round(4).to_string())


if __name__ == "__main__":
    main()
