#!/usr/bin/env python3
"""生成预训练原理与能力评测章节的五组顶刊风格科研配图（PNG + SVG）。

门禁原则：
1. 一图聚焦一个问题，不硬塞大段文字表或代码排查正文；
2. 数学公式一律使用 Matplotlib LaTeX (`$...$`) mathtext 渲染；
3. 可选使用 `matplotlib_layout_audit.audit_figure` 检查版面；未配置时跳过。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np

# 可选版面审计工具路径经环境变量 FIGURE_AUDIT_DIR 注入；未配置时跳过审计（不影响出图）
AUDIT_DIR = os.environ.get("FIGURE_AUDIT_DIR")
if AUDIT_DIR and AUDIT_DIR not in sys.path:
    sys.path.insert(0, AUDIT_DIR)

try:
    from matplotlib_layout_audit import audit_figure
    AUDIT_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - 审计工具不在场时降级
    AUDIT_AVAILABLE = False

    def audit_figure(*args, **kwargs):
        return None


sys.path.insert(0, str(Path(__file__).resolve().parent))
import llm_common as lc  # noqa: E402

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "Noto Sans SC", "Arial", "DejaVu Sans"],
        "font.monospace": ["Consolas", "Microsoft YaHei", "DejaVu Sans Mono"],
        "axes.unicode_minus": False,
        "font.size": 9.8,
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF",
    }
)

# 矿物中性顶刊色板
SLATE_DARK = "#0F172A"
SLATE = "#475569"
SLATE_LIGHT = "#94A3B8"
SLATE_BG = "#F8FAFC"
SLATE_BORDER = "#CBD5E1"

INDIGO = "#1E40AF"
INDIGO_MID = "#3B82F6"
INDIGO_BG = "#EFF6FF"
INDIGO_EDGE = "#93C5FD"

TEAL = "#0F766E"
TEAL_MID = "#14B8A6"
TEAL_BG = "#F0FDFA"
TEAL_EDGE = "#5EEAD4"

BURGUNDY = "#991B1B"
BURGUNDY_BG = "#FEF2F2"
BURGUNDY_EDGE = "#FCA5A5"

AMBER_DARK = "#92400E"
AMBER_BG = "#FFFBEB"
AMBER_EDGE = "#FCD34D"


def _arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = SLATE,
    lw: float = 1.25,
    ls: str = "-",
    rad: float = 0.0,
    ms: float = 10.5,
    zorder: int = 4,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=ms,
            linewidth=lw,
            linestyle=ls,
            color=color,
            shrinkA=0,
            shrinkB=0,
            connectionstyle=f"arc3,rad={rad}",
            zorder=zorder,
        )
    )


def _pill(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    fc: str,
    ec: str,
    tc: str = SLATE_DARK,
    fs: float = 8.6,
    bold: bool = False,
    sub: str | None = None,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.0,rounding_size=0.015",
            facecolor=fc,
            edgecolor=ec,
            linewidth=1.1,
            zorder=3,
        )
    )
    if sub:
        ax.text(
            x + w / 2,
            y + h * 0.64,
            text,
            ha="center",
            va="center",
            fontsize=fs,
            fontweight="bold" if bold else "normal",
            color=tc,
            zorder=4,
        )
        ax.text(
            x + w / 2,
            y + h * 0.27,
            sub,
            ha="center",
            va="center",
            fontsize=fs - 1.1,
            color=SLATE,
            zorder=4,
        )
    else:
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=fs,
            fontweight="bold" if bold else "normal",
            color=tc,
            zorder=4,
        )


def _panel_tag(
    ax,
    tag: str,
    title: str,
    subtitle: str | None = None,
    y_title: float = 1.075,
    y_sub: float = 1.02,
) -> None:
    ax.text(
        0.0,
        y_title,
        f"{tag}  {title}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.6,
        fontweight="bold",
        color=SLATE_DARK,
    )
    if subtitle:
        ax.text(
            0.0,
            y_sub,
            subtitle,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.2,
            color=SLATE,
        )


def _save_fig(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    issues = audit_figure(fig, name=stem)
    if issues:
        for item in issues:
            print("AUDIT:", item)
        raise RuntimeError(f"Layout audit failed for {stem}: {len(issues)} issue(s)")
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{stem}.png"
    svg_path = out_dir / f"{stem}.svg"
    fig.savefig(png_path, dpi=220, bbox_inches="tight", facecolor="#FFFFFF")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="#FFFFFF")
    plt.close(fig)
    status = "audited" if AUDIT_AVAILABLE else "layout audit skipped"
    print(f"[OK] Saved {png_path.name} + {svg_path.name} ({status})")


# ---------------------------------------------------------------------------
# 图 1：自回归错位损失 + 因果下三角掩码 + 现代残差主干与五道稳态护栏
# ---------------------------------------------------------------------------
def plot_causal_architecture(out_dir: Path) -> None:
    fig = plt.figure(figsize=(14.4, 5.9), facecolor="#FFFFFF")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.16)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    for ax in (ax1, ax2):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    # ---- Panel a: Next-Token Prediction Shift + 5x5 Causal Mask ----
    _panel_tag(
        ax1,
        "a",
        "Next-Token 并行监督与下三角因果掩码",
        "单次前向传播并行计算全序列所有位置的交叉熵梯度",
        y_title=1.05,
        y_sub=0.995,
    )

    tokens_in = [r"$x_0$", r"$x_1$", r"$x_2$", r"$x_3$", r"$x_4$"]
    tokens_tg = [r"$x_1$", r"$x_2$", r"$x_3$", r"$x_4$", r"$x_5$"]
    ax1.text(0.04, 0.90, "输入前缀张量 idx[:, :-1]", fontsize=8.6, color=INDIGO, fontweight="bold")
    tx0, tw, gap = 0.04, 0.145, 0.032
    for i, (ti, tg) in enumerate(zip(tokens_in, tokens_tg)):
        xx = tx0 + i * (tw + gap)
        _pill(ax1, xx, 0.81, tw, 0.068, ti, INDIGO_BG, INDIGO_EDGE, INDIGO, 9.2, True)
        _pill(ax1, xx, 0.675, tw, 0.068, tg, TEAL_BG, TEAL_EDGE, TEAL, 9.2, True)
        _arrow(ax1, (xx + tw / 2, 0.81), (xx + tw / 2, 0.745), color=SLATE_LIGHT, lw=1.15, ms=9)
    ax1.text(0.04, 0.635, "错位目标标签 targets[:, 1:]（并行对齐监督）", fontsize=8.4, color=TEAL, fontweight="bold")

    ax1.text(
        0.04,
        0.54,
        r"交叉熵目标：$\mathcal{L}_{\mathrm{NTP}} = -\frac{1}{|\mathcal{M}|}\sum_{t\in\mathcal{M}}\ln P_\theta(x_{t+1}\mid x_{\leq t})$",
        fontsize=9.2,
        color=SLATE_DARK,
    )

    gx0, gy0, cell = 0.06, 0.09, 0.072
    ax1.text(gx0, gy0 + 5 * cell + 0.025, r"下三角因果掩码 $M$ ($5\times 5$)", fontsize=8.9, fontweight="bold", color=SLATE_DARK)
    for r in range(5):
        ax1.text(gx0 - 0.022, gy0 + (4 - r) * cell + cell / 2, rf"$q_{r}$", ha="right", va="center", fontsize=8.2, color=SLATE)
        ax1.text(gx0 + r * cell + cell / 2, gy0 - 0.032, rf"$k_{r}$", ha="center", va="center", fontsize=8.2, color=SLATE)
        for c in range(5):
            active = c <= r
            fc = TEAL if active and r == c else (TEAL_BG if active else SLATE_BG)
            ec = TEAL if active else SLATE_BORDER
            ax1.add_patch(
                Rectangle(
                    (gx0 + c * cell, gy0 + (4 - r) * cell),
                    cell - 0.005,
                    cell - 0.005,
                    facecolor=fc,
                    edgecolor=ec,
                    linewidth=0.9,
                    zorder=3,
                )
            )
            sym = "1" if active else r"$-\infty$"
            col_txt = "#FFFFFF" if active and r == c else (TEAL if active else SLATE_LIGHT)
            ax1.text(
                gx0 + c * cell + (cell - 0.005) / 2,
                gy0 + (4 - r) * cell + (cell - 0.005) / 2,
                sym,
                ha="center",
                va="center",
                fontsize=7.8,
                color=col_txt,
                fontweight="bold" if active else "normal",
                zorder=4,
            )

    ax1.text(0.50, 0.34, r"$\bullet$ 位置 $i$ 仅聚合 $j \leq i$ 历史信息", fontsize=8.5, color=TEAL, fontweight="bold")
    ax1.text(0.50, 0.25, r"$\bullet$ 上三角 $j > i$ 掩蔽为 $-\infty$", fontsize=8.5, color=SLATE)
    ax1.text(0.50, 0.16, r"$\bullet$ 填充位按 `ignore_index=-100` 跳过", fontsize=8.3, color=SLATE)

    # ---- Panel b: Residual Stream Highway + 5 Guardrails ----
    _panel_tag(
        ax2,
        "b",
        "Transformer 块前向路径与数值稳定措施",
        "对应 llm_common.py CausalLM：残差直通与各层归一化、旋转位置编码、logits 限幅",
        y_title=1.05,
        y_sub=0.995,
    )

    # 左侧主残差干流（x=0.14 贯穿上下）
    sx = 0.14
    _pill(ax2, 0.03, 0.83, 0.30, 0.085, "tok_emb + embed_norm", SLATE_BG, SLATE_BORDER, SLATE_DARK, 8.6, True, sub="防嵌入初始尺度漂移")
    _arrow(ax2, (sx, 0.83), (sx, 0.625), color=SLATE_DARK, lw=2.0, ms=11)
    ax2.text(sx - 0.025, 0.72, "残\n差\n流", ha="right", va="center", fontsize=8.0, color=SLATE)

    # 加法节点 1 (Attention residual merge)
    c1 = Circle((sx, 0.595), 0.030, facecolor="#FFFFFF", edgecolor=SLATE_DARK, lw=1.4, zorder=5)
    ax2.add_patch(c1)
    ax2.text(sx, 0.595, "+", ha="center", va="center", fontsize=12, fontweight="bold", color=SLATE_DARK, zorder=6)

    # Attention 旁路框
    _arrow(ax2, (sx, 0.76), (0.28, 0.76), color=INDIGO, lw=1.3, ms=9.5)
    _pill(
        ax2,
        0.28,
        0.67,
        0.68,
        0.13,
        "① Pre-RMSNorm -> QKV投影 -> ② RoPE -> ③ QK-Norm -> SDPA",
        INDIGO_BG,
        INDIGO_EDGE,
        INDIGO,
        8.2,
        True,
        sub="去均值均方根缩放 (Zhang, 2019) + 相对旋转 + 点积前防 Softmax 饱和",
    )
    _arrow(ax2, (0.62, 0.67), (0.62, 0.595), color=INDIGO, lw=1.3, ms=0.1)
    _arrow(ax2, (0.62, 0.595), (sx + 0.030, 0.595), color=INDIGO, lw=1.3, ms=10)

    _arrow(ax2, (sx, 0.565), (sx, 0.365), color=SLATE_DARK, lw=2.0, ms=11)

    # 加法节点 2 (MLP residual merge)
    c2 = Circle((sx, 0.335), 0.030, facecolor="#FFFFFF", edgecolor=SLATE_DARK, lw=1.4, zorder=5)
    ax2.add_patch(c2)
    ax2.text(sx, 0.335, "+", ha="center", va="center", fontsize=12, fontweight="bold", color=SLATE_DARK, zorder=6)

    # MLP 旁路框
    _arrow(ax2, (sx, 0.49), (0.28, 0.49), color=TEAL, lw=1.3, ms=9.5)
    _pill(
        ax2,
        0.28,
        0.40,
        0.68,
        0.13,
        r"④ Pre-RMSNorm -> $\mathrm{ReLU}^2$ MLP (支持切换 SwiGLU)",
        TEAL_BG,
        TEAL_EDGE,
        TEAL,
        8.3,
        True,
        sub=r"稀疏激活 $\max(0, x)^2$ 加快收敛 (So et al., 2021)；全线性层禁用 bias",
    )
    _arrow(ax2, (0.62, 0.40), (0.62, 0.335), color=TEAL, lw=1.3, ms=0.1)
    _arrow(ax2, (0.62, 0.335), (sx + 0.030, 0.335), color=TEAL, lw=1.3, ms=10)

    # 输出端 Final RMSNorm + Softcap
    _arrow(ax2, (sx, 0.305), (sx, 0.195), color=SLATE_DARK, lw=2.0, ms=11)
    _pill(
        ax2,
        0.03,
        0.055,
        0.93,
        0.14,
        "⑤ 词表权重解耦 (tie_weights=False) -> lm_head -> Logit Soft-Capping",
        BURGUNDY_BG,
        BURGUNDY_EDGE,
        BURGUNDY,
        8.5,
        True,
        sub=r"输出层限幅 $15\tanh(z/15)$ 柔和压制极端 Logits 峰值并保留梯度，送入 CE Loss",
    )

    _save_fig(fig, out_dir, "pretrain-causal-architecture")


# ---------------------------------------------------------------------------
# 图 2：优化器几何分治（85.7% 走 Muon vs 14.3% 走 AdamW）与 Newton-Schulz 正交均能化
# ---------------------------------------------------------------------------
def plot_muon_orthogonalization(out_dir: Path) -> None:
    fig = plt.figure(figsize=(13.8, 5.6), facecolor="#FFFFFF")
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.02, 0.96, "按参数形态选择优化器与 Muon Newton-Schulz 正交化", fontsize=11.8, fontweight="bold", color=SLATE_DARK, va="top")
    ax.text(0.02, 0.905, "二维隐藏权重矩阵采用 Muon 正交更新，词嵌入与输出头保留 AdamW", fontsize=8.6, color=SLATE, va="top")

    # ---- Top Bar: Parameter Routing Breakdown ----
    bx, by, bw, bh = 0.04, 0.72, 0.92, 0.09
    w_mu = bw * (1769472 / 2065536)
    w_em = bw * (148032 / 2065536)
    w_hd = bw - w_mu - w_em
    ax.add_patch(Rectangle((bx, by), w_mu, bh, facecolor=TEAL, edgecolor="#FFFFFF", lw=1.2, zorder=3))
    ax.add_patch(Rectangle((bx + w_mu, by), w_em, bh, facecolor=INDIGO, edgecolor="#FFFFFF", lw=1.2, zorder=3))
    ax.add_patch(Rectangle((bx + w_mu + w_em, by), w_hd, bh, facecolor=SLATE, edgecolor="#FFFFFF", lw=1.2, zorder=3))
    ax.text(bx + w_mu / 2, by + bh / 2, "blocks.* 二维隐藏矩阵  85.7% (1,769,472 参数) -> Muon (NS-5, lr=0.02)", ha="center", va="center", color="#FFFFFF", fontsize=8.8, fontweight="bold", zorder=4)
    ax.text(bx + bw, by - 0.036, "tok_emb 7.2% (lr=0.3) + lm_head 7.2% (lr=0.004) -> AdamW", ha="right", va="center", color=INDIGO, fontsize=8.4, fontweight="bold")
    ax.text(bx, by + bh + 0.022, "本地 4 层 CausalLM 参数分组占比（总参数 2,065,536，pretrain_metrics.json）", fontsize=8.5, fontweight="bold", color=SLATE_DARK)

    # ---- Middle: Ill-conditioned Spectrum -> NS-5 -> Quasi-Orthogonal ----
    sx0, sy0, sw, sh = 0.04, 0.24, 0.36, 0.38
    ax.add_patch(FancyBboxPatch((sx0, sy0), sw, sh, boxstyle="round,pad=0,rounding_size=0.02", facecolor=SLATE_BG, edgecolor=SLATE_BORDER, lw=1.0))
    ax.text(sx0 + sw / 2, sy0 + sh - 0.042, "原始动量奇异值分布（主方向高度集中）", ha="center", va="center", fontsize=9.0, fontweight="bold", color=BURGUNDY)
    sv_raw = [0.92, 0.36, 0.16, 0.08, 0.04, 0.02]
    for i, v in enumerate(sv_raw):
        ax.add_patch(Rectangle((sx0 + 0.042 + i * 0.048, sy0 + 0.085), 0.030, v * 0.21, facecolor=BURGUNDY if i == 0 else SLATE_LIGHT, zorder=4))
    ax.text(sx0 + sw / 2, sy0 + 0.038, "最大奇异方向占主导，其余方向更新幅度偏小", ha="center", va="center", fontsize=8.1, color=SLATE)

    _arrow(ax, (sx0 + sw + 0.02, sy0 + sh * 0.52), (0.60 - 0.02, sy0 + sh * 0.52), color=TEAL, lw=2.2, ms=13)
    ax.text(0.50, sy0 + sh * 0.52 + 0.065, "NS-5 五步矩阵乘正交化", ha="center", va="center", fontsize=8.6, fontweight="bold", color=TEAL)
    ax.text(0.50, sy0 + sh * 0.52 - 0.055, r"可在 `bfloat16` 下运行，无需 SVD 分解", ha="center", va="center", fontsize=7.9, color=SLATE)

    sx1 = 0.60
    ax.add_patch(FancyBboxPatch((sx1, sy0), sw, sh, boxstyle="round,pad=0,rounding_size=0.02", facecolor=TEAL_BG, edgecolor=TEAL_EDGE, lw=1.0))
    ax.text(sx1 + sw / 2, sy0 + sh - 0.042, "Newton-Schulz 正交化后", ha="center", va="center", fontsize=9.0, fontweight="bold", color=TEAL)
    sv_ortho = [0.68, 0.68, 0.67, 0.66, 0.65, 0.61]
    for i, v in enumerate(sv_ortho):
        ax.add_patch(Rectangle((sx1 + 0.042 + i * 0.048, sy0 + 0.085), 0.030, v * 0.21, facecolor=TEAL, zorder=4))
    ax.plot([sx1 + 0.032, sx1 + sw - 0.032], [sy0 + 0.085 + 0.68 * 0.21] * 2, color=INDIGO, lw=1.0, ls=":")
    ax.text(sx1 + sw / 2, sy0 + 0.038, r"逼近 $U V^\top$：拉平各奇异方向更新幅度", ha="center", va="center", fontsize=8.1, color=TEAL)

    # ---- Bottom Wide Banner (plenty of horizontal space, zero overflow!) ----
    _pill(
        ax,
        0.04,
        0.04,
        0.92,
        0.135,
        "更新尺度校准与权重衰减：本地实现与大规模版本差异",
        AMBER_BG,
        AMBER_EDGE,
        AMBER_DARK,
        8.8,
        True,
        sub=r"llm_common.py 使用 $\sqrt{\max(1, A/B)}$ 缩放；Liu et al. (2025) 改为 $0.2\sqrt{\max(A, B)}$ 并加入解耦权重衰减",
    )

    _save_fig(fig, out_dir, "pretrain-muon-orthogonalization")


# ---------------------------------------------------------------------------
# 图 3：Muon vs AdamW 实测收敛轨迹 + GB10 数据扩展律 + grad_accum 显存帕累托
# ---------------------------------------------------------------------------
def plot_muon_convergence_and_scaling(out_dir: Path, results_dir: Path) -> None:
    pm = json.loads((results_dir / "pretrain_metrics.json").read_text(encoding="utf-8"))
    ss = json.loads((results_dir / "scaling" / "scaling_summary.json").read_text(encoding="utf-8"))

    fig = plt.figure(figsize=(16.5, 5.2), facecolor="#FFFFFF")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.12, 1.0, 1.0], wspace=0.45)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    mu = pm["muon"]
    ad = pm["adamw_baseline"]
    mu_tr_s = [h["step"] for h in mu["history"]]
    mu_tr_l = [h["loss"] for h in mu["history"]]
    mu_va_s = [h["step"] for h in mu["val_history"]]
    mu_va_l = [h["val_loss"] for h in mu["val_history"]]

    ad_tr_s = [h["step"] for h in ad["history"]]
    ad_tr_l = [h["loss"] for h in ad["history"]]
    ad_va_s = [h["step"] for h in ad["val_history"]]
    ad_va_l = [h["val_loss"] for h in ad["val_history"]]

    ax1.axvspan(49, 250, color=BURGUNDY_BG, alpha=0.65, zorder=0)
    ax1.text(152, 4.18, "9K-token 微语料过拟合区", ha="center", fontsize=8.0, fontweight="bold", color=BURGUNDY)

    ax1.plot(mu_tr_s, mu_tr_l, color=TEAL_MID, alpha=0.45, lw=1.1, label="Muon train")
    ax1.plot(ad_tr_s, ad_tr_l, color=INDIGO_MID, alpha=0.45, lw=1.1, label="AdamW train")
    ax1.plot(mu_va_s, mu_va_l, color=TEAL, marker="o", ms=4.5, lw=2.0, label="Muon val")
    ax1.plot(ad_va_s, ad_va_l, color=INDIGO, marker="s", ms=4.2, lw=2.0, ls="--", label="AdamW val")

    ax1.scatter([49], [1.3424], color=TEAL, s=55, zorder=5, edgecolors="#FFFFFF", linewidths=1.2)
    ax1.annotate(
        r"Muon 最优: 1.3424" + "\n" + r"(step 49，提速 $3\times$)",
        xy=(49, 1.3424),
        xytext=(58, 0.45),
        fontsize=7.9,
        fontweight="bold",
        color=TEAL,
        arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.0),
    )
    ax1.scatter([149], [1.4086], color=INDIGO, s=50, zorder=5, edgecolors="#FFFFFF", linewidths=1.2)
    ax1.annotate(
        "AdamW 最优: 1.4086\n(step 149)",
        xy=(149, 1.4086),
        xytext=(146, 2.85),
        fontsize=7.9,
        fontweight="bold",
        color=INDIGO,
        arrowprops=dict(arrowstyle="->", color=INDIGO, lw=1.0),
    )

    ax1.set_xlim(-4, 255)
    ax1.set_ylim(0, 6.25)
    ax1.set_xlabel("训练步数 (step)")
    ax1.set_ylabel("交叉熵损失 (nats/token)")
    ax1.legend(loc="upper right", fontsize=7.8, ncol=2)
    ax1.grid(axis="y", color="#E2E8F0", lw=0.6)
    _panel_tag(ax1, "a", "Muon vs AdamW 同配置实测轨迹", r"pretrain_metrics.json ($d_{\mathrm{model}}=192$, 2.07M 参数)")

    # ---- Panel b: GB10 Scaling Tiers (把 BPB 标签放进柱顶白色，远离蓝线) ----
    tiers = ss["modes"]["single"]
    t_names = ["t1_micro\n26.4K tok", "t2_small\n506K tok", "t3_mid\n5.07M tok"]
    t_bpb = np.array([t["pretrain"]["bpb"] for t in tiers], dtype=float)
    t_vloss = np.array([t["pretrain"]["best_val_loss"] for t in tiers], dtype=float)

    x_idx = np.arange(len(tiers))
    bars = ax2.bar(x_idx - 0.17, t_bpb, width=0.34, color=TEAL, label="测试集 BPB (bits/byte)", zorder=3)
    ax2_r = ax2.twinx()
    ax2_r.spines["top"].set_visible(False)
    ax2_r.plot(x_idx + 0.17, t_vloss, color=INDIGO, marker="D", ms=6.2, lw=2.0, label="最佳验证 Loss (nats)", zorder=4)

    for b, val in zip(bars, t_bpb):
        ax2.text(b.get_x() + b.get_width() / 2, val - 0.28, f"{val:.3f}", ha="center", va="top", fontsize=8.0, fontweight="bold", color="#FFFFFF", zorder=5)
    for xx, val in zip(x_idx + 0.17, t_vloss):
        ax2_r.text(xx, val + 0.30, f"{val:.2f}", ha="center", va="bottom", fontsize=8.2, fontweight="bold", color=INDIGO)

    ax2.set_xticks(x_idx)
    ax2.set_xticklabels(t_names, fontsize=8.3)
    ax2.set_ylim(0, 4.65)
    ax2_r.set_ylim(4.0, 10.7)
    ax2.set_ylabel("留出验证集 BPB (越低越好)", color=TEAL)
    ax2_r.set_ylabel("最佳验证集 Loss (nats)", color=INDIGO, labelpad=8)
    ax2.grid(axis="y", color="#E2E8F0", lw=0.6)
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2_r.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=7.8)
    _panel_tag(ax2, "b", "GB10 数据量级扩展实测", r"12.6M 参数固定；历史 BPB 使用旧口径，待重评估")

    # ---- Panel c: grad_accum VRAM vs Throughput Pareto ----
    ga_pts = [
        ("64 x 1", 3861.66, 95586.9 / 1000.0, SLATE, (-980, 1.3)),
        ("32 x 2", 2059.41, 100939.6 / 1000.0, INDIGO_MID, (180, 0.6)),
        ("16 x 4", 1135.91, 104786.5 / 1000.0, TEAL, (200, 0.2)),
        ("8 x 8", 671.69, 102751.2 / 1000.0, INDIGO, (-240, -2.5)),
    ]
    ax3.plot([p[1] for p in ga_pts], [p[2] for p in ga_pts], color=SLATE_LIGHT, lw=1.3, ls="--", zorder=2)
    for lbl, vr, tp, col, (dx, dy) in ga_pts:
        ax3.scatter([vr], [tp], color=col, s=95 if lbl == "16 x 4" else 65, zorder=4, edgecolors="#FFFFFF", linewidths=1.2)
        ax3.annotate(
            f"{lbl}\n({vr:.0f} MB, {tp:.1f}k)",
            xy=(vr, tp),
            xytext=(vr + dx, tp + dy),
            fontsize=7.8,
            fontweight="bold" if lbl == "16 x 4" else "normal",
            color=col,
        )
    ax3.set_xlim(220, 4280)
    ax3.set_ylim(92.5, 107.8)
    ax3.set_xlabel("峰值显存分配 Peak Alloc (MB)")
    ax3.set_ylabel("训练吞吐 (k tokens/s)", labelpad=6)
    ax3.grid(True, color="#E2E8F0", lw=0.6)
    _panel_tag(ax3, "c", "等效全局批 64 的梯度累积权衡", "GB10 实测：16x4 兼顾显存占用与训练吞吐")

    _save_fig(fig, out_dir, "pretrain-muon-convergence-and-scaling")


# ---------------------------------------------------------------------------
# 图 4：双轨评测机制清晰对照（基座续写似然测知识 vs 对话精确匹配测格式）
# ---------------------------------------------------------------------------
def plot_eval_stage_separation(out_dir: Path) -> None:
    fig = plt.figure(figsize=(14.4, 5.8), facecolor="#FFFFFF")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.16)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    for ax in (ax1, ax2):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    # ---- Panel a: 轨道 A 预训练基座评测 (val_bpb + CORE) ----
    _panel_tag(
        ax1,
        "a",
        "轨道 A：预训练基座无格式评测（nanochat base_eval）",
        "不依赖对话模板与停止符，直接评估连续压损与条件续写似然",
        y_title=1.05,
        y_sub=0.995,
    )

    # 上半区：val_bpb 连续窗 + LaTeX 公式
    ax1.add_patch(FancyBboxPatch((0.02, 0.56), 0.96, 0.36, boxstyle="round,pad=0,rounding_size=0.02", facecolor=TEAL_BG, edgecolor=TEAL_EDGE, lw=1.0))
    ax1.text(0.05, 0.875, "1) 固定留出流连续指标 val_bpb（比较需对齐原文）", fontsize=9.0, fontweight="bold", color=TEAL)
    for wi in range(4):
        wx = 0.05 + wi * 0.225
        ax1.add_patch(Rectangle((wx, 0.72), 0.195, 0.095, facecolor="#FFFFFF", edgecolor=TEAL, lw=1.0, zorder=3))
        ax1.text(wx + 0.0975, 0.7675, f"Window {wi}", ha="center", va="center", fontsize=8.2, color=TEAL, fontweight="bold", zorder=4)
        if wi < 3:
            _arrow(ax1, (wx + 0.197, 0.7675), (wx + 0.223, 0.7675), color=TEAL, lw=1.1, ms=8.5)
    ax1.text(0.05, 0.625, r"计算口径：$\mathrm{BPB} = \frac{\sum \mathcal{L}_{\mathrm{nats}} / \ln 2}{\sum \mathrm{token\_bytes}}$", fontsize=9.0, color=SLATE_DARK, va="center")
    ax1.text(0.57, 0.625, r"GB10 实测: $3.70 \to 2.60 \to 2.30$", fontsize=8.4, fontweight="bold", color=TEAL, va="center")

    # 下半区：CORE 条件续写条形对比 + LaTeX 居中得分公式
    ax1.add_patch(FancyBboxPatch((0.02, 0.06), 0.96, 0.44, boxstyle="round,pad=0,rounding_size=0.02", facecolor=SLATE_BG, edgecolor=SLATE_BORDER, lw=1.0))
    ax1.text(0.05, 0.455, "2) CORE 续写似然对比（不生成对话，只比整句概率高低）", fontsize=9.0, fontweight="bold", color=SLATE_DARK)
    _pill(ax1, 0.05, 0.26, 0.34, 0.115, "统一前缀 (Prefix)", "#FFFFFF", SLATE_BORDER, SLATE_DARK, 8.2, True, sub="水加热到100度通常会...")
    cands = [
        ("A. 沸腾成水蒸气（高似然胜出）", 0.36, TEAL, True),
        ("B. 立刻结成固体冰块", 0.13, SLATE_LIGHT, False),
        ("C. 全部变成金属单质", 0.07, SLATE_LIGHT, False),
    ]
    for ci, (clbl, blen, bcol, win) in enumerate(cands):
        cy = 0.365 - ci * 0.088
        ax1.text(0.43, cy, clbl, fontsize=8.0, color=TEAL if win else SLATE, fontweight="bold" if win else "normal", va="center")
        ax1.add_patch(Rectangle((0.43, cy - 0.036), blen, 0.020, facecolor=bcol, zorder=3))
        _arrow(ax1, (0.39, 0.317), (0.425, cy - 0.01), color=TEAL if win else SLATE_BORDER, lw=1.0, ms=8)

    ax1.text(
        0.05,
        0.105,
        r"居中归一化 (DCLM 22 项)：$\mathrm{Score}_k = \frac{\mathrm{Acc}_k - b_k}{1 - b_k}$（随机猜均归零）",
        fontsize=8.6,
        fontweight="bold",
        color=TEAL,
        va="center",
    )

    # ---- Panel b: 同题对比为何精确匹配把 pretrain 判成 0% ----
    _panel_tag(
        ax2,
        "b",
        "轨道 B：对话精确匹配与阶段错位示例",
        "08_eval_report.py 若把 SFT/RL 对话精确匹配套给基座，将产生虚假全零",
        y_title=1.05,
        y_sub=0.995,
    )

    # 共享题干胶囊
    _pill(
        ax2,
        0.06,
        0.77,
        0.88,
        0.135,
        r"评测输入：User: 8 minus 5?\nAssistant:",
        INDIGO_BG,
        INDIGO_EDGE,
        INDIGO,
        9.0,
        True,
        sub="判定规则：要求精确输出答案并立即生成 <EOS> 终止",
    )

    # 左分支：未经后训练的 pretrain.pt 行为
    _arrow(ax2, (0.30, 0.77), (0.25, 0.65), color=BURGUNDY, lw=1.5, ms=11)
    ax2.add_patch(FancyBboxPatch((0.02, 0.14), 0.46, 0.50, boxstyle="round,pad=0,rounding_size=0.02", facecolor=BURGUNDY_BG, edgecolor=BURGUNDY_EDGE, lw=1.1))
    ax2.text(0.25, 0.59, "pretrain.pt（未学习对话停止）", ha="center", fontsize=8.9, fontweight="bold", color=BURGUNDY)
    _pill(ax2, 0.05, 0.38, 0.40, 0.15, "实测输出（按训练语料继续续写）", "#FFFFFF", BURGUNDY_EDGE, SLATE_DARK, 8.0, True, sub="...Yes, and 9 minus 4? User: Next...")
    ax2.text(0.25, 0.30, "Exact-Match = 0%   聊天 = 0%", ha="center", fontsize=8.8, fontweight="bold", color=BURGUNDY)
    ax2.text(0.25, 0.21, "未生成 <EOS> 终止符导致判错\n（同模型 val_bpb 实际已下降）", ha="center", va="center", fontsize=8.0, color=SLATE_DARK, linespacing=1.35)

    # 右分支：经过 midtrain + SFT + RL 后训练的模型行为
    _arrow(ax2, (0.70, 0.77), (0.75, 0.65), color=INDIGO, lw=1.5, ms=11)
    ax2.add_patch(FancyBboxPatch((0.52, 0.14), 0.46, 0.50, boxstyle="round,pad=0,rounding_size=0.02", facecolor=INDIGO_BG, edgecolor=INDIGO_EDGE, lw=1.1))
    ax2.text(0.75, 0.59, "sft.pt / rl.pt（已完成对话对齐）", ha="center", fontsize=8.9, fontweight="bold", color=INDIGO)
    _pill(ax2, 0.55, 0.38, 0.40, 0.15, "对齐输出（按格式生成并终止）", "#FFFFFF", INDIGO_EDGE, INDIGO, 8.2, True, sub="输出：3 <EOS>（截断正确）")
    ax2.text(0.75, 0.30, "本地 2.07M：聊天 25% | 算术 26%", ha="center", fontsize=8.6, fontweight="bold", color=INDIGO)
    ax2.text(0.75, 0.21, "评测职责：检验知识召回、\n指令遵循与推理精确度", ha="center", va="center", fontsize=8.0, color=SLATE_DARK, linespacing=1.35)

    # 底部极简一行分界条
    ax2.text(0.50, 0.055, "工程结论：基座期仅看 val_bpb 与续写似然；精确匹配从 midtrain/SFT 起纳入对比", ha="center", va="center", fontsize=8.3, fontweight="bold", color=SLATE_DARK)

    _save_fig(fig, out_dir, "pretrain-eval-stage-separation")


# ---------------------------------------------------------------------------
# 图 5：连续指标平滑下降 vs 离散 Exact-Match 虚假零分平台（LaTeX 公式版）
# ---------------------------------------------------------------------------
def plot_continuous_vs_discrete_metrics(out_dir: Path, results_dir: Path) -> None:
    ss = json.loads((results_dir / "scaling" / "scaling_summary.json").read_text(encoding="utf-8"))
    tiers = ss["modes"]["single"]

    fig = plt.figure(figsize=(14.8, 5.4), facecolor="#FFFFFF")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.30)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    logN = np.linspace(6.0, 11.5, 300)
    N = 10 ** logN
    c, alpha = 1e6, -0.22
    ce = (N / c) ** alpha + 0.06
    p_tok = np.exp(-ce)

    acc_L4 = p_tok ** 4
    acc_L10 = p_tok ** 10

    ax1.plot(logN, p_tok, color=TEAL, lw=2.3, label=r"单 Token 概率 $p = e^{-\mathcal{L}_{\mathrm{CE}}}$ [连续]")
    ax1.plot(logN, acc_L4, color=INDIGO, lw=1.9, ls="--", label=r"4-Token Exact-Match $p^4$ [非线性]")
    ax1.plot(logN, acc_L10, color=BURGUNDY, lw=2.1, ls="-.", label=r"10-Token Exact-Match $p^{10}$ [零分平台]")

    ax1.axhspan(0.0, 0.05, color=BURGUNDY_BG, alpha=0.8)
    ax1.annotate(
        "Exact-Match 近零平台区\n（单 token 概率仍在稳步上升）",
        xy=(7.4, 0.015),
        xytext=(6.2, 0.22),
        fontsize=8.0,
        fontweight="bold",
        color=BURGUNDY,
        arrowprops=dict(arrowstyle="->", color=BURGUNDY, lw=1.0),
    )
    ax1.annotate(
        "平滑改善一直存在",
        xy=(8.1, float(p_tok[np.argmin(np.abs(logN - 8.1))])),
        xytext=(6.25, 0.70),
        fontsize=8.2,
        fontweight="bold",
        color=TEAL,
        arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.1),
    )

    ax1.set_xlim(6.0, 11.5)
    ax1.set_ylim(-0.02, 1.03)
    ax1.set_xlabel(r"模型参数规模 $\log_{10}(N)$")
    ax1.set_ylabel("得分 / 概率 (0 ~ 1)")
    ax1.legend(loc="upper left", fontsize=8.0)
    ax1.grid(True, color="#E2E8F0", lw=0.6)
    _panel_tag(ax1, "a", "理论机制：连续概率与离散 Exact-Match 刻度差异", "复现 Schaeffer et al. (2023, arXiv:2304.15004) 解析模型")

    labels = ["t1_micro\n26.4K tokens", "t2_small\n506K tokens", "t3_mid\n5.07M tokens"]
    x = np.arange(len(labels))
    pre_bpb = [t["pretrain"]["bpb"] for t in tiers]
    sft_bpb = [t["sft"]["test_bpb"] for t in tiers]
    sft_hit = [t["sft"]["val_answer_hit_rate"] * 100.0 for t in tiers]

    ax2.plot(x, pre_bpb, color=TEAL, marker="o", ms=7.0, lw=2.2, label="Pretrain val_bpb（连续）")
    ax2.plot(x, sft_bpb, color=INDIGO, marker="s", ms=6.5, lw=2.0, ls="--", label="SFT test_bpb（连续）")
    for xi, pb, sb in zip(x, pre_bpb, sft_bpb):
        ax2.text(xi, pb + 0.13, f"{pb:.3f}", ha="center", fontsize=8.2, fontweight="bold", color=TEAL)
        ax2.text(xi, sb - 0.27, f"{sb:.3f}", ha="center", fontsize=8.2, fontweight="bold", color=INDIGO)

    ax2_r = ax2.twinx()
    ax2_r.spines["top"].set_visible(False)
    ax2_r.bar(x, [max(v, 0.8) for v in sft_hit], width=0.28, color=BURGUNDY_EDGE, edgecolor=BURGUNDY, lw=1.0, alpha=0.85, label="下游问答离散命中率 (%) [实测均 0.0%]")
    for xi in x:
        ax2_r.text(xi, 3.3, "0.0%", ha="center", fontsize=8.2, fontweight="bold", color=BURGUNDY)

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8.4)
    ax2.set_ylim(0.5, 4.35)
    ax2_r.set_ylim(0, 50)
    ax2.set_ylabel("留出验证集 BPB (bits/byte，连续下降)", color=TEAL)
    ax2_r.set_ylabel("下游问答命中率 (%)", color=BURGUNDY)
    ax2.grid(axis="y", color="#E2E8F0", lw=0.6)

    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2_r.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=7.8)
    _panel_tag(ax2, "b", "GB10 旧口径 BPB 与离散命中率记录", "数据来源：scaling_summary.json；旧 BPB 待重评估")

    _save_fig(fig, out_dir, "pretrain-continuous-vs-discrete-metrics")


def main() -> int:
    ap = argparse.ArgumentParser(description="生成预训练原理与评测章节科研配图")
    ap.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "docs" / "zh" / "inference" / "figures",
    )
    ap.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="指标来源目录（缺省 $XL_OUT_DIR）",
    )
    args = ap.parse_args()

    results_dir = args.results_dir or lc.out_dir()
    plot_causal_architecture(args.outdir)
    plot_muon_orthogonalization(args.outdir)
    plot_muon_convergence_and_scaling(args.outdir, results_dir)
    plot_eval_stage_separation(args.outdir)
    plot_continuous_vs_discrete_metrics(args.outdir, results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
