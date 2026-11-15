#!/usr/bin/env python3
"""生成分词器章节的五组顶刊风格科研配图（PNG + 矢量 SVG）。

设计门禁：
1. 一图讲清一件事：编解码链路、四类范式、词表权衡、CPU 训练流水线各自成图，不强行硬拼；
2. 数学公式一律使用 Matplotlib LaTeX (`$...$`) mathtext 渲染；
3. 可选使用 `matplotlib_layout_audit.audit_figure` 检查像素碰撞与容器溢出；未配置时跳过。
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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
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
        "font.monospace": ["Consolas", "Microsoft YaHei", "Noto Sans SC", "DejaVu Sans Mono"],
        "axes.unicode_minus": False,
        "font.size": 10.0,
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF",
    }
)

# ---------------------------------------------------------------------------
# Nature / ICML 矿物中性色板
# ---------------------------------------------------------------------------
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


def _arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = SLATE,
    lw: float = 1.25,
    ls: str = "-",
    rad: float = 0.0,
    ms: float = 11,
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
            zorder=5,
        )
    )


def _draw_iso_cuboid(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    dx: float,
    dy: float,
    front_fc: str,
    top_fc: str,
    side_fc: str,
    ec: str,
    lw: float = 1.0,
    zorder: int = 3,
) -> None:
    """绘制 3D 等距长方体张量/矩阵块。"""
    front = Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)], closed=True, facecolor=front_fc, edgecolor=ec, lw=lw, zorder=zorder)
    top = Polygon([(x, y + h), (x + dx, y + h + dy), (x + w + dx, y + h + dy), (x + w, y + h)], closed=True, facecolor=top_fc, edgecolor=ec, lw=lw, zorder=zorder)
    side = Polygon([(x + w, y), (x + w + dx, y + dy), (x + w + dx, y + h + dy), (x + w, y + h)], closed=True, facecolor=side_fc, edgecolor=ec, lw=lw, zorder=zorder)
    ax.add_patch(front)
    ax.add_patch(top)
    ax.add_patch(side)


def _draw_doc_sheet(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    lines: list[str],
    ec: str = SLATE_BORDER,
    fontsize: float = 8.0,
    zorder_base: int = 2,
) -> None:
    """带折角的纸页实体，表现原始语料与导出文档。"""
    fold = min(w, h) * 0.22
    pts = [(x, y), (x + w, y), (x + w, y + h - fold), (x + w - fold, y + h), (x, y + h)]
    ax.add_patch(Polygon(pts, closed=True, facecolor="#FFFFFF", edgecolor=ec, lw=1.1, zorder=zorder_base))
    ax.add_patch(
        Polygon(
            [(x + w - fold, y + h), (x + w - fold, y + h - fold), (x + w, y + h - fold)],
            closed=True,
            facecolor=SLATE_BG,
            edgecolor=ec,
            lw=0.9,
            zorder=zorder_base + 1,
        )
    )
    for idx, line in enumerate(lines):
        has_cjk = any("\u4e00" <= c <= "\u9fff" or "\u3000" <= c <= "\u303f" or "\uff00" <= c <= "\uffef" for c in line)
        ax.text(
            x + w * 0.08,
            y + h * (0.68 - idx * 0.27),
            line,
            fontsize=fontsize,
            color=SLATE_DARK,
            family="sans-serif" if has_cjk else "monospace",
            va="center",
            ha="left",
            zorder=zorder_base + 2,
        )


def _draw_token_chip(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    fc: str,
    ec: str,
    tc: str = SLATE_DARK,
    fontsize: float = 8.6,
    mono: bool = True,
    bold: bool = False,
) -> None:
    """绘制切分后的紧凑词元拼块。"""
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=0.45",
            facecolor=fc,
            edgecolor=ec,
            linewidth=0.95,
            zorder=3,
        )
    )
    has_cjk = any("\u4e00" <= c <= "\u9fff" or "\u3000" <= c <= "\u303f" or "\uff00" <= c <= "\uffef" for c in text)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=tc,
        family="monospace" if (mono and not has_cjk) else "sans-serif",
        fontweight="bold" if bold else "normal",
        zorder=4,
    )


def _save_both(fig: plt.Figure, out_dir: Path, stem: str, dpi: int = 240) -> None:
    issues = audit_figure(fig, name=stem)
    if issues:
        for item in issues:
            print("AUDIT:", item)
        raise RuntimeError(f"Layout audit failed for {stem}: {len(issues)} issue(s)")
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{stem}.png"
    svg_path = out_dir / f"{stem}.svg"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="#FFFFFF")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="#FFFFFF")
    plt.close(fig)
    status = "audited" if AUDIT_AVAILABLE else "layout audit skipped"
    print(f"[OK] Saved {png_path.name} & {svg_path.name} ({status})")


# ---------------------------------------------------------------------------
# 图 1：文本编解码物理链路（单张专精图：UTF-8 字节 -> 切分 -> 嵌入查表 -> 采样 -> 流式拼接）
# ---------------------------------------------------------------------------
def plot_byte_codec_flow(out_dir: Path) -> None:
    fig = plt.figure(figsize=(13.6, 5.0), facecolor="#FFFFFF")
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.text(1.0, 96.5, "文本编解码物理链路：字符流、UTF-8 字节单元、嵌入查表与流式字节重组", fontsize=12.0, fontweight="bold", color=SLATE_DARK, va="top")
    ax.plot([1.0, 99.0], [89.5, 89.5], color=SLATE_BORDER, lw=0.8)

    # Stage 1
    ax.text(10.5, 84.0, "1. 原始字符串与 UTF-8 字节", ha="center", va="center", fontsize=9.3, fontweight="bold", color=SLATE_DARK)
    _draw_doc_sheet(ax, 2.2, 55.0, 16.2, 22.0, ["I'm AI 你好", "UTF-8: 13 Bytes"], ec=SLATE_LIGHT, fontsize=8.2)
    byte_tiles = [
        ("49", SLATE_BG, SLATE_BORDER, 2.0),
        ("27 6D", SLATE_BG, SLATE_BORDER, 3.0),
        ("20", SLATE_BG, SLATE_BORDER, 2.0),
        ("41 49", SLATE_BG, SLATE_BORDER, 3.0),
        ("20", SLATE_BG, SLATE_BORDER, 2.0),
        ("E4 BD A0", INDIGO_BG, INDIGO_EDGE, 4.8),
        ("E5 A5 BD", INDIGO_BG, INDIGO_EDGE, 4.8),
    ]
    bx = 0.5
    for hx, fc, ec, bw in byte_tiles:
        _draw_token_chip(ax, bx, 43.0, bw, 7.6, hx, fc=fc, ec=ec, tc=INDIGO if ec == INDIGO_EDGE else SLATE, fontsize=6.4)
        bx += bw + 0.22
    ax.text(11.0, 36.5, "汉字固定占 3 个原生字节", ha="center", va="center", fontsize=8.1, color=INDIGO)

    # Stage 2
    ax.text(34.5, 84.0, "2. 正则预切分与字节映射", ha="center", va="center", fontsize=9.3, fontweight="bold", color=INDIGO)
    chunks = [
        ("I", 2.6, INDIGO_BG, INDIGO_EDGE, INDIGO),
        ("'m", 3.3, INDIGO_BG, INDIGO_EDGE, INDIGO),
        ("ĠAI", 4.1, INDIGO_BG, INDIGO_EDGE, INDIGO),
        ("Ġ", 2.6, TEAL_BG, TEAL_EDGE, TEAL),
        ("E4 BD A0", 6.2, BURGUNDY_BG, BURGUNDY_EDGE, BURGUNDY),
    ]
    cx = 22.8
    for idx_c, (lbl, cw, fc, ec, tc) in enumerate(chunks):
        _draw_token_chip(ax, cx, 59.5, cw, 11.5, lbl, fc=fc, ec=ec, tc=tc, fontsize=8.2, bold=True)
        if idx_c < len(chunks) - 1:
            ax.plot([cx + cw + 0.35, cx + cw + 0.35], [56.5, 74.0], color=SLATE_LIGHT, lw=0.9, ls="--", zorder=2)
        cx += cw + 0.72
    ax.text(34.5, 49.0, "正则切分阻止词与标点合并\n未登录汉字拆回单字节或短片段", ha="center", va="center", fontsize=8.1, color=SLATE, linespacing=1.35)

    # Stage 3
    ax.text(58.5, 84.0, "3. 词表查表与嵌入抽取", ha="center", va="center", fontsize=9.3, fontweight="bold", color=SLATE_DARK)
    ids = [("40", 3.4), ("1101", 4.6), ("15592", 5.4), ("220", 3.8), ("163", 3.8)]
    ix = 47.0
    for idx_i, (tid, tw) in enumerate(ids):
        _draw_token_chip(ax, ix, 63.5, tw, 8.8, tid, fc="#FFFFFF", ec=SLATE_DARK if idx_i < 4 else BURGUNDY, tc=SLATE_DARK if idx_i < 4 else BURGUNDY, fontsize=8.3, bold=True)
        ix += tw + 0.48
    ax.text(58.5, 57.5, "Token ID 序列（示例索引）", ha="center", va="center", fontsize=7.8, color=SLATE)
    _draw_iso_cuboid(ax, 51.0, 38.0, 14.5, 11.5, 2.2, 2.8, INDIGO_BG, "#DBEAFE", "#BFDBFE", INDIGO, lw=1.0)
    ax.text(58.2, 43.7, r"$E \in \mathbb{R}^{V \times d}$", ha="center", va="center", fontsize=9.0, color=INDIGO, fontweight="bold")

    # Stage 4: Causal attention grid icon + softmax logits bar
    ax.text(85.0, 84.0, "4. 因果自回归与采样", ha="center", va="center", fontsize=9.3, fontweight="bold", color=SLATE_DARK)
    # Draw small 4x4 causal triangular attention matrix instead of generic boxes
    gx, gy, csize = 73.5, 42.5, 3.0
    for r in range(4):
        for c in range(4):
            ok = c <= r
            ax.add_patch(Rectangle((gx + c * csize, gy + (3 - r) * csize), csize - 0.3, csize - 0.3, facecolor=INDIGO_BG if ok else SLATE_BG, edgecolor=INDIGO_EDGE if ok else SLATE_BORDER, lw=0.7, zorder=3))
    ax.text(79.5, 38.2, r"因果注意力 $L$ 层", ha="center", va="center", fontsize=8.0, color=INDIGO, fontweight="bold")

    px0, py0, pw, ph = 88.0, 42.5, 10.5, 22.0
    ax.plot([px0, px0 + pw], [py0, py0], color=SLATE_DARK, lw=1.0)
    probs = [0.14, 0.26, 0.78, 0.18, 0.08]
    for bi, pr in enumerate(probs):
        bc = TEAL if bi == 2 else SLATE_BORDER
        ax.add_patch(Rectangle((px0 + 0.8 + bi * 1.9, py0), 1.3, pr * ph, facecolor=bc, edgecolor="none", zorder=3))
    ax.text(px0 + 0.8 + 2 * 1.9 + 0.65, py0 + 0.78 * ph + 2.0, "next ID", ha="center", va="bottom", fontsize=7.8, color=TEAL, fontweight="bold")
    ax.text(93.2, 38.2, "Softmax 采样", ha="center", va="center", fontsize=7.9, color=SLATE)

    # Stage 5: Streaming UTF-8 buffer return path
    ax.plot([93.2, 93.2], [33.5, 16.5], color=TEAL, lw=1.3)
    _arrow(ax, (93.2, 16.5), (69.0, 16.5), color=TEAL, lw=1.4)
    ax.text(81.2, 20.5, "流式逐个返回字节片段", ha="center", va="bottom", fontsize=8.1, color=TEAL)

    ax.add_patch(FancyBboxPatch((33.5, 6.5), 35.0, 20.0, boxstyle="round,pad=0,rounding_size=0.8", facecolor=TEAL_BG, edgecolor=TEAL, lw=1.0, zorder=2))
    ax.text(51.0, 22.6, "5. 流式逆映射缓冲区（UTF-8 凑齐三字节后输出）", ha="center", va="center", fontsize=8.5, color=TEAL, fontweight="bold")
    _draw_token_chip(ax, 35.5, 9.5, 6.0, 8.2, "[E4]", fc="#FFFFFF", ec=BURGUNDY_EDGE, tc=BURGUNDY, fontsize=8.2, bold=True)
    ax.text(42.5, 13.6, "+", ha="center", va="center", fontsize=10.5, color=TEAL, fontweight="bold")
    _draw_token_chip(ax, 43.7, 9.5, 9.0, 8.2, "[BD A0]", fc="#FFFFFF", ec=TEAL_EDGE, tc=TEAL, fontsize=8.2, bold=True)
    _arrow(ax, (53.4, 13.6), (57.2, 13.6), color=TEAL, lw=1.2, ms=9.5)
    _draw_token_chip(ax, 57.8, 9.0, 9.2, 9.2, "\"你\" (完整)", fc="#FFFFFF", ec=TEAL, tc=SLATE_DARK, fontsize=8.6, mono=False, bold=True)

    _arrow(ax, (19.0, 65.0), (22.2, 65.0), color=SLATE_DARK)
    _arrow(ax, (44.9, 65.8), (46.8, 67.5), color=INDIGO)
    _arrow(ax, (68.0, 46.5), (73.0, 48.5), color=SLATE_DARK)
    _arrow(ax, (85.6, 51.0), (87.8, 51.0), color=TEAL)

    fig.text(0.015, 0.01, "注：以示例字符串展示数据形态演化（非特定词表真实编号），解码器以 4x4 因果三角注意力网格示意。", fontsize=8.3, color=SLATE)
    _save_both(fig, out_dir, "tokenizer-byte-codec-flow")


# ---------------------------------------------------------------------------
# 图 2：四类词表构建范式对比（含规范 LaTeX 公式 + 合并树/词格/熵曲线具象图示）
# ---------------------------------------------------------------------------
def plot_paradigms_comparison(out_dir: Path) -> None:
    fig = plt.figure(figsize=(14.2, 5.4), facecolor="#FFFFFF")
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.text(1.0, 96.5, "四类分词算法对比：BPE、WordPiece、Unigram LM 与无词表方案", fontsize=12.0, fontweight="bold", color=SLATE_DARK, va="top")
    ax.plot([1.0, 99.0], [89.5, 89.5], color=SLATE_BORDER, lw=0.8)

    col_bounds = [(1.0, 24.5), (25.8, 49.3), (50.6, 74.1), (75.4, 99.0)]
    for _, x1 in col_bounds[:-1]:
        ax.plot([x1 + 0.65, x1 + 0.65], [5.0, 86.5], color=SLATE_BORDER, lw=0.8, ls=":")

    # 列 1: BBPE 合并树
    ax.text(12.8, 84.0, "BBPE（字节级 BPE）", ha="center", fontsize=10.0, fontweight="bold", color=INDIGO)
    ax.text(12.8, 78.2, "从 256 基础字节自底向上贪心合并", ha="center", fontsize=8.1, color=SLATE)
    ax.text(12.8, 66.5, "合并准则：高频共现相邻对优先\n无 <unk>，覆盖 GPT 与开源主流", ha="center", va="center", fontsize=8.1, color=SLATE_DARK, linespacing=1.4)
    b_chars = ["l", "o", "w", "e", "r"]
    bx0, step = 4.3, 3.6
    for i, ch in enumerate(b_chars):
        _draw_token_chip(ax, bx0 + i * step, 9.5, 2.7, 6.8, ch, fc=SLATE_BG, ec=SLATE_BORDER, fontsize=8.1)
    ax.plot([bx0 + 1.35, bx0 + 1.35, bx0 + step + 1.35, bx0 + step + 1.35], [16.8, 20.5, 20.5, 16.8], color=INDIGO_MID, lw=1.1)
    _draw_token_chip(ax, bx0 + 1.35, 22.3, 3.6, 6.8, "lo", fc=INDIGO_BG, ec=INDIGO_EDGE, tc=INDIGO, fontsize=8.2, bold=True)
    ax.plot([bx0 + 3 * step + 1.35, bx0 + 3 * step + 1.35, bx0 + 4 * step + 1.35, bx0 + 4 * step + 1.35], [16.8, 20.5, 20.5, 16.8], color=INDIGO_MID, lw=1.1)
    _draw_token_chip(ax, bx0 + 3 * step + 1.35, 22.3, 3.6, 6.8, "er", fc=INDIGO_BG, ec=INDIGO_EDGE, tc=INDIGO, fontsize=8.2, bold=True)
    ax.plot([bx0 + 3.15, bx0 + 3.15, bx0 + 2 * step + 1.35, bx0 + 2 * step + 1.35], [29.6, 33.5, 33.5, 16.8], color=INDIGO, lw=1.15)
    _draw_token_chip(ax, bx0 + 3.5, 35.3, 4.6, 7.2, "low", fc=INDIGO_BG, ec=INDIGO, tc=INDIGO, fontsize=8.4, bold=True)
    ax.plot([bx0 + 5.8, bx0 + 5.8, bx0 + 3 * step + 3.15, bx0 + 3 * step + 3.15], [43.0, 47.0, 47.0, 29.6], color=INDIGO, lw=1.25)
    _draw_token_chip(ax, bx0 + 6.8, 49.0, 6.6, 7.6, "lower", fc="#DBEAFE", ec=INDIGO, tc=INDIGO, fontsize=8.6, bold=True)

    # 列 2: WordPiece 规范 LaTeX 公式
    ax.text(37.5, 84.0, "WordPiece（似然增益）", ha="center", fontsize=10.0, fontweight="bold", color=SLATE_DARK)
    ax.text(37.5, 78.2, "按语料互信息合并，推理走最长前缀", ha="center", fontsize=8.1, color=SLATE)
    ax.text(
        37.5,
        62.5,
        r"$\mathrm{score}(x,y) = \frac{\mathrm{freq}(xy)}{\mathrm{freq}(x)\cdot\mathrm{freq}(y)}$",
        ha="center",
        va="center",
        fontsize=9.6,
        color=SLATE_DARK,
    )
    _draw_token_chip(ax, 28.0, 43.5, 19.0, 7.5, "输入词：unaffordable", fc=SLATE_BG, ec=SLATE_BORDER, fontsize=8.2)
    _arrow(ax, (37.5, 42.8), (37.5, 36.2), color=SLATE)
    _draw_token_chip(ax, 27.6, 26.8, 4.2, 8.0, "un", fc=TEAL_BG, ec=TEAL, tc=TEAL, fontsize=8.3, bold=True)
    _draw_token_chip(ax, 32.3, 26.8, 8.2, 8.0, "##afford", fc=TEAL_BG, ec=TEAL, tc=TEAL, fontsize=8.1, bold=True)
    _draw_token_chip(ax, 41.0, 26.8, 6.8, 8.0, "##able", fc=TEAL_BG, ec=TEAL, tc=TEAL, fontsize=8.1, bold=True)
    ax.text(37.5, 15.0, "续接标记 ## 标明词内边界（BERT 代表）\n词表未覆盖字可能退回 [UNK]", ha="center", va="center", fontsize=8.0, color=SLATE, linespacing=1.35)

    # 列 3: Unigram LM 规范 LaTeX 公式 + Viterbi 词格
    ax.text(62.3, 84.0, "Unigram LM（自顶向下裁剪）", ha="center", fontsize=10.0, fontweight="bold", color=TEAL)
    ax.text(62.3, 78.2, "超大候选表按 EM 似然迭代剔除", ha="center", fontsize=8.1, color=SLATE)
    ax.text(
        62.3,
        62.5,
        r"$s^* = \arg\max_s \prod_i P(t_i)$",
        ha="center",
        va="center",
        fontsize=9.8,
        color=TEAL,
    )
    nodes_x = [53.0, 57.5, 62.0, 66.5, 71.0]
    for nx in nodes_x:
        ax.scatter([nx], [31.5], s=44, color=TEAL, zorder=5)
    _arrow(ax, (53.2, 32.3), (61.8, 32.3), color=TEAL, lw=1.45, rad=-0.35, ms=9)
    _arrow(ax, (62.2, 32.3), (70.8, 32.3), color=TEAL, lw=1.45, rad=-0.35, ms=9)
    ax.text(57.5, 42.2, "\"hel\"", fontsize=8.1, color=TEAL, fontweight="bold", ha="center")
    ax.text(66.5, 42.2, "\"lo\"", fontsize=8.1, color=TEAL, fontweight="bold", ha="center")
    _arrow(ax, (53.2, 30.7), (66.3, 30.7), color=BURGUNDY_EDGE, lw=1.0, ls="--", rad=0.32, ms=8)
    ax.text(59.8, 17.5, "\"hell\"（低似然贡献被剪除）", fontsize=7.8, color=BURGUNDY, ha="center")
    ax.text(62.3, 10.5, "SentencePiece / T5 常用\n支持按概率随机采样多种切分路径", ha="center", va="center", fontsize=8.0, color=SLATE, linespacing=1.35)

    # 列 4: 无静态词表方案
    ax.text(87.2, 84.0, "无静态词表方案", ha="center", fontsize=10.0, fontweight="bold", color=SLATE_DARK)
    ax.text(87.2, 78.2, "绕开固定 BPE 词表限制", ha="center", fontsize=8.1, color=SLATE)
    ax.text(77.0, 62.5, "ByT5：", fontsize=8.3, fontweight="bold", color=SLATE_DARK, va="center")
    for k in range(8):
        _draw_token_chip(ax, 82.8 + k * 1.9, 59.0, 1.65, 6.5, "B", fc=SLATE_BG, ec=SLATE_LIGHT, fontsize=7.2)

    ax.text(77.0, 43.5, "BLT：", fontsize=8.3, fontweight="bold", color=INDIGO, va="center")
    ex = np.linspace(83.0, 97.5, 60)
    ey = 40.0 + 3.0 * np.exp(-((ex - 87.5) / 0.9) ** 2) + 4.5 * np.exp(-((ex - 93.2) / 0.8) ** 2) + 0.6 * np.sin(ex * 2.5)
    ax.plot(ex, ey, color=INDIGO, lw=1.2)
    for px, pw in [(83.0, 4.3), (87.7, 5.2), (93.3, 4.2)]:
        ax.add_patch(Rectangle((px, 33.2), pw, 4.0, facecolor=INDIGO_BG, edgecolor=INDIGO_EDGE, lw=0.9))
    ax.text(90.2, 29.0, "按预测熵峰值动态聚合成 patch", ha="center", fontsize=7.5, color=SLATE)

    ax.text(77.0, 15.5, "PIXEL：", fontsize=8.3, fontweight="bold", color=TEAL, va="center")
    for gi in range(3):
        px = 84.0 + gi * 4.5
        ax.add_patch(Rectangle((px, 11.0), 3.8, 9.0, facecolor="#FFFFFF", edgecolor=TEAL, lw=0.9))
        for r in range(3):
            for c in range(2):
                ax.add_patch(Rectangle((px + 0.4 + c * 1.5, 12.0 + r * 2.5), 1.2, 1.8, facecolor=TEAL_BG if (r + c + gi) % 2 == 0 else SLATE_BG, edgecolor="none"))
    ax.text(90.5, 6.5, "文本渲染为像素图像块送入 ViT", ha="center", fontsize=7.5, color=SLATE)

    fig.text(0.015, 0.005, "注：四类词表构建与后子词机制依 Cognetta et al. (2026) 现代分词综述整理。", fontsize=8.3, color=SLATE)
    _save_both(fig, out_dir, "tokenizer-paradigms-comparison")


# ---------------------------------------------------------------------------
# 图 3：词表规模的三重物理权衡（独立成图，宽松间距 + LaTeX 公式）
# ---------------------------------------------------------------------------
def plot_vocab_tradeoffs(out_dir: Path) -> None:
    fig = plt.figure(figsize=(13.8, 5.1), facecolor="#FFFFFF")
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.text(1.0, 96.5, r"词表规模 $V$ 对参数分布与上下文开销的影响（$d_{\mathrm{model}}=320$，6 层骨干实测）", fontsize=11.6, fontweight="bold", color=SLATE_DARK, va="top")
    ax.plot([1.0, 99.0], [89.5, 89.5], color=SLATE_BORDER, lw=0.8)

    # 左半区：首尾矩阵尺度对比
    ax.text(24.5, 83.5, r"① 首尾矩阵尺度：$\mathrm{tok\_emb} + \mathrm{lm\_head} = 2 \cdot V \cdot d_{\mathrm{model}}$", ha="center", fontsize=9.6, fontweight="bold", color=SLATE_DARK)
    ax.text(13.0, 76.5, r"紧凑词表 $V=2,048$（占比 15.1%）", ha="center", fontsize=8.5, fontweight="bold", color=INDIGO)
    _draw_iso_cuboid(ax, 5.5, 53.5, 15.0, 2.8, 1.5, 1.8, INDIGO_BG, "#DBEAFE", "#BFDBFE", INDIGO, lw=1.0)
    ax.text(13.0, 61.5, "lm_head (0.66M)", ha="center", va="center", fontsize=8.0, color=INDIGO, fontweight="bold")
    _draw_iso_cuboid(ax, 5.5, 32.5, 15.0, 17.0, 1.5, 1.8, SLATE_BG, "#F1F5F9", "#E2E8F0", SLATE, lw=1.1)
    ax.text(13.0, 41.0, "6 层 Transformer 骨干\n7.37M 参数主导计算", ha="center", va="center", fontsize=8.0, color=SLATE_DARK, linespacing=1.35)
    _draw_iso_cuboid(ax, 5.5, 25.5, 15.0, 2.8, 1.5, 1.8, INDIGO_BG, "#DBEAFE", "#BFDBFE", INDIGO, lw=1.0)
    ax.text(13.0, 20.5, "tok_emb (0.66M)", ha="center", va="center", fontsize=8.0, color=INDIGO, fontweight="bold")
    ax.text(13.0, 11.5, "总参数 8.68M | 验证利用率 88.1%", ha="center", fontsize=8.1, color=INDIGO)

    ax.text(37.5, 76.5, r"大词表 $V=32,768$（占比 74.0%）", ha="center", fontsize=8.5, fontweight="bold", color=BURGUNDY)
    _draw_iso_cuboid(ax, 30.0, 53.5, 15.0, 18.0, 1.5, 1.8, BURGUNDY_BG, "#FEE2E2", "#FECACA", BURGUNDY, lw=1.1)
    ax.text(37.5, 62.5, "lm_head\n10.49M 参数\n与 logits 峰值", ha="center", va="center", fontsize=7.9, color=BURGUNDY, fontweight="bold", linespacing=1.25)
    _draw_iso_cuboid(ax, 30.0, 32.5, 15.0, 17.0, 1.5, 1.8, SLATE_BG, "#F1F5F9", "#E2E8F0", SLATE, lw=1.0)
    ax.text(37.5, 41.0, "相同 7.37M 骨干", ha="center", va="center", fontsize=8.0, color=SLATE_DARK)
    _draw_iso_cuboid(ax, 30.0, 11.0, 15.0, 18.0, 1.5, 1.8, BURGUNDY_BG, "#FEE2E2", "#FECACA", BURGUNDY, lw=1.1)
    ax.text(37.5, 20.0, "tok_emb\n10.49M 参数\n含长尾欠训练词元", ha="center", va="center", fontsize=7.9, color=BURGUNDY, fontweight="bold", linespacing=1.25)
    ax.text(37.5, 4.5, "总参数 28.34M | 验证利用率仅 16.2%", ha="center", fontsize=8.1, color=BURGUNDY)

    ax.plot([50.2, 50.2], [5.0, 86.0], color=SLATE_BORDER, lw=0.9, ls=":")

    # 右半区：12 字节原文在不同压缩率下的上下文与注意力网格对比
    ax.text(75.0, 83.5, r"② 压缩率决定序列长度 $T$ 与注意力 $\mathcal{O}(T^2)$ 网格", ha="center", fontsize=9.6, fontweight="bold", color=SLATE_DARK)
    ax.text(53.0, 74.0, "匹配语料 (~3.8 B/token)：12 字节 -> 3 个 token（窗口余量充裕）", ha="left", fontsize=8.4, fontweight="bold", color=TEAL)
    for k in range(3):
        _draw_token_chip(ax, 53.5 + k * 5.2, 62.5, 4.8, 7.5, f"t{k+1}", fc=TEAL_BG, ec=TEAL, tc=TEAL, fontsize=8.3, bold=True)
    for k in range(3, 8):
        ax.add_patch(FancyBboxPatch((53.5 + k * 5.2, 62.5), 4.8, 7.5, boxstyle="round,pad=0,rounding_size=0.4", facecolor="#FFFFFF", edgecolor=SLATE_BORDER, ls="--", lw=0.8))
    ax.text(95.5, 66.2, "充裕", ha="left", va="center", fontsize=8.0, color=TEAL, fontweight="bold")

    gx, gy, cell = 56.0, 42.5, 4.2
    ax.add_patch(Rectangle((gx - 0.3, gy - 0.3), 3 * cell + 0.4, 3 * cell + 0.4, facecolor="none", edgecolor=SLATE_BORDER, lw=0.85, ls=":"))
    for r in range(3):
        for c in range(r + 1):
            ax.add_patch(Rectangle((gx + c * cell, gy + (2 - r) * cell), cell * 0.88, cell * 0.88, facecolor=TEAL_MID, alpha=0.55, edgecolor=TEAL, lw=0.7))
    ax.text(72.5, 48.5, r"$T=3 \Rightarrow 3\times 3$ 注意力三角" + "\n算力基线 ($1\\times$)", ha="left", va="center", fontsize=8.2, color=TEAL, linespacing=1.35)

    ax.text(53.0, 34.5, "失配语料 (~1.06 B/token)：同 12 字节膨胀成 11 个碎 token", ha="left", fontsize=8.4, fontweight="bold", color=BURGUNDY)
    for k in range(11):
        _draw_token_chip(ax, 53.5 + k * 3.85, 23.5, 3.55, 6.8, f"b{k+1}", fc=BURGUNDY_BG, ec=BURGUNDY_EDGE, tc=BURGUNDY, fontsize=7.0)

    gx2, gy2, cell2 = 54.5, 3.2, 1.55
    ax.add_patch(Rectangle((gx2 - 0.3, gy2 - 0.3), 11 * cell2 + 0.4, 11 * cell2 + 0.4, facecolor="none", edgecolor=SLATE_BORDER, lw=0.85, ls=":"))
    for r in range(11):
        for c in range(r + 1):
            ax.add_patch(Rectangle((gx2 + c * cell2, gy2 + (10 - r) * cell2), cell2 * 0.86, cell2 * 0.86, facecolor=BURGUNDY, alpha=0.45, edgecolor="none"))
    ax.text(74.5, 11.5, r"$T=11 \Rightarrow 11\times 11$ 注意力矩阵" + "\n" + r"序列拉长，注意力计算增至 $\sim 13.4\times$", ha="left", va="center", fontsize=8.2, color=BURGUNDY, fontweight="bold", linespacing=1.35)

    fig.text(0.015, 0.005, "注：左侧参数比依 untied weights 模型（8.68M vs 28.34M）实测计算；右侧以 12 字节文本作同口径注意力网格对比。", fontsize=8.3, color=SLATE)
    _save_both(fig, out_dir, "tokenizer-vocab-tradeoffs")


# ---------------------------------------------------------------------------
# 图 4：01_train_tokenizer.py 无 GPU 训练流水线（充裕纵向间隔，零文字碰框）
# ---------------------------------------------------------------------------
def plot_training_pipeline(out_dir: Path) -> None:
    fig = plt.figure(figsize=(13.8, 5.0), facecolor="#FFFFFF")
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.text(1.0, 96.5, "CPU 分词器训练流程（01_train_tokenizer.py）：流式读取、正则切分与输出文件", fontsize=11.8, fontweight="bold", color=SLATE_DARK, va="top")
    ax.plot([1.0, 99.0], [89.5, 89.5], color=SLATE_BORDER, lw=0.8)

    # Step 1
    ax.text(9.0, 82.5, "1. Parquet 流式读取", ha="center", fontsize=9.3, fontweight="bold", color=SLATE_DARK)
    _draw_doc_sheet(ax, 3.6, 42.5, 12.2, 27.5, [], ec=SLATE_BORDER, zorder_base=1)
    _draw_doc_sheet(ax, 2.0, 39.0, 12.2, 27.5, ["climbmix", "doc_cap=10k", "max=30M"], ec=SLATE, fontsize=7.9, zorder_base=3)
    ax.text(8.5, 25.0, "按 row group 流式读取\n单篇截断限制长文档占比", ha="center", fontsize=8.0, color=SLATE, linespacing=1.35)

    # Step 2: 充裕纵距彻底消除红字与芯片的任何碰撞
    ax.text(32.8, 82.5, "2. 正则预切分与 ByteLevel 基础表", ha="center", fontsize=9.3, fontweight="bold", color=INDIGO)
    ax.text(32.8, 74.0, "原始输入片段：\"I'm 2025!\"", ha="center", fontsize=8.4, color=SLATE_DARK)
    splits = [
        ("I", 2.8, INDIGO_BG, INDIGO_EDGE, INDIGO),
        ("'m", 3.6, INDIGO_BG, INDIGO_EDGE, INDIGO),
        ("Ġ202", 5.2, BURGUNDY_BG, BURGUNDY_EDGE, BURGUNDY),
        ("5", 2.8, BURGUNDY_BG, BURGUNDY_EDGE, BURGUNDY),
        ("!", 2.8, TEAL_BG, TEAL_EDGE, TEAL),
    ]
    sx = 22.0
    for idx_s, (sl, sw, fc, ec, tc) in enumerate(splits):
        _draw_token_chip(ax, sx, 56.0, sw, 10.0, sl, fc=fc, ec=ec, tc=tc, fontsize=8.2, bold=True)
        if idx_s < len(splits) - 1:
            ax.plot([sx + sw + 0.40, sx + sw + 0.40], [53.2, 68.5], color=SLATE_LIGHT, lw=0.95, ls="--")
        sx += sw + 0.80
    # 红色文字大幅下移至 y=39（与 y=56 芯片底缘间隔整整 17 个坐标系单位！）
    ax.text(32.8, 39.0, "红框标注 \\p{N}{1,3} 左起三位切分\n空格转为 Ġ（UTF-8 编码占 2 字节）", ha="center", va="center", fontsize=8.0, color=BURGUNDY, linespacing=1.4)
    ax.add_patch(Rectangle((21.6, 17.5), 22.4, 7.2, facecolor=SLATE_BG, edgecolor=SLATE_BORDER, lw=0.95))
    ax.text(32.8, 21.1, "0x00 .. 0xFF（256 字节无 <unk> 底表）", ha="center", va="center", fontsize=7.8, color=SLATE_DARK)

    # Step 3
    ax.text(58.8, 82.5, "3. BPE 合并规则序列（Merge List）", ha="center", fontsize=9.3, fontweight="bold", color=INDIGO)
    ax.add_patch(Rectangle((49.6, 22.0), 18.4, 52.0, facecolor="#FFFFFF", edgecolor=INDIGO_EDGE, lw=1.0))
    merges = [
        ("ID 0: <|endoftext|>", SLATE),
        ("ID 1: <|pad|>", SLATE),
        ("256: Ġ + t  -> Ġt", INDIGO),
        ("257: h + e  -> he", INDIGO),
        ("258: Ġt + he -> Ġthe", INDIGO),
        ("... (止于 vocab_size", SLATE),
        ("    或 min_frequency)", SLATE),
    ]
    for mi, (mt, mc) in enumerate(merges):
        has_cjk = any("\u4e00" <= c <= "\u9fff" or "\u3000" <= c <= "\u303f" or "\uff00" <= c <= "\uffef" for c in mt)
        ax.text(51.0, 68.0 - mi * 6.8, mt, fontsize=7.9, family="sans-serif" if has_cjk else "monospace", color=mc, va="center")

    # Step 4
    ax.text(86.2, 82.5, "4. 训练输出文件", ha="center", fontsize=9.3, fontweight="bold", color=SLATE_DARK)
    _draw_doc_sheet(ax, 74.8, 56.5, 23.4, 17.5, ["tokenizer.json", "固定词表 ID 与 merge 规则"], ec=INDIGO_EDGE, fontsize=7.8)
    _draw_doc_sheet(ax, 74.8, 35.5, 23.4, 17.5, ["tokenizer_info.json", "50 行往返校验: 0 fail"], ec=SLATE_BORDER, fontsize=7.8)
    ax.text(75.5, 28.5, "token_bytes.pt（一维字节长，ID 0/1 置 0）", ha="left", fontsize=7.9, fontweight="bold", color=TEAL)
    tb_vals = [("0", BURGUNDY_BG, BURGUNDY), ("0", BURGUNDY_BG, BURGUNDY), ("1", TEAL_BG, TEAL), ("1", TEAL_BG, TEAL), ("1", TEAL_BG, TEAL), ("...", SLATE_BG, SLATE), ("4", TEAL_BG, TEAL)]
    for ti, (tv, fc, tc) in enumerate(tb_vals):
        _draw_token_chip(ax, 75.5 + ti * 3.15, 16.0, 2.85, 8.4, tv, fc=fc, ec=SLATE_LIGHT, tc=tc, fontsize=7.9, bold=True)
    ax.text(75.5, 9.5, "Ġ 对应原始空格，计为 1 字节", ha="left", fontsize=7.5, color=AMBER_DARK)

    _arrow(ax, (16.5, 55.0), (20.8, 60.5), color=SLATE_DARK)
    _arrow(ax, (45.2, 60.5), (49.0, 56.0), color=INDIGO)
    ax.plot([68.3, 71.5], [48.5, 48.5], color=SLATE_DARK, lw=1.2)
    ax.plot([71.5, 71.5], [20.2, 65.0], color=SLATE_DARK, lw=1.2)
    _arrow(ax, (71.5, 65.0), (74.6, 65.0), color=INDIGO)
    _arrow(ax, (71.5, 44.2), (74.6, 44.2), color=SLATE_DARK)
    _arrow(ax, (71.5, 20.2), (74.6, 20.2), color=TEAL)

    _save_both(fig, out_dir, "tokenizer-training-pipeline")


# ---------------------------------------------------------------------------
# 图 5：DGX Spark 实测多面板定量图（保留并跑像素审计）
# ---------------------------------------------------------------------------
def plot_empirical_eval(eval_json: Path, out_dir: Path) -> None:
    raw = json.loads(eval_json.read_text(encoding="utf-8"))
    by_name = {d["tokenizer_name"]: d for d in raw}

    order_toks = ["toy771", "v2048", "v8192", "v32768"]
    tok_labels = {
        "toy771": "toy771 (本地饱和)",
        "v2048": "v2048 (climbmix 30M)",
        "v8192": "v8192 (climbmix 30M)",
        "v32768": "v32768 (climbmix 30M)",
    }
    tok_colors = {
        "toy771": "#CBD5E1",
        "v2048": "#94A3B8",
        "v8192": INDIGO,
        "v32768": TEAL,
    }

    domain_order = [
        "chinese_tech",
        "mixed_zh_en_code",
        "latex_math",
        "digits_arithmetic",
        "python_code",
        "climbmix_val",
        "smoltalk",
    ]
    domain_labels = {
        "chinese_tech": "中文技术文本",
        "mixed_zh_en_code": "中英代码混合",
        "latex_math": "LaTeX 数学公式",
        "digits_arithmetic": "连续数字算术",
        "python_code": "Python 代码",
        "climbmix_val": "climbmix 英文验证",
        "smoltalk": "smoltalk 英文对话",
    }

    fig = plt.figure(figsize=(13.4, 7.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.22, 1.0], height_ratios=[1.05, 1.0], wspace=0.28, hspace=0.40)

    ax_a = fig.add_subplot(gs[:, 0])
    y_pos = np.arange(len(domain_order))
    bar_h = 0.19
    offsets = [-1.5 * bar_h, -0.5 * bar_h, 0.5 * bar_h, 1.5 * bar_h]

    for idx, tname in enumerate(order_toks):
        vals = [by_name[tname]["domains"][d]["bytes_per_token"] for d in domain_order]
        bars = ax_a.barh(
            y_pos + offsets[idx],
            vals,
            height=bar_h,
            color=tok_colors[tname],
            label=tok_labels[tname],
            edgecolor="none",
            zorder=3,
        )
        if tname in ("v8192", "v32768"):
            for b, val in zip(bars, vals):
                ax_a.text(
                    val + 0.06,
                    b.get_y() + b.get_height() / 2,
                    f"{val:.2f}",
                    va="center",
                    ha="left",
                    fontsize=8.0,
                    color=INDIGO if tname == "v8192" else TEAL,
                    fontweight="bold" if tname == "v8192" else "normal",
                )

    ax_a.axvline(1.0, color=BURGUNDY, linestyle="--", linewidth=1.15, zorder=2)
    ax_a.text(1.06, 6.68, "1.0 byte/token（逐字节无压缩下界）", color=BURGUNDY, fontsize=8.3, fontweight="bold", va="center", ha="left")

    v8192_en = by_name["v8192"]["domains"]["climbmix_val"]["bytes_per_token"]
    v8192_zh = by_name["v8192"]["domains"]["chinese_tech"]["bytes_per_token"]
    tax_ratio = v8192_en / v8192_zh
    ax_a.text(
        2.35,
        0.10,
        f"v8192 中文同字节需 {tax_ratio:.2f}" + r"$\times$ token" + "\n（中文语料压缩率偏低）",
        va="center",
        ha="left",
        fontsize=8.5,
        color=BURGUNDY,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", facecolor=BURGUNDY_BG, edgecolor=BURGUNDY, lw=0.95),
        zorder=5,
    )
    ax_a.annotate(
        "",
        xy=(1.02, -0.01),
        xytext=(2.32, 0.02),
        arrowprops=dict(arrowstyle="-|>", color=BURGUNDY, lw=1.05, shrinkA=0, shrinkB=2, connectionstyle="arc3,rad=-0.22"),
        zorder=5,
    )

    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels([domain_labels[d] for d in domain_order], fontsize=9.6, color=SLATE_DARK)
    ax_a.set_xlim(0, 5.25)
    ax_a.set_ylim(-0.62, 7.05)
    ax_a.set_xlabel("压缩率（bytes / token，越大压缩越强）", fontsize=9.8, color=SLATE_DARK)
    ax_a.text(-0.08, 1.03, "a", transform=ax_a.transAxes, fontsize=13.0, fontweight="bold", color=SLATE_DARK)
    ax_a.set_title("7 领域实测压缩率对比（每域 10 万字符截断）", loc="left", fontsize=11.0, fontweight="bold", color=SLATE_DARK, pad=10)
    ax_a.grid(axis="x", linestyle=":", alpha=0.50, color=SLATE_BORDER, zorder=1)
    ax_a.legend(loc="lower right", bbox_to_anchor=(0.98, 0.25), fontsize=8.4, ncol=1)

    ax_b = fig.add_subplot(gs[0, 1])
    d_model = 320
    transformer_m = 7.3728
    vocab_sizes = [by_name[t]["vocab_size"] for t in order_toks]
    emb_head_m = [2 * vs * d_model / 1e6 for vs in vocab_sizes]
    total_m = [transformer_m + eh for eh in emb_head_m]
    util_pct = [by_name[t]["domains"]["climbmix_val"]["vocab_utilization"] * 100 for t in order_toks]

    x = np.arange(len(order_toks))
    w_bar = 0.50
    ax_b.bar(x, [transformer_m] * len(order_toks), width=w_bar, color="#CBD5E1", label="Transformer 骨干 (固定 7.37M)", zorder=3)
    ax_b.bar(x, emb_head_m, bottom=[transformer_m] * len(order_toks), width=w_bar, color=INDIGO, label=r"tok_emb + lm_head ($2\cdot V \cdot d_{\mathrm{model}}$)", zorder=3)

    for i, (tot, eh, u) in enumerate(zip(total_m, emb_head_m, util_pct)):
        share = eh / tot * 100
        ax_b.text(i, tot + 0.65, f"词表占比 {share:.1f}%\n验证利用率 {u:.1f}%", ha="center", va="bottom", fontsize=8.0, color=SLATE_DARK, linespacing=1.22)

    ax_b.set_xticks(x)
    ax_b.set_xticklabels([f"{t}\n(V={vs:,})" for t, vs in zip(order_toks, vocab_sizes)], fontsize=8.9, color=SLATE_DARK)
    ax_b.set_xlim(-0.68, 3.65)
    ax_b.set_ylim(0, 35.5)
    ax_b.set_ylabel("参数量（百万 M）", fontsize=9.6, color=SLATE_DARK)
    ax_b.text(-0.12, 1.05, "b", transform=ax_b.transAxes, fontsize=13.0, fontweight="bold", color=SLATE_DARK)
    ax_b.set_title("d_model=320 首尾参数占比与英文验证利用率", loc="left", fontsize=10.8, fontweight="bold", color=SLATE_DARK, pad=10)
    ax_b.grid(axis="y", linestyle=":", alpha=0.50, color=SLATE_BORDER, zorder=1)
    ax_b.legend(loc="upper left", fontsize=8.1)

    ax_c = fig.add_subplot(gs[1, 1])
    raw_bytes = [by_name[t]["token_bytes_audit"]["reference_raw_utf8_bytes"] / 1000 for t in order_toks]
    true_bytes = [by_name[t]["token_bytes_audit"]["reference_total_true_bytes"] / 1000 for t in order_toks]
    form_bytes = [by_name[t]["token_bytes_audit"]["reference_total_formula_bytes"] / 1000 for t in order_toks]

    w2 = 0.25
    ax_c.bar(x - w2, raw_bytes, width=w2, color="#CBD5E1", label="文本真实 UTF-8 (103.39 KB)", zorder=3)
    ax_c.bar(x, true_bytes, width=w2, color=TEAL, label="ByteLevel 反查还原 (103.39 KB)", zorder=3)
    ax_c.bar(x + w2, form_bytes, width=w2, color=BURGUNDY, label="encode('utf-8') 公式 (120.27 KB)", zorder=3)

    ax_c.axhline(raw_bytes[0], color=TEAL, linestyle=":", linewidth=1.0, zorder=2)
    ax_c.annotate(
        "统一高估 +16.33%\n(Ġ 把 1B 空格算成 2B)\n使绝对 BPB 偏低约 14.0%",
        xy=(2 + w2, form_bytes[2]),
        xytext=(1.65, 146),
        fontsize=8.0,
        color=BURGUNDY,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=BURGUNDY_BG, edgecolor=BURGUNDY, lw=0.95),
        arrowprops=dict(arrowstyle="-|>", color=BURGUNDY, lw=1.0, shrinkA=0, shrinkB=2),
        zorder=5,
    )

    ax_c.set_xticks(x)
    ax_c.set_xticklabels(order_toks, fontsize=9.1, color=SLATE_DARK)
    ax_c.set_ylim(0, 195)
    ax_c.set_ylabel("climbmix_val 字节累计（千字节 KB）", fontsize=9.4, color=SLATE_DARK)
    ax_c.text(-0.12, 1.05, "c", transform=ax_c.transAxes, fontsize=13.0, fontweight="bold", color=SLATE_DARK)
    ax_c.set_title("token_bytes.pt 分母口径校准审计", loc="left", fontsize=10.8, fontweight="bold", color=SLATE_DARK, pad=10)
    ax_c.grid(axis="y", linestyle=":", alpha=0.50, color=SLATE_BORDER, zorder=1)
    ax_c.legend(loc="upper left", fontsize=7.8)

    fig.text(
        0.015,
        -0.02,
        "数据来源：DGX Spark 与本地基准实测（domain_compression_eval.json）；Panel b 按 untied weights 计算，Panel c 以 1,000 字节为 1 KB。",
        fontsize=8.3,
        color=SLATE,
    )
    _save_both(fig, out_dir, "tokenizer-domain-eval-and-bpb-audit")


def main() -> int:
    default_eval_json = None  # 缺省 $XL_OUT_DIR/tokenizer/domain_compression_eval.json
    default_out_dir = Path(__file__).resolve().parents[3] / "docs" / "zh" / "inference" / "figures"

    ap = argparse.ArgumentParser(description="生成分词器章节科研配图（PNG + SVG）")
    ap.add_argument("--eval-json", type=Path, default=default_eval_json,
                    help="domain_compression_eval.json 路径（缺省 $XL_OUT_DIR/tokenizer/ 下）")
    ap.add_argument("--out-dir", type=Path, default=default_out_dir, help="输出图片目录")
    args = ap.parse_args()

    eval_json = args.eval_json or (lc.out_dir() / "tokenizer" / "domain_compression_eval.json")
    plot_byte_codec_flow(args.out_dir)
    plot_paradigms_comparison(args.out_dir)
    plot_vocab_tradeoffs(args.out_dir)
    plot_training_pipeline(args.out_dir)
    plot_empirical_eval(eval_json, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
