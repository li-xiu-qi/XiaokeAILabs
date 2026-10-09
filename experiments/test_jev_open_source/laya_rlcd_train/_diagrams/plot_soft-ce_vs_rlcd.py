# -*- coding: utf-8 -*-
"""soft-ce 与 rlcd 两个损失函数的机制对比（Obsidian 笔记配图，白底学术风）

左面板 soft-ce：logits 过 softmax 后与教师分布直接算交叉熵，一步梯度回传。
右面板 rlcd：在当前 logits 周围做高斯扰动采样，proper scoring rule 给采样打分当奖励，
优势归一化后经高斯的 log 密度做 REINFORCE 策略梯度，末尾再加回 soft CE。

颜色语义：灰色链是 no_grad 的采样与打分（不回传梯度），蓝色是回传梯度的路径。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch

# ---------------------------------------------------------------- 字体
FONT_PATH = ("C:/Users/ke/Documents/projects/obsidian_projects/pkm-hub-skills/"
             "skills/_global/figure-routing/examples/fonts/PingFangSC-Regular.ttf")
font_manager.fontManager.addfont(FONT_PATH)

plt.rcParams.update({
    "font.family": font_manager.FontProperties(fname=FONT_PATH).get_name(),
    "mathtext.fontset": "dejavusans",
    "mathtext.default": "regular",
    "axes.unicode_minus": False,
    "figure.dpi": 150,
    "savefig.dpi": 150,
})

# ---------------------------------------------------------------- 色板
INK = "#2a2f3a"
BLUE = "#3b82c4"
GREY_TXT = "#7a828e"
GREY_FILL = "#eceef1"
GREY_EDGE = "#aab0bb"
BLUE_FILL = "#f7f9fc"
BLUE_EDGE = "#3b82c4"


# ---------------------------------------------------------------- 原语
def box(ax, x0, y0, w, h, face="white", edge=INK, lw=1.4):
    patch = FancyBboxPatch((x0, y0), w, h, boxstyle="square,pad=0",
                           facecolor=face, edgecolor=edge,
                           linewidth=lw, zorder=3)
    ax.add_patch(patch)
    return patch


def arrow(ax, x0, y0, x1, y1, color=INK, lw=1.2, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=11, color=color,
                                 linewidth=lw, linestyle=ls, zorder=2,
                                 shrinkA=0, shrinkB=0))


def elbow(ax, x, y_from, y_to, x_end, color=INK, lw=1.1):
    """先横后竖再横的三段折线：起点横向到 x，竖直到 y_to，再横向到 x_end。"""
    ax.plot([x[0], x[1]], [y_from, y_from], color=color, linewidth=lw, zorder=2)
    ax.plot([x[1], x[1]], [y_from, y_to], color=color, linewidth=lw, zorder=2)
    arrow(ax, x[1], y_to, x_end, y_to, color=color, lw=lw)


def merge(ax, cx, cy, r=3.2, label="+"):
    ax.add_patch(Circle((cx, cy), r, facecolor="white", edgecolor=INK,
                        linewidth=1.4, zorder=4))
    ax.text(cx, cy, label, ha="center", va="center",
            fontsize=13, color=INK, zorder=5)


def label(ax, x, y, text, size=10, color=INK, ha="center", va="center"):
    ax.text(x, y, text, ha=ha, va=va, fontsize=size, color=color, zorder=5)


if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(15.5, 7.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # 面板分隔
    ax.plot([50, 50], [4, 96], color="#d6dae0", linewidth=1.0, zorder=1)

    # ================================================================ 左：soft-ce
    LX = 24
    label(ax, LX, 92, "soft-ce：一步蒸馏", size=13)

    box(ax, LX - 15, 74, 30, 9)
    label(ax, LX, 78.5, "模型 logits（当前分布）")

    box(ax, LX - 15, 60, 30, 9)
    label(ax, LX, 64.5, "softmax（仅本题选项内）")

    box(ax, LX - 15, 42, 30, 11, face=BLUE_FILL, edge=BLUE_EDGE, lw=1.6)
    label(ax, LX, 50, "交叉熵", size=11, color=BLUE)
    label(ax, LX, 46, r"$-\sum t_i \log q_i$", size=10.5, color=BLUE)

    box(ax, LX - 15, 26, 30, 8, face=GREY_FILL, edge=GREY_EDGE)
    label(ax, LX, 30, "教师分布 t（数据集给定）")

    arrow(ax, LX, 74, LX, 69.2)
    arrow(ax, LX, 60, LX, 53.2)
    arrow(ax, LX, 34, LX, 41.8, color=GREY_EDGE)

    # 一步梯度回传（虚线，沿右缘外侧）
    arrow(ax, LX + 15, 47.5, LX + 15, 78.5, color=BLUE, lw=1.1, ls=(0, (4, 3)))
    label(ax, LX + 17, 63, "梯度一步\n直接回传", size=9.5, color=BLUE, ha="left")

    label(ax, LX, 18, "学的是：教师的分布形状是什么样", size=10, color=GREY_TXT)
    label(ax, LX, 12.5, "target 就是教师软分布，直接拟合", size=10, color=GREY_TXT)

    # ================================================================ 右：rlcd
    RX = 77
    label(ax, RX, 92, "rlcd：采样 + 打分两步", size=13)

    box(ax, RX - 12, 76, 24, 8)
    label(ax, RX, 80, "模型 logits")

    # 采样链（灰，no_grad）
    box(ax, RX - 12, 56, 24, 14, face=GREY_FILL, edge=GREY_EDGE, lw=1.6)
    label(ax, RX, 66.5, "detach 后加高斯扰动", size=10)
    label(ax, RX, 62, r"$z=\mathrm{logits}+\varepsilon,\ \varepsilon\sim N(0,\sigma^2)$", size=8.6)
    label(ax, RX, 58, "零和投影，采样 4 个 z 后各自 softmax 得 q", size=8.6)

    # 打分链（灰，no_grad）
    box(ax, RX - 12, 41, 24, 11, face=GREY_FILL, edge=GREY_EDGE, lw=1.6)
    label(ax, RX, 48, "proper scoring rule 打分当奖励", size=9.5)
    label(ax, RX, 43.8, r"$r=\mathrm{reward}(q,t)$，优势 $\mathrm{adv}=(r-\bar r)/\mathrm{std}$", size=8.6)

    # 策略梯度项（蓝）
    box(ax, RX - 24, 24, 20, 12, face=BLUE_FILL, edge=BLUE_EDGE, lw=1.6)
    label(ax, RX - 14, 33, "策略梯度项", size=10.5, color=BLUE)
    label(ax, RX - 14, 28.5, r"$-\mathrm{adv}\cdot\log p(z\,|\,\mathrm{logits},\sigma)$", size=8.2, color=BLUE)

    # soft CE 项（蓝，与左面板同一个）
    box(ax, RX + 4, 24, 20, 12, face=BLUE_FILL, edge=BLUE_EDGE, lw=1.6)
    label(ax, RX + 14, 33, "soft CE 项", size=10.5, color=BLUE)
    label(ax, RX + 14, 28.5, "（与左面板相同）", size=9, color=BLUE)

    merge(ax, RX, 18)
    label(ax, RX, 12, "loss = 策略梯度 + soft CE", size=10)

    # 主流程（灰）
    arrow(ax, RX, 76, RX, 70.2)
    arrow(ax, RX, 56, RX, 52.2)
    # adv 数值送进策略梯度项（左缘外侧折线）
    elbow(ax, (RX - 12, RX - 17), 46.5, 33, RX - 24, color=GREY_EDGE)

    # logits 经 log p 回传梯度到策略梯度项（左缘外侧折线）
    elbow(ax, (RX - 12, RX - 21), 80, 27, RX - 24, color=BLUE)
    label(ax, RX - 5, 38.5, "log p 里的 logits\n这一路回传梯度", size=8.8, color=BLUE, ha="left")

    # logits 回传梯度到 soft CE 项（右缘外侧折线）
    elbow(ax, (RX + 12, RX + 17), 80, 27, RX + 24, color=BLUE)

    arrow(ax, RX - 14, 24, RX - 3, 19.6, color=BLUE, lw=1.1)
    arrow(ax, RX + 14, 24, RX + 3, 19.6, color=BLUE, lw=1.1)

    label(ax, RX - 24, 5.5, "灰链全程 no_grad；σ 从 0.4 线性退火到 0.1", size=9,
          color=GREY_TXT, ha="left")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    out = ("C:/Users/ke/Documents/projects/CodeProjects/Windows-Native/XiaokeAILabs-suite/"
           "XiaokeAILabs/experiments/test_jev_open_source/laya_rlcd_train/_diagrams/"
           "soft-ce-vs-rlcd损失机制.png")
    fig.savefig(out, facecolor="white")
    print("saved:", out)
