# -*- coding: utf-8 -*-
"""test_index_essence · 机制示意图（帮助理解为什么公式成立）

  fig4-hnsw-structure.png   HNSW 多层近邻图：上层长跳定位、下层精细，查询对数步下降
  fig5-ivf-pruning.png      IVF 聚类剪枝：只搜 nprobe 个簇，其余整块跳过
  fig6-pq-quantization.png  PQ 乘积量化：向量切段、各段查小码本、存 m 个字节
  fig7-storage-compare.png  百万向量各索引存储对比（呼应存储文档的外推数字）
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from matplotlib import font_manager
import numpy as np

_CN = ["Microsoft YaHei", "Noto Sans SC", "SimHei"]
_avail = {f.name for f in font_manager.fontManager.ttflist}
_cn = next((f for f in _CN if f in _avail), "sans-serif")
plt.rcParams.update({"font.family": _cn, "axes.unicode_minus": False,
                     "figure.dpi": 110, "savefig.dpi": 150, "font.size": 12})

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "figures")
os.makedirs(OUT, exist_ok=True)
BLUE, GREEN, RED, GRAY, ORANGE = "#2471a3", "#229954", "#c0392b", "#95a5a6", "#e67e22"


def fig4_hnsw():
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    rng = np.random.default_rng(7)
    # 三层：y 越高越稀疏。手工布点保证可读
    layers = [
        (3.0, [1.5, 5.0, 8.5]),                 # 顶层：3 节点
        (2.0, [0.8, 2.6, 4.4, 6.2, 8.0, 9.6]),  # 中层：6 节点
        (1.0, [0.5, 1.4, 2.3, 3.2, 4.1, 5.0, 5.9, 6.8, 7.7, 8.6, 9.5]),  # 底层
    ]
    for y, xs in layers:
        # 同层近邻连边
        for a, b in zip(xs[:-1], xs[1:]):
            ax.plot([a, b], [y, y], color=BLUE, alpha=0.35, lw=1.2, zorder=1)
        ax.scatter(xs, [y] * len(xs), s=90, color=BLUE, zorder=3)
    # 层间继承（虚线）：上层节点在下层也存在
    for x in [1.5, 5.0, 8.5]:
        ax.plot([x, x], [2.0, 3.0], color=GRAY, ls=":", lw=1, zorder=1)
    for x in [0.8, 2.6, 4.4, 6.2, 8.0, 9.6]:
        ax.plot([x, x], [1.0, 2.0], color=GRAY, ls=":", lw=0.8, alpha=0.6, zorder=1)

    # 查询路径（贪心下降）：入口顶层 1.5 -> 8.5（长跳）-> 降中层 8.0 -> 降底层 -> 目标
    qx = 9.3
    ax.scatter([qx], [1.0], marker="*", s=420, color=RED, zorder=5)
    ax.text(qx, 0.55, "查询 q", color=RED, ha="center", fontsize=12)
    path = [(1.5, 3.0), (8.5, 3.0), (8.0, 2.0), (8.6, 1.0), (9.3, 1.0)]
    for (x1, y1), (x2, y2) in zip(path[:-1], path[1:]):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                     mutation_scale=18, color=ORANGE, lw=2.2, zorder=4,
                     connectionstyle="arc3,rad=0.05"))
    ax.scatter([1.5], [3.0], s=160, facecolors="none", edgecolors=ORANGE, lw=2, zorder=5)
    ax.text(1.5, 3.35, "入口", color=ORANGE, ha="center", fontsize=11)

    ax.text(-2.15, 3.0, "第 2 层（最稀疏）\n长跳，快速接近", fontsize=10.5, color="#333", va="center", ha="right")
    ax.text(-2.15, 2.0, "第 1 层", fontsize=10.5, color="#333", va="center", ha="right")
    ax.text(-2.15, 1.0, "第 0 层（最密集）\n短边，精确找近邻", fontsize=10.5, color="#333", va="center", ha="right")
    ax.text(5.0, 3.75, "每层只走 efSearch 个节点，层数 ≈ log(n)，\n所以查询步数是 O(log n)，数据翻倍只多走几步",
            ha="center", fontsize=11, color="#444",
            bbox=dict(boxstyle="round,pad=0.4", fc="#fef5e7", ec=ORANGE, alpha=0.9))
    ax.set_xlim(-2.5, 10.6); ax.set_ylim(0.3, 4.1)
    ax.axis("off")
    ax.set_title("HNSW 多层近邻图：从顶层贪心下降，对数步找到近邻", fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig4-hnsw-structure.png")); plt.close(fig)


def fig5_ivf():
    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    rng = np.random.default_rng(3)
    centers = np.array([[2, 7], [7, 7.5], [3, 2.5], [7.5, 2.5], [5, 5]])
    probe = {0, 3}   # nprobe=2，高亮这两个簇
    q = np.array([6.3, 3.0])
    for i, c in enumerate(centers):
        pts = c + rng.normal(0, 0.9, size=(28, 2))
        on = i in probe
        col = GREEN if on else "#bdc3c7"
        ax.scatter(pts[:, 0], pts[:, 1], s=22, color=col, alpha=0.8 if on else 0.5, zorder=2)
        ax.add_patch(Circle(c, 1.7, color=col, alpha=0.12 if on else 0.06, zorder=1))
        ax.scatter(*c, marker="^", s=130, color=col if on else GRAY, zorder=3)
    ax.scatter(*q, marker="*", s=460, color=RED, zorder=5)
    ax.text(q[0] + 0.15, q[1] - 0.5, "查询 q", color=RED, fontsize=12)
    # q 到两个被探测簇中心
    for i in probe:
        ax.add_patch(FancyArrowPatch(q, centers[i], arrowstyle="-|>",
                     mutation_scale=15, color=GREEN, lw=1.8, ls="--", zorder=4, alpha=0.8))
    ax.text(2.0, 8.6, "绿色簇：nprobe=2 个最近簇，只扫这些", color=GREEN, fontsize=11)
    ax.text(6.0, 8.6, "灰色簇：整块跳过", color=GRAY, fontsize=11)
    ax.text(5.0, 0.4,
            "访问向量数 ≈ nprobe × (n / nlist)，与总数据量 n 无关；\n"
            "nprobe 越小越快但可能漏，调 nprobe 就是在召回和延迟间取舍",
            ha="center", fontsize=11, color="#444",
            bbox=dict(boxstyle="round,pad=0.4", fc="#eafaf1", ec=GREEN, alpha=0.9))
    ax.set_xlim(0, 10); ax.set_ylim(0, 9.2)
    ax.axis("off")
    ax.set_title("IVF 倒排：先用聚类把空间切块，查询只进 nprobe 个簇", fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig5-ivf-pruning.png")); plt.close(fig)


def fig6_pq():
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    m = 4
    seg_colors = ["#e74c3c", "#f39c12", "#27ae60", "#2980b9"]
    y0 = 4.2
    # 左：原始向量切成 4 段
    x_start, seg_w = 0.5, 1.5
    for i in range(m):
        ax.add_patch(Rectangle((x_start + i * seg_w, y0), seg_w * 0.92, 0.7,
                     color=seg_colors[i], alpha=0.8))
        ax.text(x_start + i * seg_w + seg_w * 0.46, y0 + 0.35, f"段{i+1}",
                ha="center", va="center", color="white", fontsize=11)
    ax.text(x_start + m * seg_w / 2, y0 + 1.0, "一个 d 维向量切成 m 段", ha="center", fontsize=12)
    ax.text(x_start + m * seg_w / 2, y0 - 0.45, "原始：每维 4 字节，共 d×4 字节", ha="center", fontsize=10, color="#666")

    # 中：每段一个小码本（几个中心），箭头选最近
    for i in range(m):
        sx = x_start + i * seg_w + seg_w * 0.4
        cy = 2.2
        for k in range(4):
            cx = sx - 0.9 + k * 0.6
            ax.scatter(cx, cy + (k % 2) * 0.4 - 0.2, s=70, color=seg_colors[i], alpha=0.45,
                       marker="s")
        ax.add_patch(FancyArrowPatch((sx, y0), (sx, cy + 0.5), arrowstyle="-|>",
                     mutation_scale=13, color=seg_colors[i], lw=1.6))
        ax.text(sx, cy - 0.75, f"段{i+1}码本\n2^b 个中心", ha="center", fontsize=8.5, color="#555")

    # 右：压缩后 m 个字节码（水平并排，一眼看出是 m 个码）
    rx, cw, cg = 7.7, 0.42, 0.13
    for i in range(m):
        x = rx + i * (cw + cg)
        ax.add_patch(Rectangle((x, y0), cw, 0.62, color=seg_colors[i], alpha=0.9))
        ax.text(x + cw / 2, y0 + 0.31, str(i + 1), ha="center", va="center", color="white", fontsize=10)
    cx = rx + (m * (cw + cg) - cg) / 2
    ax.text(cx, y0 + 1.0, "压缩后：m 个码", ha="center", fontsize=12)
    ax.text(cx, y0 - 0.45, "每码 1 字节，共 m 字节", ha="center", fontsize=10, color="#666")
    ax.add_patch(FancyArrowPatch((6.9, 2.6), (rx - 0.15, y0 + 0.3),
                 arrowstyle="-|>", mutation_scale=16, color=GRAY, lw=2,
                 connectionstyle="arc3,rad=-0.25"))

    ax.text(4.6, 0.5,
            "距离不用解压：查询向量各段到中心的距离预先算成表，\n"
            "查 m 次表相加即得近似距离（ADC），成本与维度 d 基本无关 → PQ 又小又快",
            ha="center", fontsize=11, color="#444",
            bbox=dict(boxstyle="round,pad=0.4", fc="#fef9e7", ec=ORANGE, alpha=0.9))
    ax.set_xlim(-0.3, 10.4); ax.set_ylim(0, 5.6)
    ax.axis("off")
    ax.set_title("PQ 乘积量化：切段 → 各段在小码本里找最近中心 → 只存 m 个字节", fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig6-pq-quantization.png")); plt.close(fig)


def fig7_storage():
    names = ["PQ (m=8)", "IVF-PQ", "SQ8", "FLAT", "HNSW (M=16)"]
    mb = [8, 16, 366, 1465, 1600]
    cols = [GREEN, GREEN, ORANGE, RED, RED]
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    bars = ax.barh(names, mb, color=cols, alpha=0.85)
    ax.set_xlabel("存储占用 (MiB，对数轴)  —— 100 万向量 × 384 维")
    ax.set_xscale("log")
    for b, v in zip(bars, mb):
        ax.text(v * 1.08, b.get_y() + b.get_height() / 2, f"{v} MiB",
                va="center", fontsize=11)
    ax.set_title("同样一百万向量，索引不同，存储差两个数量级", fontsize=13)
    ax.set_xlim(5, 4000)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig7-storage-compare.png")); plt.close(fig)


if __name__ == "__main__":
    fig4_hnsw(); fig5_ivf(); fig6_pq(); fig7_storage()
    print("机制示意图已生成到", os.path.abspath(OUT))
    for f in ["fig4-hnsw-structure.png", "fig5-ivf-pruning.png",
              "fig6-pq-quantization.png", "fig7-storage-compare.png"]:
        print(" ", f)
