# -*- coding: utf-8 -*-
"""
test_vector_db_bench · 图表生成脚本
====================================

主报告 4 张图读 index_bench.py 结果 JSON：
  fig1-recall-latency.png        召回-延迟散点（帕累托前沿）  → 主报告第四节 ef 扫描
  fig2-storage-amplification.png 索引存储放大倍数（净/原始）  → 主报告第三节 存储分段
  fig3-index-build-time.png      索引构建时间排序柱状图       → 主报告第二节 构建时间
  fig4-write-throughput.png      写入吞吐排序柱状图           → 主报告第一节 吞吐

其余 3 张图数据来自对应报告里的表格（存档数据/调参根因，无对应 JSON）：
  fig5-surreal-milvus-ef.png     SurrealDB vs Milvus ef 异常  → 随机向量专项测试 第二节
  fig6-lancedb-param-root.png    LanceDB IVF_PQ 调参根因      → 真实数据五库对比 根因节
  fig7-deployment-heatmap.png    部署能力矩阵热力图           → 部署能力矩阵（原部署报告，结论已迁知识库，数据内联于本脚本）
  fig10-hybrid-rrf.png          混合检索 RRF 命中率对比       → 混合检索BM25对比 第二节

用法（需带 matplotlib 等绘图依赖的环境，本目录无 .venv）：
  python make_figures.py
  ... make_figures.py <results.json> <outdir>

fig10 单独生成（读 hybridbench JSON）：
  python make_figures.py --hybrid ../results/hybridbench-<stamp>.json
"""
import json
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---- 中文字体：优先 Microsoft YaHei，缺失回退 Noto Sans SC / SimHei ----
_CN = ["Microsoft YaHei", "Noto Sans SC", "SimHei", "Microsoft JhengHei"]
_avail = {f.name for f in font_manager.fontManager.ttflist}
_cn = next((f for f in _CN if f in _avail), "sans-serif")
plt.rcParams.update({
    "font.sans-serif": [_cn, "DejaVu Sans"],
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.28,
    "grid.linewidth": 0.5,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})

# ---- 配色：Okabe-Ito（色盲安全、扁平、学术）按库固定 ----
DB_COLOR = {
    "Milvus":    "#0072B2",  # 蓝
    "Qdrant":    "#D55E00",  # 朱红
    "SurrealDB": "#CC79A7",  # 紫红
    "LanceDB":   "#009E73",  # 绿
    "sqlite-vec": "#999999", # 灰
    "ChromaDB":  "#E69F00",  # 橙
}


# 内存基准专用：索引调色板（区分同库不同索引）
IDX_COLOR = {
    "IVF_PQ":   "#0072B2",
    "IVF_FLAT": "#56B4E9",
    "IVF_SQ":   "#009E73",
    "IVF_SQ8":  "#009E73",
    "HNSW_PQ":  "#CC79A7",
    "HNSW":     "#CC79A7",
    "DISKANN":  "#D55E00",
    "SCANN":    "#E69F00",
    "FLAT":     "#999999",
}


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def short_idx(r):
    idx = r["index"].replace("HNSW_", "H").replace("HNSW", "H")
    return idx


def fig_recall_latency(data, out):
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    for r in data:
        x = r["ef128"]["p50_ms"]
        y = r["ef128"]["recall"]
        c = DB_COLOR.get(r["db"], "#333333")
        ax.scatter(x, y, s=95, color=c, edgecolor="white",
                   linewidth=0.8, zorder=3, alpha=0.93)
    # 帕累托最优区（高召回低延迟）淡色底
    ax.axhspan(0.985, 1.06, xmin=0, xmax=1, color="#009E73",
               alpha=0.06, zorder=0)
    # 前沿聚类整体标注：12 个高效配置挤在 recall≈1、4.5-8ms
    ax.annotate("高效配置聚集区\n（12 个，recall≈1.0，4.5–8ms）",
                xy=(8.1, 0.995), xytext=(10.6, 0.90),
                fontsize=7.6, color="#009E73", fontweight="bold",
                ha="left", va="center", linespacing=1.25,
                arrowprops=dict(arrowstyle="->", color="#009E73",
                                lw=0.9, shrinkA=2, shrinkB=4))
    # 只单独标注有区分度的离群点（空间上分散，标签不撞）
    outliers = [
        ("Milvus IVF_PQ",   4.50, 0.390,   7,  2, "left",  "bottom"),
        ("LanceDB HNSW_PQ", 5.32, 0.537,   7,  0, "left",  "center"),
        ("Qdrant HScalarI8", 4.42, 0.960, -8, -4, "right", "top"),
        ("Qdrant HBinary",  4.66, 0.775,   7,  0, "left",  "center"),
        ("Qdrant HPQ",      6.22, 0.838,   7,  0, "left",  "center"),
        ("sqlite-vec FLAT", 17.0, 1.000,  0, -14, "center", "top"),
        ("Milvus DISKANN",  19.68, 1.000, 0, -14, "center", "top"),
    ]
    for text, x, y, dx, dy, ha, va in outliers:
        ax.annotate(text, (x, y), textcoords="offset points",
                    xytext=(dx, dy), ha=ha, va=va,
                    fontsize=7.0, color="#2b2b2b", zorder=4)
    ax.set_xlabel("查询延迟 p50 @ ef=128（毫秒，越左越快）")
    ax.set_ylabel("召回率 recall@10（越上越准）")
    ax.set_title("召回 - 延迟分布：每个点是一个 (库, 索引) 配置")
    ax.set_xlim(3, 22)
    ax.set_ylim(0.30, 1.06)
    # 库图例放图外底部，避免压点
    handles = [plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=DB_COLOR[db],
                          markersize=9, label=db)
               for db in DB_COLOR if any(r["db"] == db for r in data)]
    ax.legend(handles=handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.11), frameon=False, ncol=5)
    fig.savefig(out)
    plt.close(fig)


# ============================================================
# 图 2：索引存储放大倍数（索引净字节 / 原始向量字节）
# ============================================================
def fig_storage_amplification(data, out):
    rows = []
    for r in data:
        raw = r["storage"]["raw_vector_bytes"]
        net = r["storage"]["index_net_bytes"]
        amp = (net / raw) if raw else 0.0
        rows.append((r["db"], short_idx(r), amp))
    rows.sort(key=lambda t: t[2])
    CAP = 4.5  # 截断阈值：Qdrant 预分配高柱截断，小柱才看得清
    fig, ax = plt.subplots(figsize=(9.4, 6.2))
    xs = range(len(rows))
    colors = [DB_COLOR[db] for db, _, _ in rows]
    vals = [a for _, _, a in rows]
    disp = [min(v, CAP) if v > CAP else v for v in vals]
    ax.bar(xs, disp, color=colors, edgecolor="white",
           linewidth=0.6, zorder=3)
    for i, (db, idx, a, d) in enumerate(zip(
            [r[0] for r in rows], [r[1] for r in rows], vals, disp)):
        if abs(a) < 0.02:
            ax.text(i, 0.12, "0\n(无独立索引)", ha="center", va="bottom",
                    fontsize=6.6, color="#555555", linespacing=1.0)
        elif a > CAP:
            ax.text(i, CAP + 0.15, f"≈{a:.1f}x\n(截断)", ha="center",
                    va="bottom", fontsize=6.6, color="#D55E00",
                    fontweight="bold", linespacing=1.0)
        elif a < 0:
            ax.text(i, a - 0.4, f"{a:.1f}x", ha="center", va="top",
                    fontsize=7, color="#D55E00", fontweight="bold")
        else:
            ax.text(i, d + 0.08, f"{a:.2f}x", ha="center", va="bottom",
                    fontsize=6.8, color="#2b2b2b")
    ax.axhline(1.0, color="#888888", linewidth=0.8, linestyle="--",
               zorder=2)
    ax.axhline(CAP, color="#bbbbbb", linewidth=0.6, linestyle=":",
               zorder=1)
    ax.text(0.3, CAP + 0.1, "截断线 4.5x", ha="left", va="bottom",
            fontsize=6.5, color="#999999")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"{db}\n{idx}" for db, idx, _ in rows],
                       rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("索引净开销 / 原始向量字节")
    ax.set_title("索引存储放大倍数（高出的 Qdrant 柱已截断）")
    ax.set_ylim(-12, CAP + 1.6)
    ax.text(0.005, 0.975,
            "Qdrant HNSW 真值 ≈17x，含空集合预分配（报告第五节）\n"
            "Qdrant PQ 为负：量化后回收原始向量；负值不截断",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=7, color="#555555", linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.3", fc="#fafafa",
                      ec="#dddddd", lw=0.5))
    handles = [plt.Line2D([0], [0], marker="s", color="w",
                          markerfacecolor=DB_COLOR[db],
                          markersize=9, label=db)
               for db in DB_COLOR if any(d == db for d, _, _ in rows)]
    ax.legend(handles=handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.14), frameon=False, ncol=5)
    fig.savefig(out)
    plt.close(fig)


# ============================================================
# 图 3：索引构建时间
# ============================================================
def fig_build_time(data, out):
    rows = sorted(((r["db"], short_idx(r), r["index_build_s"])
                   for r in data), key=lambda t: t[2])
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    xs = range(len(rows))
    colors = [DB_COLOR[db] for db, _, _ in rows]
    ax.bar(xs, [t for _, _, t in rows], color=colors,
           edgecolor="white", linewidth=0.6, zorder=3)
    for i, (db, idx, t) in enumerate(rows):
        if t < 0.05:
            ax.text(i, 0.12, "0", ha="center", va="bottom",
                    fontsize=7.5, color="#555555")
        else:
            ax.text(i, t + 0.18, f"{t:.2f}", ha="center", va="bottom",
                    fontsize=7, color="#2b2b2b")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"{db}\n{idx}" for db, idx, _ in rows],
                       rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("建索引时间（秒）")
    ax.set_title("索引构建时间排序（口径差异见报告第二节）")
    ax.set_ylim(0, max(t for _, _, t in rows) * 1.15)
    ax.text(0.99, 0.97,
            "SurrealDB=随写入构建，sqlite-vec=无 ANN 索引\n"
            "Qdrant=增量同步等待，Milvus=异步轮询 Finished\n"
            "三者口径不同，不能直接比绝对值",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=7, color="#666666", linespacing=1.3)
    fig.savefig(out)
    plt.close(fig)


# ============================================================
# 图 4：写入吞吐
# ============================================================
def fig_write_throughput(data, out):
    rows = sorted(((r["db"], short_idx(r), r["write_vec_per_s"])
                   for r in data), key=lambda t: t[2])
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    xs = range(len(rows))
    colors = [DB_COLOR[db] for db, _, _ in rows]
    ax.bar(xs, [t for _, _, t in rows], color=colors,
           edgecolor="white", linewidth=0.6, zorder=3)
    for i, (db, idx, t) in enumerate(rows):
        ax.text(i, t + max(v for *_, v in rows) * 0.012,
                f"{t:,.0f}", ha="center", va="bottom",
                fontsize=7, color="#2b2b2b")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"{db}\n{idx}" for db, idx, _ in rows],
                       rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("写入吞吐（vec/s）")
    ax.set_title("写入吞吐排序（嵌入式天然高于客户端-服务器）")
    ax.set_ylim(0, max(t for _, _, t in rows) * 1.14)
    ax.text(0.99, 0.97,
            "SurrealDB 走 HTTP /sql，400 条以上 413 拒收，是受限下限\n"
            "LanceDB / sqlite-vec 嵌入式无网络往返，吞吐天然高",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=7, color="#666666", linespacing=1.3)
    fig.savefig(out)
    plt.close(fig)


# ============================================================
# 图 5：SurrealDB vs Milvus 的 ef 异常（随机向量深潜）
# 数据来源：reports/2026-09-03-随机向量专项测试.md 第二节表格
# （30k×768 随机向量，ef=32..512 深潜；存档数据，仅示异常）
# ============================================================
def fig_surreal_milvus_ef(data, out):
    ef = [32, 48, 64, 128, 256, 512]
    surreal_p50 = [16210, 13, 95, 84, 84, 22]      # ef=32 劣化到 16 秒
    surreal_rec = [0.61, 0.33, 1.00, 1.00, 1.00, 0.84]  # 非单调
    milvus_p50 = [6, 5, 6, 6, 9, 8]
    milvus_rec = [0.30, 0.37, 0.40, 0.55, 0.70, 0.83]  # 单调
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    # 延迟：跨 4 个数量级（5ms–16s），对数轴
    ax1.plot(ef, surreal_p50, "-o", color=DB_COLOR["SurrealDB"],
             label="SurrealDB", markersize=5)
    ax1.plot(ef, milvus_p50, "-o", color=DB_COLOR["Milvus"],
             label="Milvus", markersize=5)
    ax1.axhline(1000, color="#cc4444", linewidth=0.7, linestyle=":")
    ax1.annotate("ef=32 劣化到 16s", xy=(32, 16210), xytext=(60, 6000),
                 fontsize=7.5, color="#cc4444",
                 arrowprops=dict(arrowstyle="->", color="#cc4444", lw=0.8))
    ax1.set_yscale("log")
    ax1.set_xlabel("ef（搜索宽度）")
    ax1.set_ylabel("查询延迟 p50（毫秒，对数轴）")
    ax1.set_title("延迟 vs ef：SurrealDB 曲线不平滑")
    ax1.legend(frameon=False, loc="lower right")
    # 召回
    ax2.plot(ef, surreal_rec, "-o", color=DB_COLOR["SurrealDB"],
             label="SurrealDB", markersize=5)
    ax2.plot(ef, milvus_rec, "-o", color=DB_COLOR["Milvus"],
             label="Milvus", markersize=5)
    ax2.annotate("ef=512 召回反降到 0.84", xy=(512, 0.84),
                 xytext=(180, 0.55), fontsize=7.5, color="#cc4444",
                 arrowprops=dict(arrowstyle="->", color="#cc4444", lw=0.8))
    ax2.set_xlabel("ef（搜索宽度）")
    ax2.set_ylabel("召回率 recall@10")
    ax2.set_title("召回 vs ef：SurrealDB 非单调")
    ax2.legend(frameon=False, loc="lower right")
    fig.suptitle("SurrealDB vs Milvus · ef 响应异常（30k×768 随机向量，存档）",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out)
    plt.close(fig)


# ============================================================
# 图 6：LanceDB IVF_PQ 调参根因（真实数据）
# 数据来源：reports/2026-09-03-真实数据-五库对比.md 根因表
# ============================================================
def fig_lancedb_param_root(data, out):
    configs = ["sub=32\npart=256\n(旧)", "sub=48\npart=128",
               "sub=64\npart=128", "sub=64 part=128\n+refine=10",
               "sub=96\npart=64"]
    ef16 = [0.608, 0.696, 0.774, 0.980, 0.862]
    ef64 = [0.608, 0.702, 0.780, 1.000, 0.866]
    ef256 = [0.608, 0.702, 0.780, 1.000, 0.866]
    import numpy as np
    x = np.arange(len(configs))
    w = 0.26
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ax.bar(x - w, ef16, w, label="ef=16", color="#9ecae1",
           edgecolor="white", linewidth=0.5)
    ax.bar(x,     ef64, w, label="ef=64", color="#4292c6",
           edgecolor="white", linewidth=0.5)
    ax.bar(x + w, ef256, w, label="ef=256", color="#084594",
           edgecolor="white", linewidth=0.5)
    ax.axhline(1.0, color="#888888", linewidth=0.7, linestyle="--")
    # 标出 refine 那一组（第 4 根，index 3）
    ax.annotate("refine 精排把召回拉到 1.00", xy=(3 + w, 1.0),
                xytext=(1.7, 1.04), ha="center", va="bottom",
                fontsize=7.8, color="#084594", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#084594", lw=0.9),
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec="#084594", lw=0.8, alpha=0.95))
    ax.annotate("无 refine：ef 无响应，天花板 0.61-0.78",
                xy=(0, 0.608), xytext=(0.35, 0.32), fontsize=7.8,
                color="#cc4444", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#cc4444", lw=0.9),
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec="#cc4444", lw=0.8, alpha=0.95))
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=7.5)
    ax.set_ylabel("召回率 recall@10")
    ax.set_xlabel("IVF_PQ 参数配置")
    ax.set_title("LanceDB IVF_PQ 调参根因：refine_factor 才是关键")
    ax.set_ylim(0, 1.12)
    ax.legend(frameon=False, title="ef", loc="upper left")
    fig.savefig(out)
    plt.close(fig)


# ============================================================
# 图 7：部署能力矩阵热力图（定性）
# 数据来源：部署能力矩阵（原 reports/2026-09-04-部署方式与平台支持.md，已迁知识库，数值内联于本脚本）
# 支持度编码：2=完整原生 / 1=有条件或部分 / 0=不支持或不需要
# ============================================================
def fig_deployment_heatmap(data, out):
    import numpy as np
    from matplotlib.colors import ListedColormap
    dbs = ["Milvus", "Qdrant", "SurrealDB", "LanceDB", "sqlite-vec"]
    cols = ["独立服务", "嵌入式库", "云托管", "分布式集群", "Windows 免 Docker"]
    # 行=库，列=维度；值 2/1/0；text 为格内短注
    score = np.array([
        [2, 1, 2, 2, 0],   # Milvus: Lite 限 Linux/Mac
        [2, 1, 2, 2, 2],   # Qdrant: 内存模式；Windows 原生 exe
        [2, 2, 1, 2, 2],   # SurrealDB: Rust/WASM；Cloud 生态小
        [0, 2, 2, 1, 2],   # LanceDB: 无服务；企业版分布式
        [0, 2, 0, 0, 2],   # sqlite-vec: SQLite 扩展
    ])
    note = [
        ["有", "Lite限Linux/Mac", "Zilliz", "K8s", "仅 Docker"],
        ["有", "内存模式", "Qdrant Cloud", "有", "原生 exe"],
        ["有", "Rust/WASM", "生态较小", "TiKV", "原生 exe"],
        ["无", "有", "Cloud", "企业版", "pip 即用"],
        ["无", "SQLite 扩展", "无", "无", "pip 即用"],
    ]
    cmap = ListedColormap(["#e8e8e8", "#c6dbef", "#43a2ca"])
    fig, ax = plt.subplots(figsize=(8.8, 3.9))
    ax.imshow(score, cmap=cmap, vmin=0, vmax=2, aspect="auto")
    for i in range(len(dbs)):
        for j in range(len(cols)):
            v = score[i, j]
            color = "white" if v == 2 else ("#08306b" if v == 1 else "#555555")
            ax.text(j, i, note[i][j], ha="center", va="center",
                    fontsize=7.6, color=color, fontweight="bold")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=9)
    ax.set_yticks(range(len(dbs)))
    ax.set_yticklabels(dbs, fontsize=9)
    ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(dbs), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(axis="both", length=0)
    ax.set_title("部署能力矩阵：深色=完整原生，中色=有条件，浅色=无/不需要")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_memory_index_scale(mem3, out):
    """嵌入式库索引级内存：50k vs 100k 总增量对比。

    用 build_index 完成后的绝对内存减去基线（before_mb）得到净增量，
    避免 pyarrow mmap 导致的分阶段 RSS 测量噪声（write/index 阶段有负值）。
    """
    rows = []
    for db, specs in mem3.items():
        if db in ("milvus", "qdrant"):
            continue
        for spec, runs in specs.items():
            for r in runs:
                # 净增量 = after_index - before，用绝对内存差值
                net = r["after_index_mb"] - r["before_mb"]
                rows.append((db, spec, r["n"], net, r["before_mb"], r["after_index_mb"]))
    if not rows:
        return

    rows.sort(key=lambda x: (x[0], x[1], x[2]))
    # 提取 (db, spec) 分组
    groups = []
    seen = set()
    for db, spec, n, net, before, after in rows:
        if (db, spec) not in seen:
            seen.add((db, spec))
            groups.append((db, spec))
    # 每组取 50k 和 100k
    group_data = {}
    for db, spec, n, net, before, after in rows:
        group_data.setdefault((db, spec), {})[n] = net

    fig, ax = plt.subplots(figsize=(11, 5.2))
    bar_w = 0.36
    x = range(len(groups))
    for i, (db, spec) in enumerate(groups):
        nets = group_data[(db, spec)]
        v50 = nets.get(50000, 0)
        v100 = nets.get(100000, 0)
        ax.bar(i - bar_w / 2, v50, bar_w, color="#0072B2",
               label="50k" if i == 0 else "")
        ax.bar(i + bar_w / 2, v100, bar_w, color="#D55E00",
               label="100k" if i == 0 else "")
        ax.text(i - bar_w / 2, v50 + 12, "%+.0f" % v50, ha="center", fontsize=8)
        ax.text(i + bar_w / 2, v100 + 12, "%+.0f" % v100, ha="center", fontsize=8)

    ax.set_xticks(list(x))
    ax.set_xticklabels(["%s\n%s" % (db, spec) for db, spec in groups], fontsize=8.5)
    ax.set_ylabel("内存净增量（MB，after_index - before）")
    ax.set_title("嵌入式库索引级内存：50k vs 100k 净增量（绝对内存差值）")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.axhline(0, color="#666", linewidth=0.8)
    fig.savefig(out)
    plt.close(fig)


def fig_memory_docker_index(mem3, out):
    """Docker 库索引级内存：各索引在 100k 下的容器内存增量柱状图。

    Docker 库只测容器内存增量，不分阶段。
    """
    rows = []
    for db in ("milvus", "qdrant"):
        for spec, runs in mem3.get(db, {}).items():
            for r in runs:
                rows.append((db, spec, r))
    if not rows:
        return

    rows.sort(key=lambda x: (x[0], x[1], x[2]["n"]))
    labels = ["%s %s %dk" % (db, spec, r["n"] // 1000) for db, spec, r in rows]
    totals = [r["total_delta"] for db, spec, r in rows]
    colors = [IDX_COLOR.get(spec, "#333333") for _, spec, _ in rows]

    fig, ax = plt.subplots(figsize=(11, 5.0))
    bars = ax.bar(range(len(rows)), totals, color=colors, edgecolor="white", linewidth=0.6)
    for i, (bar, t) in enumerate(zip(bars, totals)):
        ax.text(bar.get_x() + bar.get_width() / 2, t + 3, "%+.0f" % t,
                ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, fontsize=7.5, rotation=30, ha="right")
    ax.set_ylabel("容器内存增量（MB）")
    ax.set_title("Docker 库索引级内存：各索引在 50k/100k 下的容器内存增量")
    # 索引图例
    handles = [plt.Rectangle((0, 0), 1, 1, color=IDX_COLOR[s])
               for s in sorted(IDX_COLOR) if s in {sp for _, sp, _ in rows}]
    leg_labels = [s for s in sorted(IDX_COLOR) if s in {sp for _, sp, _ in rows}]
    ax.legend(handles, leg_labels, loc="upper right", frameon=False, fontsize=8)
    fig.savefig(out)
    plt.close(fig)


def fig_hybrid_rrf(data, out):
    """混合检索：纯向量 / 纯 BM25 / RRF 融合后的 top10 命中率分组柱状图。

    读 hybridbench-*.json。两个指标（对向量 GT / 对 BM25 GT）并排，
    用于看 RRF 是否两头兼顾。
    """
    libs = [k for k in ("milvus", "lance") if isinstance(data.get(k), list)]
    if not libs:
        return
    keys = ["vector_hit", "bm25_hit", "hybrid_vec_hit", "hybrid_bm25_hit"]
    labels = ["纯向量\n(对向量GT)", "纯BM25\n(对BM25 GT)",
              "RRF混合\n(对向量GT)", "RRF混合\n(对BM25 GT)"]
    colors = ["#4C78A8", "#F58518", "#9ecae1", "#fdae6b"]

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    n_g, n_b = len(labels), len(libs)
    w = 0.36
    for gi, lab in enumerate(labels):
        for bi, lib in enumerate(libs):
            rows = data[lib]
            v = float(np.mean([r[keys[gi]] for r in rows]))
            x = gi + (bi - (n_b - 1) / 2.0) * w
            bar = ax.bar(x, v, width=w * 0.9, color=colors[gi],
                         edgecolor="white", linewidth=0.6,
                         hatch="//" if bi == 1 else None, zorder=3)
            ax.text(x, v + 0.015, "%.3f" % v, ha="center", va="bottom",
                    fontsize=8, fontweight="bold")
    ax.set_xticks(range(n_g))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("top10 命中率")
    ax.set_ylim(0, 1.08)
    ax.set_title("混合检索：RRF 融合在两种 ground truth 上的命中率")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor="#888888",
                            hatch=None if i == 0 else "//",
                            edgecolor="white")
               for i in range(len(libs))]
    ax.legend(handles, libs, loc="upper right", frameon=False, fontsize=9)
    # 标注两臂向量索引类型不同
    ax.text(0.01, -0.22, "注：两库向量臂索引类型不同（Milvus=HNSW M16/efConstruction200，"
                       "LanceDB 默认 IVF_PQ），向量召回差距含配置因素",
            transform=ax.transAxes, fontsize=7.5, color="#666666")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    # --hybrid <json> 只生成 fig10
    if len(sys.argv) > 2 and sys.argv[1] == "--hybrid":
        outdir = sys.argv[3] if len(sys.argv) > 3 else here
        out = os.path.join(outdir, "fig10-hybrid-rrf.png")
        fig_hybrid_rrf(load(sys.argv[2]), out)
        print(f"写出 fig10-hybrid-rrf.png  ({os.path.getsize(out)/1024:.0f} KB)")
        return
    src = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(here, "..", "results",
                     "indexbench-real-20news-20260904-151957.json")
    outdir = sys.argv[2] if len(sys.argv) > 2 else here
    data = load(src)
    print(f"读取 {len(data)} 组配置 <- {src}")
    jobs = [
        ("fig1-recall-latency.png", fig_recall_latency),
        ("fig2-storage-amplification.png", fig_storage_amplification),
        ("fig3-index-build-time.png", fig_build_time),
        ("fig4-write-throughput.png", fig_write_throughput),
    ]
    # fig5/6/7 不依赖 indexbench JSON，固定生成
    jobs += [
        ("fig5-surreal-milvus-ef.png", fig_surreal_milvus_ef),
        ("fig6-lancedb-param-root.png", fig_lancedb_param_root),
        ("fig7-deployment-heatmap.png", fig_deployment_heatmap),
    ]
    for name, fn in jobs:
        out = os.path.join(outdir, name)
        fn(data, out)
        sz = os.path.getsize(out)
        print(f"  写出 {name}  ({sz/1024:.0f} KB)")
    # fig8/9 读 membench3 JSON（索引级内存），文件不存在时跳过
    mem3_path = os.path.join(here, "..", "results", "membench3-index-scale-latest.json")
    if os.path.exists(mem3_path):
        mem3 = load(mem3_path)
        for name, fn in [
            ("fig8-memory-index-embedded.png", fig_memory_index_scale),
            ("fig9-memory-index-docker.png", fig_memory_docker_index),
        ]:
            out = os.path.join(outdir, name)
            fn(mem3, out)
            print(f"  写出 {name}  ({os.path.getsize(out)/1024:.0f} KB)")
    print("完成。")


if __name__ == "__main__":
    main()
