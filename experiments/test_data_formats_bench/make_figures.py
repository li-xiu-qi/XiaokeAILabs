# -*- coding: utf-8 -*-
"""
test_data_formats_bench · 图表生成脚本
=====================================

数据来源分两类：
  1. README 里的存档表格（体积膨胀、索引扫描、五类查询耗时）
     这些场景没有对应 JSON，数值随脚本正文维护。
  2. output/bench_json_results.json（bench_json.py 的产物）
     该目录已被根 .gitignore 覆盖，脚本缺失时跳过对应图。

产物写入同级 figures/ 目录（未忽略，随仓库入库）。

用法（解释器用 pkm-hub-runtime 全局环境，本目录无 .venv）：
  "C:/Users/ke/Documents/projects/obsidian_projects/pkm-hub-runtime/.venv/Scripts/python.exe" make_figures.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
RESULTS = os.path.join(HERE, "output", "bench_json_results.json")

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

# ---- 配色：Okabe-Ito（色盲安全、扁平、学术）按格式固定 ----
C_XLSX = "#0072B2"   # 蓝
C_SQLITE = "#D55E00"  # 朱红
C_JSON = "#009E73"   # 绿
C_PANDAS = "#CC79A7"  # 紫红
C_JSONC = "#999999"   # 灰（json 列式）
C_JSONL = "#E69F00"   # 橙（jsonl）


def bar_labels(ax, bars, fmt="%.1fx", dy=0.02):
    """柱顶标注倍数，dy 为相对轴高的偏移。"""
    ymax = ax.get_ylim()[1]
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + ymax * dy,
                fmt % h, ha="center", va="bottom", fontsize=9)


# ======================================================================
# fig1 · sqlite 相对 xlsx 的体积膨胀（常规 vs 极端）
# ======================================================================
# 数据：README「体积：膨胀倍数等于 xlsx 的压缩率」节
SIZE_DATA = [
    ("常规业务数据\n5万行x15列", 1.2),
    ("枚举高重复\n20万行x20列", 3.5),
    ("列值500字节\n10万行", 43.6),
    ("500字节全同值\n100万行x20列", 207.0),
]

# ======================================================================
# fig2 · 索引数量对 sqlite 体积的影响
# ======================================================================
# 数据：README 索引数量扫描表（10万行x20列，取值5种，xlsx 21.89MB）
INDEX_DATA = [
    (0, 26.12, 1.0),
    (3, 32.09, 1.2),
    (10, 46.04, 1.8),
    (20, 65.96, 2.5),
]

# ======================================================================
# fig3 · 五类查询耗时对比（50万行x8列，对数轴）
# ======================================================================
# 数据：README「速度：索引是点查专用加速器」节
SPEED_OPS = ["点查 id", "等值过滤 cat", "范围过滤 val1", "分组聚合", "全表求和"]
SPEED_DATA = {
    "sqlite 无索引": [43.4, 54.5, 68.6, 242.1, 51.8],
    "sqlite 3 索引":  [0.2, 41.3, 288.1, 1644.3, 56.6],
    "pandas 读 xlsx": [4.4, 35.3, 15.8, 46.5, 0.3],
}


def fig1():
    tags = [d[0] for d in SIZE_DATA]
    vals = [d[1] for d in SIZE_DATA]
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    bars = ax.bar(tags, vals, color=C_SQLITE, width=0.55)
    ax.set_yscale("log")
    ax.set_ylabel("sqlite 体积 / xlsx 体积（倍，对数轴）")
    ax.set_title("sqlite 相对 xlsx 的体积膨胀：常规 1.2 倍，极端 207 倍")
    ax.set_ylim(1, 400)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.15, "%.1fx" % v,
                ha="center", va="bottom", fontsize=9)
    ax.axhline(1.0, color="#888888", lw=0.8, ls="--")
    ax.text(len(tags) - 0.5, 1.15, "体积不变（1x）", ha="right", fontsize=8,
            color="#666666")
    fig.savefig(os.path.join(FIG, "fig1-size-amplification.png"))
    plt.close(fig)


def fig2():
    n = [d[0] for d in INDEX_DATA]
    mb = [d[1] for d in INDEX_DATA]
    rel = [d[2] for d in INDEX_DATA]
    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    bars = ax.bar([str(x) for x in n], mb, color=C_SQLITE, width=0.5)
    ax.set_xlabel("索引数量（列）")
    ax.set_ylabel("sqlite 体积（MB）")
    ax.set_title("索引数量对体积的影响：建满 20 个索引，体积翻 2.5 倍")
    for b, r in zip(bars, rel):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2,
                "%.1f 倍" % r, ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, max(mb) * 1.18)
    fig.savefig(os.path.join(FIG, "fig2-index-size.png"))
    plt.close(fig)


def fig3():
    import numpy as np
    x = np.arange(len(SPEED_OPS))
    w = 0.26
    fig, ax = plt.subplots(figsize=(8.4, 3.9))
    for i, (tag, vals) in enumerate(SPEED_DATA.items()):
        color = [C_SQLITE, C_SQLITE, C_PANDAS][i]
        off = (i - 1) * w
        bars = ax.bar(x + off, vals, w, label=tag, color=color,
                      alpha=1.0 if i != 1 else 0.55)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v * 1.12, "%.1f" % v,
                    ha="center", va="bottom", fontsize=7.5, rotation=0)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(SPEED_OPS)
    ax.set_ylabel("耗时（ms，对数轴）")
    ax.set_title("五类等价查询耗时：索引只在点查上快 217 倍，其余是负优化")
    ax.set_ylim(0.1, 6000)
    ax.legend(loc="upper left", ncol=3, frameon=False)
    fig.savefig(os.path.join(FIG, "fig3-query-speed.png"))
    plt.close(fig)


def fig4_json():
    """json 与 xlsx 的体积比（来自 bench_json.py 产物）。"""
    if not os.path.exists(RESULTS):
        return
    with open(RESULTS, encoding="utf-8") as f:
        res = json.load(f)
    tags = [r["tag"] for r in res]
    xr = [r["json_rows_bytes"] / r["xlsx_bytes"] for r in res]
    xc = [r["json_cols_bytes"] / r["xlsx_bytes"] for r in res]
    import numpy as np
    x = np.arange(len(tags))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.8, 3.9))
    b1 = ax.bar(x - w / 2, xr, w, label="json 行式", color=C_JSON)
    b2 = ax.bar(x + w / 2, xc, w, label="json 列式", color=C_JSONC)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.12,
                    "%.1fx" % b.get_height(), ha="center", va="bottom",
                    fontsize=8.5)
    ax.axhline(1.0, color="#888888", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(tags, fontsize=8.5)
    ax.set_ylabel("json 体积 / xlsx 体积（倍）")
    ax.set_title("json 相对 xlsx 的体积：行式 1.9~6.7 倍，列式更小但仍更大")
    ax.set_ylim(0, max(xr + xc) * 1.2)
    ax.legend(frameon=False)
    fig.savefig(os.path.join(FIG, "fig4-json-size.png"))
    plt.close(fig)


def fig5_json():
    """json 与 xlsx 的读写耗时对比（对数轴）。"""
    if not os.path.exists(RESULTS):
        return
    with open(RESULTS, encoding="utf-8") as f:
        res = json.load(f)
    import numpy as np
    tags = [r["tag"] for r in res]
    x = np.arange(len(tags))
    w = 0.2
    fig, ax = plt.subplots(figsize=(7.8, 3.9))
    series = [
        ("xlsx 写", [r["xlsx_write_s"] for r in res], "#0072B2"),
        ("xlsx 读", [r["xlsx_read_s"] for r in res], "#56B4E9"),
        ("json 行式 写", [r["json_rows_write_s"] for r in res], "#009E73"),
        ("json 行式 读", [r["json_rows_read_s"] for r in res], "#CC79A7"),
    ]
    for i, (label, vals, color) in enumerate(series):
        off = (i - 1.5) * w
        ax.bar(x + off, vals, w, label=label, color=color)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(tags, fontsize=8.5)
    ax.set_ylabel("耗时（秒，对数轴）")
    ax.set_title("json 与 xlsx 读写耗时：xlsx 读取慢一到两个数量级")
    ax.legend(frameon=False, ncol=4, loc="upper center")
    fig.savefig(os.path.join(FIG, "fig5-json-speed.png"))
    plt.close(fig)


def fig6_jsonl():
    """jsonl 相对 json 全量形态的三个差异：体积持平、全量读更慢、追加与流式快。

    双面板：左为追加一行的耗时（json 需整文件重写），右为流式读前 1000 行
    相对全量读的耗时比，两轴都用对数以便同图展示三个数量级。
    """
    if not os.path.exists(RESULTS):
        return
    with open(RESULTS, encoding="utf-8") as f:
        res = json.load(f)
    # 只有含 jsonl 字段的产物才画这张图，兼容旧结果
    res = [r for r in res if "jsonl_bytes" in r]
    if not res:
        return
    import numpy as np
    tags = [r["tag"].replace("_", "\n") for r in res]
    app_j = [r["json_rows_append_s"] for r in res]
    app_l = [r["jsonl_append_s"] for r in res]
    speedup = [r["jsonl_read_s"] / r["jsonl_stream1000_s"] for r in res]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 3.9))
    x = np.arange(len(tags))
    w = 0.36
    a1.bar(x - w / 2, app_j, w, label="json 全量（读回+重写）", color=C_JSON)
    a1.bar(x + w / 2, app_l, w, label="jsonl（追加一行）", color=C_JSONL)
    a1.set_yscale("log")
    a1.set_ylabel("追加一行耗时（秒，对数轴）")
    a1.set_title("追加一行：jsonl 与 json 全量形态相差三个数量级")
    a1.set_xticks(x)
    a1.set_xticklabels(tags, fontsize=8)
    a1.legend(frameon=False, fontsize=8)
    for i, v in enumerate(app_j):
        a1.text(i - w / 2, v * 1.15, "%.2fs" % v, ha="center", va="bottom", fontsize=7.5)
    for i, v in enumerate(app_l):
        a1.text(i + w / 2, max(v, 1e-4) * 1.15, "%.4fs" % v, ha="center",
                va="bottom", fontsize=7.5)

    b = a2.bar(x, speedup, 0.5, color=C_JSONL)
    for bb, v in zip(b, speedup):
        a2.text(bb.get_x() + bb.get_width() / 2, v * 1.1, "%.0fx" % v,
                ha="center", va="bottom", fontsize=8.5)
    a2.set_yscale("log")
    a2.set_ylabel("全量读耗时 / 流式读前1000行（倍）")
    a2.set_title("只取前 1000 行：流式读跳过文件其余部分")
    a2.set_xticks(x)
    a2.set_xticklabels(tags, fontsize=8)
    a2.set_ylim(1, max(speedup) * 3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig6-jsonl.png"))
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    fig1()
    fig2()
    fig3()
    fig4_json()
    fig5_json()
    fig6_jsonl()
    names = sorted(os.listdir(FIG))
    print("写出 %d 张图：" % len([n for n in names if n.endswith(".png")]))
    for n in names:
        if n.endswith(".png"):
            print("  %s" % n)


if __name__ == "__main__":
    main()
