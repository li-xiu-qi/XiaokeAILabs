# -*- coding: utf-8 -*-
"""test_index_essence · 延迟模型配图生成

读 results/ 下的标定 JSON，出三张图，把数学模型和实测点画在一起：
  fig1-latency-vs-n.png   延迟随 n：FLAT 线性 O(n) vs HNSW 对数 O(log n)，标公式
  fig2-hnsw-tradeoff.png  HNSW 召回-延迟权衡（efSearch 旋钮）
  fig3-ivf-tradeoff.png   IVF nprobe 权衡：延迟与召回随 nprobe 变化

用法：
  <venv>/Scripts/python.exe figures/make_figures.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

_CN = ["Microsoft YaHei", "Noto Sans SC", "SimHei", "Microsoft JhengHei"]
_avail = {f.name for f in font_manager.fontManager.ttflist}
_cn = next((f for f in _CN if f in _avail), "sans-serif")
plt.rcParams.update({
    "font.family": _cn, "axes.unicode_minus": False,
    "figure.dpi": 110, "savefig.dpi": 150, "font.size": 12,
    "axes.grid": True, "grid.alpha": 0.3,
})

C_FLAT, C_HNSW, C_PQ, C_RECALL = "#c0392b", "#2471a3", "#229954", "#e67e22"
HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "..", "results")
OUT = os.path.join(HERE, "..", "figures")
os.makedirs(OUT, exist_ok=True)


def load(name):
    with open(os.path.join(RES, name), encoding="utf-8") as f:
        return json.load(f)


def fig1():
    """延迟随 n：线性 vs 对数，理论曲线叠实测点。"""
    d = load("query_latency_d128.json")
    flat = d["flat_nscale"]
    hnsw = d["hnsw_nscale"]
    ns = np.array([r["n"] for r in flat], dtype=float)
    tf = np.array([r["ms"] for r in flat])
    th = np.array([r["ms"] for r in hnsw])

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    # 理论曲线：FLAT = n/Q（Q 用 n≥1万 的大数据点标定，摊销充分），HNSW = a*log2(n)+b
    big = ns >= 10000
    Q = float(np.mean(ns[big] / (tf[big] / 1000)))  # 向量/秒
    xs = np.logspace(np.log10(ns.min()), np.log10(ns.max() * 3), 80)
    ax.plot(xs, xs / Q * 1000, "--", color=C_FLAT, alpha=0.55,
            label=f"理论 O(n)：t = n / Q（Q≈{Q/1e6:.0f}M/s）")
    coef = np.polyfit(np.log2(ns), th, 1)
    ax.plot(xs, coef[0] * np.log2(xs) + coef[1], "--", color=C_HNSW, alpha=0.55,
            label="理论 O(log n)：t = a·log(n) + b")
    ax.plot(ns, tf, "o-", color=C_FLAT, lw=2, ms=7, label="FLAT 实测")
    ax.plot(ns, th, "s-", color=C_HNSW, lw=2, ms=7, label="HNSW 实测（ef=64）")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("向量数 n")
    ax.set_ylabel("单次查询延迟 (ms，对数轴)")
    ax.set_title("n 从 100 到 20 万：FLAT 线性增长，HNSW 几乎走平")
    ax.legend(loc="upper left", fontsize=10)
    ax.annotate("小数据 FLAT 更快（扫几百个比图遍历便宜）\n大数据 HNSW 反超，差距越拉越大",
                xy=(3000, 0.1), xytext=(200, 0.8), fontsize=9.5,
                color="#444", arrowprops=dict(arrowstyle="->", color="#999"))
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1-latency-vs-n.png"))
    plt.close(fig)


def fig2():
    """HNSW 召回-延迟权衡：efSearch 是旋钮。"""
    d = load("recall_clustered_d128.json")
    rows = d["hnsw_ef"]
    lat = [r["ms"] for r in rows]
    rec = [r["recall"] for r in rows]
    ef = [r["ef"] for r in rows]

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    ax.plot(lat, rec, "o-", color=C_HNSW, lw=2, ms=8)
    for i, (x, y, e) in enumerate(zip(lat, rec, ef)):
        dy = 9 if e not in (32, 64) else -16
        ax.annotate(f"ef={e}", (x, y), textcoords="offset points",
                    xytext=(8, dy), fontsize=10, color="#333")
    ax.axhline(0.95, color=C_RECALL, ls=":", alpha=0.7)
    ax.text(lat[-1] * 0.55, 0.962, "召回 0.95 目标线", color=C_RECALL, fontsize=10, ha="center")
    ax.set_xlabel("单次查询延迟 (ms)")
    ax.set_ylabel("召回率 Recall@10")
    ax.set_ylim(0.7, 1.02)
    ax.set_title("HNSW：efSearch 调多大 = 用多少延迟换多少召回（簇状数据）")
    ax.annotate("计算量 t ∝ efSearch\n访问节点越多，召回越高",
                xy=(lat[3], rec[3]), xytext=(lat[1], 0.82), fontsize=10,
                color="#444", arrowprops=dict(arrowstyle="->", color="#999"))
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2-hnsw-tradeoff.png"))
    plt.close(fig)


def fig3():
    """IVF nprobe：延迟线性、召回饱和，Flat vs PQ。"""
    d = load("recall_clustered_d128.json")
    rows = d["ivf_nprobe"]
    npb = [r["nprobe"] for r in rows]
    lat_f = [r["ms"] for r in rows]
    lat_p = [r.get("ms", r.get("ivfpq_ms")) for r in rows]
    rec_f = [r["recall"] for r in rows]
    # 簇状脚本 ivf 只存了 flat 召回；延迟里 PQ 用延迟脚本数据补充
    d2 = load("query_latency_d128.json")["ivf_nprobe"]
    lat_p = [r["ivfpq_ms"] for r in d2 if r["nprobe"] in npb]
    lat_f2 = [r["ivfflat_ms"] for r in d2 if r["nprobe"] in npb]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))
    ax1.plot(npb, lat_f2, "o-", color=C_FLAT, lw=2, label="IVF-Flat 延迟")
    ax1.plot(npb[:len(lat_p)], lat_p, "s-", color=C_PQ, lw=2, label="IVF-PQ 延迟")
    ax1.set_xlabel("nprobe（探测的簇数）"); ax1.set_ylabel("延迟 (ms)")
    ax1.set_title("延迟 ∝ nprobe（访问 nprobe/nlist 的向量）")
    ax1.legend(fontsize=10)

    ax2.plot(npb, rec_f, "o-", color=C_RECALL, lw=2, label="IVF-Flat 召回")
    ax2.axhline(1.0, color="#888", ls=":", alpha=0.6)
    ax2.set_xlabel("nprobe"); ax2.set_ylabel("召回率 Recall@10")
    ax2.set_ylim(0.8, 1.03)
    ax2.set_title("召回随 nprobe 快速饱和（nprobe=4 即 1.0）")
    ax2.legend(fontsize=10)
    fig.suptitle("IVF：nprobe 是延迟-召回旋钮，PQ 再省约 3 倍距离计算", fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3-ivf-tradeoff.png"))
    plt.close(fig)


if __name__ == "__main__":
    fig1(); fig2(); fig3()
    print("图已生成到", os.path.abspath(OUT))
    for f in sorted(os.listdir(OUT)):
        if f.endswith(".png"):
            print(" ", f)
