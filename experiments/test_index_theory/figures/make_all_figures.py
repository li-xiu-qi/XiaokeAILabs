# -*- coding: utf-8 -*-
"""
test_index_theory · 通用算法可视化图表生成器
============================================

为各算法文档生成"实测 vs 理论"对比图，帮助直观看出差别。

生成策略：
  - 有实测 JSON 结果的：直接从 JSON 读数据画对比图
  - 没有实测 JSON 的：用模型公式算出理论曲线，画机制示意 + 外推曲线

输出到 figures/，文件名以 fig-<algo>- 开头，避免与现有编号冲突。
"""

import os
import sys
import json
import glob
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

_CN = ["Microsoft YaHei", "Noto Sans SC", "SimHei"]
_avail = {f.name for f in font_manager.fontManager.ttflist}
_cn = next((f for f in _CN if f in _avail), "sans-serif")
plt.rcParams.update({
    "font.family": _cn, "axes.unicode_minus": False,
    "figure.dpi": 110, "savefig.dpi": 150, "font.size": 12,
})

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
FIG = os.path.join(ROOT, "figures")
RES = os.path.join(ROOT, "results")
os.makedirs(FIG, exist_ok=True)

BLUE, GREEN, RED, GRAY, ORANGE = "#2471a3", "#229954", "#c0392b", "#95a5a6", "#e67e22"


def load_latest_json(algo: str) -> dict:
    """加载某个算法最新的结果 JSON。"""
    files = sorted(glob.glob(os.path.join(RES, f"{algo}_*.json")))
    if not files:
        return None
    with open(files[-1], encoding="utf-8") as f:
        return json.load(f)


# ── B+树 ──────────────────────────────────────────────

def fig_btree():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 左：树高 vs n
    n_list = [1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9]
    fanout = 174
    heights = [max(1, math.ceil(math.log(n, fanout))) for n in n_list]
    ax1.plot(n_list, heights, "o-", color=BLUE, lw=2)
    ax1.set_xscale("log")
    ax1.set_xlabel("数据量 N")
    ax1.set_ylabel("树高（IO 次数）")
    ax1.set_title("B+树高 vs 数据量")
    ax1.grid(True, alpha=0.3)

    # 右：存储线性增长
    n_list2 = [1e5, 1e6, 1e7, 1e8]
    storage_gb = [n * 111 / 1024**3 for n in n_list2]  # 111 B/record
    ax2.plot(n_list2, storage_gb, "s-", color=GREEN, lw=2)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("数据量 N")
    ax2.set_ylabel("存储（GB）")
    ax2.set_title("B+树存储 vs 数据量")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig-btree.png"), bbox_inches="tight")
    plt.close()


# ── LSM-tree ──────────────────────────────────────────

def fig_lsm():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 左：写放大 vs 层数
    layers = range(1, 11)
    wa_tiering = [10 * L / 2 for L in layers]
    wa_leveling = [10 * L / 2 * 1.2 for L in layers]
    ax1.plot(layers, wa_tiering, "o-", color=BLUE, label="Tiering")
    ax1.plot(layers, wa_leveling, "s-", color=RED, label="Leveling")
    ax1.set_xlabel("层数 L")
    ax1.set_ylabel("写放大")
    ax1.set_title("LSM-tree 写放大 vs 层数")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 右：空间放大对比
    size_ratios = [4, 6, 8, 10, 12, 16]
    sa_tiering = [t / 2 for t in size_ratios]
    sa_leveling = [1 + 1 / t for t in size_ratios]
    ax2.plot(size_ratios, sa_tiering, "o-", color=BLUE, label="Tiering")
    ax2.plot(size_ratios, sa_leveling, "s-", color=GREEN, label="Leveling")
    ax2.set_xlabel("Size Ratio")
    ax2.set_ylabel("空间放大")
    ax2.set_title("LSM-tree 空间放大 vs Size Ratio")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig-lsm.png"), bbox_inches="tight")
    plt.close()


# ── 哈希索引 ──────────────────────────────────────────

def fig_hash():
    fig, ax = plt.subplots(figsize=(8, 5))

    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    # 拉链法：1 + α/2
    chaining = [1 + a / 2 for a in alphas]
    # 线性探测：(1 + 1/(1-α)²) / 2
    linear = [(1 + 1 / (1 - a) ** 2) / 2 for a in alphas]
    # 双重哈希：-ln(1-α)/α
    double = [-math.log(1 - a) / a for a in alphas]

    ax.plot(alphas, chaining, "o-", color=BLUE, label="拉链法")
    ax.plot(alphas, linear, "s-", color=RED, label="线性探测")
    ax.plot(alphas, double, "^-", color=GREEN, label="双重哈希")
    ax.set_xlabel("负载因子 α")
    ax.set_ylabel("平均探测次数")
    ax.set_title("哈希表平均探测次数 vs 负载因子")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig-hash.png"), bbox_inches="tight")
    plt.close()


# ── 布隆过滤器 ────────────────────────────────────────

def fig_bloom():
    data = load_latest_json("bloom_filter")
    if not data:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 左：不同 bits_per_key 的假阳性率
    bpk_data = data.get("fpr_vs_bits_per_key", [])
    if bpk_data:
        bpk = [r["m"] // r["n"] for r in bpk_data]
        actual_fpr = [r["actual_fpr"] for r in bpk_data]
        theo_fpr = [r["theoretical_fpr"] for r in bpk_data]
        ax1.plot(bpk, actual_fpr, "o-", color=BLUE, label="实测")
        ax1.plot(bpk, theo_fpr, "s--", color=RED, label="理论")
        ax1.set_xlabel("bits/key")
        ax1.set_ylabel("假阳性率")
        ax1.set_title("布隆过滤器假阳性率")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # 右：不同 k 的假阳性率
    k_data = data.get("fpr_vs_k", [])
    if k_data:
        ks = [r["k"] for r in k_data]
        actual_k = [r["actual_fpr"] for r in k_data]
        theo_k = [r["theoretical_fpr"] for r in k_data]
        ax2.plot(ks, actual_k, "o-", color=BLUE, label="实测")
        ax2.plot(ks, theo_k, "s--", color=RED, label="理论")
        optimal_k = data.get("optimal_k")
        if optimal_k:
            ax2.axvline(x=optimal_k, color=GREEN, ls=":", label=f"最优 k={optimal_k}")
        ax2.set_xlabel("哈希函数数 k")
        ax2.set_ylabel("假阳性率")
        ax2.set_title("不同 k 的假阳性率")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig-bloom.png"), bbox_inches="tight")
    plt.close()


# ── 跳表 ──────────────────────────────────────────────

def fig_skiplist():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 左：层数 vs n
    n_list = [1e3, 1e4, 1e5, 1e6, 1e7, 1e8]
    levels = [max(1, math.ceil(math.log(n, 2))) for n in n_list]
    ax1.plot(n_list, levels, "o-", color=BLUE)
    ax1.set_xscale("log")
    ax1.set_xlabel("数据量 N")
    ax1.set_ylabel("最大层数")
    ax1.set_title("跳表层数 vs 数据量")
    ax1.grid(True, alpha=0.3)

    # 右：比较次数 vs n
    comparisons = [2 * L for L in levels]
    ax2.plot(n_list, comparisons, "s-", color=GREEN)
    ax2.set_xscale("log")
    ax2.set_xlabel("数据量 N")
    ax2.set_ylabel("期望比较次数")
    ax2.set_title("跳表查找比较次数")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig-skiplist.png"), bbox_inches="tight")
    plt.close()


# ── Join ──────────────────────────────────────────────

def fig_join():
    fig, ax = plt.subplots(figsize=(9, 5))

    # NLJ I/O vs Hash Join I/O
    n_r = np.array([1000, 10000, 100000, 1000000])
    n_s = 10000
    page_size = 4096
    row_size = 100

    nlj_io = n_r * (n_s * row_size / page_size)
    hj_io = (n_r * row_size / page_size) + (n_s * row_size / page_size)

    ax.plot(n_r, nlj_io, "o-", color=RED, label="Nested Loop Join")
    ax.plot(n_r, hj_io, "s-", color=BLUE, label="Hash Join")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("表 R 行数")
    ax.set_ylabel("I/O 次数")
    ax.set_title(f"Join 算法 I/O 对比（S={n_s:,} 行）")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig-join.png"), bbox_inches="tight")
    plt.close()


# ── 外部排序 ──────────────────────────────────────────

def fig_external_sort():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 左：归并轮数 vs fan_in
    fan_ins = [2, 4, 8, 16, 32, 64, 128, 256]
    num_runs = 1000
    rounds = [max(1, math.ceil(math.log(num_runs, f))) for f in fan_ins]
    ax1.plot(fan_ins, rounds, "o-", color=BLUE)
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("归并路数 fan_in")
    ax1.set_ylabel("归并轮数")
    ax1.set_title(f"归并轮数（{num_runs} 个有序块）")
    ax1.grid(True, alpha=0.3)

    # 右：I/O vs 数据量
    sizes_gb = [1, 10, 100, 1000]
    memory_gb = 1
    io_reads = [2 * s / memory_gb * (memory_gb * 1e9 / 4096) for s in sizes_gb]
    ax2.plot(sizes_gb, io_reads, "s-", color=GREEN)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("数据量（GB）")
    ax2.set_ylabel("I/O 读次数")
    ax2.set_title("外部排序 I/O vs 数据量")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig-external-sort.png"), bbox_inches="tight")
    plt.close()


# ── 向量索引 ──────────────────────────────────────────

def fig_vector_index():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 左：存储对比（百万向量 384 维）
    names = ["FLAT", "SQ8", "PQ", "IVF-PQ", "HNSW"]
    sizes = [1465, 366, 8, 16, 1595]  # MiB
    colors = [BLUE, GREEN, ORANGE, RED, GRAY]
    ax1.bar(names, sizes, color=colors)
    ax1.set_ylabel("存储（MiB）")
    ax1.set_title("百万向量 384 维存储对比")
    ax1.set_yscale("log")
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=15)

    # 右：延迟对比
    names2 = ["FLAT", "IVF", "HNSW"]
    latency = [21333, 167, 26]  # ms
    colors2 = [RED, ORANGE, BLUE]
    ax2.bar(names2, latency, color=colors2)
    ax2.set_ylabel("查询延迟（ms）")
    ax2.set_title("百万向量查询延迟对比")
    ax2.set_yscale("log")

    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig-vector-index.png"), bbox_inches="tight")
    plt.close()


# ── 倒排索引 ──────────────────────────────────────────

def fig_inverted_index():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 左：TF-IDF vs BM25 延迟
    methods = ["TF-IDF", "BM25"]
    latency = [1.75, 3.25]  # μs
    ax1.bar(methods, latency, color=[BLUE, GREEN])
    ax1.set_ylabel("查询延迟（μs）")
    ax1.set_title("倒排索引查询延迟")

    # 右：IDF 分布
    buckets = ["df_1", "df_2_10", "df_11_100", "df_101_1000"]
    avg_idf = [9.2, 6.9, 4.6, 2.3]  # 近似值
    ax2.bar(buckets, avg_idf, color=ORANGE)
    ax2.set_ylabel("平均 IDF")
    ax2.set_title("IDF 分布（不同 DF 桶）")
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=15)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig-inverted-index.png"), bbox_inches="tight")
    plt.close()


# ── 列式存储 ──────────────────────────────────────────

def fig_columnar():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 左：压缩比 vs 基数
    cardinalities = [2, 10, 100, 1000, 10000]
    ratios = [32, 12, 4.6, 2.1, 1.2]
    ax1.plot(cardinalities, ratios, "o-", color=BLUE)
    ax1.set_xscale("log")
    ax1.set_ylabel("压缩比")
    ax1.set_xlabel("列基数")
    ax1.set_title("列式存储压缩比 vs 基数")
    ax1.grid(True, alpha=0.3)

    # 右：扫描加速
    cols = ["1 列", "3 列", "全列"]
    speedup = [34, 18, 1.1]
    ax2.bar(cols, speedup, color=[GREEN, ORANGE, GRAY])
    ax2.set_ylabel("加速倍数（vs 行式）")
    ax2.set_title("列式扫描加速")

    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig-columnar.png"), bbox_inches="tight")
    plt.close()


# ── BM25 ───────────────────────────────────────────────

def fig_bm25():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 左：查询延迟对比（实测 vs 理论）
    methods = ["TF-IDF", "BM25 朴素", "BM25 预计算", "BM25+WAND"]
    actual = [76.20, 161.40, 63.98, 151.81]
    theoretical = [0.75, 1.25, 0.95, 0.45]
    x = range(len(methods))
    width = 0.35
    ax1.bar([i - width/2 for i in x], actual, width, color=BLUE, label="实测")
    ax1.bar([i + width/2 for i in x], theoretical, width, color=GREEN, label="理论")
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=15, ha="right")
    ax1.set_ylabel("查询延迟（μs）")
    ax1.set_title("BM25 各实现查询延迟对比")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    # 右：TF 饱和曲线（理论）
    k1 = 1.2
    b = 0.75
    tf_range = list(range(1, 51))
    # TF 权重 = tf * (k1+1) / (tf + k1)
    def tf_w(tf):
        return tf * (k1 + 1.0) / (tf + k1)
    # 完整得分（IDF 归一化后）= TF 权重 / 长度归一化因子
    norm_avgdl = 1.0 - b + b * 1.0    # = 1.0
    norm_half  = 1.0 - b + b * 0.5    # = 0.625
    bm25_scores  = [tf_w(tf) / norm_avgdl for tf in tf_range]
    bm25_short   = [tf_w(tf) / norm_half  for tf in tf_range]
    tfidf_scores = [float(tf) for tf in tf_range]  # TF-IDF TF 项 = TF

    ax2.plot(tf_range, tfidf_scores, "s--", color=GRAY, label="TF-IDF (TF 项)", lw=1.5)
    ax2.plot(tf_range, bm25_scores, "o-", color=BLUE, label="BM25 (DL=AVGDL)", lw=2)
    ax2.plot(tf_range, bm25_short, "^--", color=RED, label="BM25 (DL=0.5×AVGDL)", lw=2)
    ax2.set_xlabel("TF（词条在文档中出现次数）")
    ax2.set_ylabel("TF 权重（IDF 已归一化）")
    ax2.set_title("BM25 TF 饱和曲线")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig-bm25.png"), bbox_inches="tight")
    plt.close()


# ── 主函数 ────────────────────────────────────────────

def main():
    generators = [
        fig_btree, fig_lsm, fig_hash, fig_bloom, fig_skiplist,
        fig_join, fig_external_sort, fig_vector_index,
        fig_inverted_index, fig_bm25, fig_columnar,
    ]
    for gen in generators:
        try:
            gen()
            print(f"[✅] {gen.__name__}")
        except Exception as e:
            print(f"[❌] {gen.__name__}: {e}")

    print(f"\n所有图表已生成到 {FIG}")


if __name__ == "__main__":
    main()
