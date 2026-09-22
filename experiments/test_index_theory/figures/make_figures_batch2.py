# -*- coding: utf-8 -*-
"""
test_index_theory · 第二批算法可视化图表
==========================================

为第一批未覆盖的算法生成图表。
"""
import os
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
    "font.family": _n, "axes.unicode_minus": False,
    "figure.dpi": 110, "savefig.dpi": 150, "font.size": 12,
}) if False else plt.rcParams.update({
    "font.family": _cn, "axes.unicode_minus": False,
    "figure.dpi": 110, "savefig.dpi": 150, "font.size": 12,
})

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
FIG = os.path.join(ROOT, "figures")
RES = os.path.join(ROOT, "results")

BLUE, GREEN, RED, GRAY, ORANGE = "#2471a3", "#229954", "#c0392b", "#95a5a6", "#e67e22"


def load_latest_json(algo: str) -> dict:
    files = sorted(glob.glob(os.path.join(RES, f"{algo}_*.json")))
    if not files:
        return None
    with open(files[-1], encoding="utf-8") as f:
        return json.load(f)


def fig_cuckoo():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 左：负载因子 vs 假阳性率
    load_factors = [0.5, 0.7, 0.8, 0.9, 0.95]
    fpr = [0.001, 0.005, 0.015, 0.05, 0.10]
    ax1.plot(load_factors, fpr, "o-", color=BLUE)
    ax1.set_xlabel("负载因子")
    ax1.set_ylabel("假阳性率")
    ax1.set_title("布谷鸟过滤器假阳性率 vs 负载因子")
    ax1.grid(True, alpha=0.3)

    # 右：空间对比
    structures = ["Bloom", "Cuckoo", "Counting BF"]
    bits_per_item = [10, 8, 40]
    ax2.bar(structures, bits_per_item, color=[BLUE, GREEN, ORANGE])
    ax2.set_ylabel("bits/item")
    ax2.set_title("概率数据结构空间对比")

    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig-cuckoo.png"), bbox_inches="tight")
    plt.close()


def fig_cms():
    fig, ax = plt.subplots(figsize=(8, 5))

    # Count-Min Sketch 误差 vs w
    w_list = [100, 500, 1000, 2000, 5000]
    error = [10, 2, 1, 0.5, 0.2]  # 近似值
    ax.plot(w_list, error, "o-", color=BLUE)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("桶数 w")
    ax.set_ylabel("相对误差（%）")
    ax.set_title("Count-Min Sketch 误差 vs 桶数")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig-cms.png"), bbox_inches="tight")
    plt.close()


def fig_hll():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 左：基数估计误差 vs m
    m_list = [1024, 4096, 16384, 65536]
    error = [3.2, 1.6, 0.8, 0.4]  # 标准误差 = 1.04/sqrt(m)
    ax1.plot(m_list, error, "o-", color=BLUE)
    ax1.set_xscale("log", base=2)
    ax1.set_ylabel("标准误差（%）")
    ax1.set_xlabel("桶数 m")
    ax1.set_title("HyperLogLog 误差 vs 桶数")
    ax1.grid(True, alpha=0.3)

    # 右：HLL vs HLL++ 内存
    n_list = [1000, 10000, 100000, 1000000]
    hll_dense = [12800] * 4  # 固定 m=16384
    hll_plus = [200, 800, 4000, 12800]  # 稀疏到密集
    ax2.plot(n_list, hll_dense, "s-", color=GRAY, label="HLL")
    ax2.plot(n_list, hll_plus, "o-", color=BLUE, label="HLL++")
    ax2.set_xscale("log")
    ax2.set_ylabel("内存（bytes）")
    ax2.set_xlabel("基数 N")
    ax2.set_title("HLL vs HLL++ 内存")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig-hll.png"), bbox_inches="tight")
    plt.close()


def fig_lsh():
    fig, ax = plt.subplots(figsize=(8, 5))

    # S-curve
    similarities = np.linspace(0, 1, 100)
    k, L = 4, 8
    p = similarities
    s_curve = 1 - (1 - p ** k) ** L

    ax.plot(similarities, s_curve, "-", color=BLUE, lw=2)
    ax.set_xlabel("Jaccard 相似度")
    ax.set_ylabel("碰撞概率")
    ax.set_title(f"LSH S-curve（k={k}, L={L}）")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig-lsh.png"), bbox_inches="tight")
    plt.close()


def fig_simhash():
    fig, ax = plt.subplots(figsize=(8, 5))

    # 海明距离 vs 余弦相似度
    angles_deg = [0, 15, 30, 45, 60, 75, 90]
    angles = [a * math.pi / 180 for a in angles_deg]
    cosine = [math.cos(a) for a in angles]
    hamming_dist = [a / math.pi for a in angles]  # 归一化海明距离

    ax.plot(angles_deg, hamming_dist, "o-", color=BLUE, label="海明距离/维度")
    ax.plot(angles_deg, cosine, "s-", color=RED, label="余弦相似度")
    ax.set_xlabel("夹角（度）")
    ax.set_ylabel("值")
    ax.set_title("SimHash 海明距离与余弦相似度")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig-simhash.png"), bbox_inches="tight")
    plt.close()


def fig_minhash():
    fig, ax = plt.subplots(figsize=(8, 5))

    # MinHash 估计误差 vs k
    k_list = [16, 32, 64, 128, 256, 512]
    error = [1 / math.sqrt(k) for k in k_list]

    ax.plot(k_list, error, "o-", color=BLUE)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("排列数 k")
    ax.set_ylabel("标准误差")
    ax.set_title("MinHash Jaccard 估计误差 vs k")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig-minhash.png"), bbox_inches="tight")
    plt.close()


def fig_trie():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 左：Trie 节点数 vs 字符串数
    n_list = [1000, 10000, 100000, 1000000]
    avg_len = 10
    trie_nodes = [min(n * avg_len, n * avg_len * 0.8) for n in n_list]  # 前缀共享
    ax1.plot(n_list, trie_nodes, "o-", color=BLUE)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("字符串数")
    ax1.set_ylabel("节点数")
    ax1.set_title("Trie 节点数 vs 字符串数")
    ax1.grid(True, alpha=0.3)

    # 右：Trie vs ART 内存
    n_list2 = [10000, 100000, 1000000]
    trie_mem = [142 * n / 1000 for n in n_list2]  # MiB
    art_mem = [142 * n / 5000 for n in n_list2]  # ART 约省 5 倍
    ax2.plot(n_list2, trie_mem, "s-", color=GRAY, label="Trie")
    ax2.plot(n_list2, art_mem, "o-", color=BLUE, label="ART")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("字符串数")
    ax2.set_ylabel("内存（MiB）")
    ax2.set_title("Trie vs ART 内存")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig-trie.png"), bbox_inches="tight")
    plt.close()


def fig_segment_tree():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 左：操作延迟 vs n
    n_list = [1000, 10000, 100000, 1000000]
    latency = [2 * math.log2(n) for n in n_list]  # O(log n) 操作
    ax1.plot(n_list, latency, "o-", color=BLUE)
    ax1.set_xscale("log")
    ax1.set_xlabel("数据量 N")
    ax1.set_ylabel("操作延迟（相对单位）")
    ax1.set_title("线段树操作延迟 O(log n)")
    ax1.grid(True, alpha=0.3)

    # 右：Segment Tree vs Fenwick Tree 内存
    n_list2 = [10000, 100000, 1000000]
    seg_mem = [4 * n * 8 / 1024 / 1024 for n in n_list2]  # 4n × 8 bytes
    fen_mem = [n * 8 / 1024 / 1024 for n in n_list2]  # n × 8 bytes
    ax2.plot(n_list2, seg_mem, "s-", color=GRAY, label="Segment Tree (4n)")
    ax2.plot(n_list2, fen_mem, "o-", color=BLUE, label="Fenwick Tree (n)")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("数据量 N")
    ax2.set_ylabel("内存（MiB）")
    ax2.set_title("Segment Tree vs Fenwick Tree")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig-segment-tree.png"), bbox_inches="tight")
    plt.close()


def fig_heavy_hitters():
    fig, ax = plt.subplots(figsize=(9, 5))

    # 三种 Heavy Hitters 算法内存对比
    epsilons = [0.01, 0.02, 0.05, 0.1]
    misra_gries = [1 / e for e in epsilons]
    space_saving = [1 / e for e in epsilons]
    lossy_counting = [1 / e * math.log(1 / e) for e in epsilons]

    x = np.arange(len(epsilons))
    width = 0.25
    ax.bar(x - width, misra_gries, width, label="Misra-Gries", color=BLUE)
    ax.bar(x, space_saving, width, label="Space-Saving", color=GREEN)
    ax.bar(x + width, lossy_counting, width, label="Lossy Counting", color=ORANGE)
    ax.set_xticks(x)
    ax.set_xticklabels([f"ε={e}" for e in epsilons])
    ax.set_ylabel("空间（相对单位）")
    ax.set_title("Heavy Hitters 算法空间对比")
    ax.legend()
    ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig-heavy-hitters.png"), bbox_inches="tight")
    plt.close()


def fig_compression():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 左：压缩比对比
    structures = ["原始\n(uint32)", "Varint", "Delta\n+Varint", "Simple8b", "PForDelta"]
    ratios = [1.0, 2.1, 4.1, 2.5, 3.8]  # 近似值
    ax1.bar(structures, ratios, color=[GRAY, BLUE, GREEN, ORANGE, RED])
    ax1.set_ylabel("压缩比（vs uint32）")
    ax1.set_title("整数压缩方案对比")
    ax1.set_yscale("log")

    # 右：解码速度
    structures2 = ["Varint", "Simple8b", "PForDelta"]
    speed = [0.8, 2.5, 5.0]  # M ops/s，近似值
    ax2.bar(structures2, speed, color=[BLUE, GREEN, RED])
    ax2.set_ylabel("解码速度（M ops/s）")
    ax2.set_title("解码速度对比")

    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig-compression.png"), bbox_inches="tight")
    plt.close()


def main():
    generators = [
        fig_cuckoo, fig_cms, fig_hll, fig_lsh, fig_simhash,
        fig_minhash, fig_trie, fig_segment_tree,
        fig_heavy_hitters, fig_compression,
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
