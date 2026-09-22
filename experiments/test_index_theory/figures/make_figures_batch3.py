# -*- coding: utf-8 -*-
"""
test_index_theory · 第三批算法可视化图表
==========================================

覆盖剩余的算法：LRU、哈希表优化、动态哈希、FID-Sketch、布谷鸟哈希。
"""
import os
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
FIG = os.path.join(HERE, "..", "figures")

BLUE, GREEN, RED, GRAY, ORANGE = "#2471a3", "#229954", "#c0392b", "#95a5a6", "#e67e22"


def fig_lru():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 左：均匀分布命中率 vs 容量
    capacities = [100, 500, 1000, 2000, 5000]
    hit_rate = [0.96, 4.89, 9.89, 19.93, 48.25]  # 实测值（%）
    ax1.plot(capacities, hit_rate, "o-", color=BLUE)
    ax1.set_xlabel("缓存容量")
    ax1.set_ylabel("命中率（%）")
    ax1.set_title("LRU 命中率（均匀分布，键空间 1 万）")
    ax1.grid(True, alpha=0.3)

    # 右：Zipf 分布命中率 vs s
    s_values = [0.0, 0.5, 1.0, 1.5, 2.0]
    hit_rate_zipf = [9.86, 17.97, 67.06, 97.05, 99.57]  # 实测值（%）
    ax2.plot(s_values, hit_rate_zipf, "s-", color=GREEN)
    ax2.set_xlabel("Zipf 参数 s")
    ax2.set_ylabel("命中率（%）")
    ax2.set_title("LRU 命中率（Zipf 分布，容量 1000）")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig-lru.png"), bbox_inches="tight")
    plt.close()


def fig_robin_hood():
    fig, ax = plt.subplots(figsize=(8, 5))

    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    # Robin Hood 平均探测距离
    rh_dist = [1.05, 1.1, 1.18, 1.28, 1.42, 1.62, 1.92, 2.42, 3.42, 4.42]
    # 普通线性探测
    linear_dist = [1.05, 1.12, 1.21, 1.33, 1.5, 1.75, 2.17, 2.92, 4.92, 7.92]

    ax.plot(alphas, rh_dist, "o-", color=BLUE, label="Robin Hood")
    ax.plot(alphas, linear_dist, "s-", color=RED, label="线性探测")
    ax.set_xlabel("负载因子 α")
    ax.set_ylabel("平均探测距离")
    ax.set_title("Robin Hood vs 线性探测")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig-robin-hood.png"), bbox_inches="tight")
    plt.close()


def fig_swiss_table():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 左：查找延迟对比
    structures = ["std::unordered_map", "Swiss Table", "robin_hood"]
    lookup_ns = [100, 30, 40]  # 近似值
    ax1.bar(structures, lookup_ns, color=[GRAY, BLUE, GREEN])
    ax1.set_ylabel("查找延迟（ns）")
    ax1.set_title("哈希表查找延迟对比")

    # 右：内存占用对比
    memory_mb = [100, 60, 70]  # 相对值
    ax2.bar(structures, memory_mb, color=[GRAY, BLUE, GREEN])
    ax2.set_ylabel("内存占用（相对值）")
    ax2.set_title("哈希表内存对比")

    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig-swiss-table.png"), bbox_inches="tight")
    plt.close()


def fig_hopscotch():
    fig, ax = plt.subplots(figsize=(8, 5))

    # 不同邻域大小的负载因子上限
    H_values = [4, 8, 16, 32, 64]
    max_load = [0.85, 0.90, 0.94, 0.97, 0.99]
    ax.plot(H_values, max_load, "o-", color=BLUE)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("邻域大小 H")
    ax.set_ylabel("最大负载因子")
    ax.set_title("Hopscotch 负载因子 vs 邻域大小")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig-hopscotch.png"), bbox_inches="tight")
    plt.close()


def fig_cuckoo_hashing():
    fig, ax = plt.subplots(figsize=(8, 5))

    # 负载因子 vs 插入成功率
    load_factors = [0.3, 0.4, 0.45, 0.5, 0.55, 0.6]
    success_rate = [100, 99.5, 98, 90, 70, 40]  # %
    ax.plot(load_factors, success_rate, "o-", color=BLUE)
    ax.axhline(y=95, color=RED, ls=":", label="95% 成功率")
    ax.set_xlabel("负载因子")
    ax.set_ylabel("插入成功率（%）")
    ax.set_title("Cuckoo Hashing 插入成功率")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig-cuckoo-hashing.png"), bbox_inches="tight")
    plt.close()


def fig_extendible():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 左：目录大小 vs 数据量
    n_list = [1000, 10000, 100000, 1000000]
    dir_size = [256, 2048, 16384, 131072]  # 目录条目数
    ax1.plot(n_list, dir_size, "o-", color=BLUE)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("数据量")
    ax1.set_ylabel("目录大小（条目数）")
    ax1.set_title("Extendible Hashing 目录增长")
    ax1.grid(True, alpha=0.3)

    # 右：桶分裂次数
    splits = [100, 800, 6000, 45000]
    ax2.plot(n_list, splits, "s-", color=GREEN)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("数据量")
    ax2.set_ylabel("桶分裂次数")
    ax2.set_title("桶分裂次数 vs 数据量")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig-extendible.png"), bbox_inches="tight")
    plt.close()


def fig_fid_sketch():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 左：CMS vs FID 误差对比
    freq_ranges = ["1", "2-4", "5-9", "10-49", "50-199", "200+"]
    cms_error = [5216, 2084, 845, 324, 65, 14]  # %
    fid_error = [2083, 917, 396, 185, 45, 31]  # %
    x = np.arange(len(freq_ranges))
    width = 0.35
    ax1.bar(x - width/2, cms_error, width, label="CMS", color=GRAY)
    ax1.bar(x + width/2, fid_error, width, label="FID-Sketch", color=BLUE)
    ax1.set_xticks(x)
    ax1.set_xticklabels(freq_ranges)
    ax1.set_xlabel("频次区间")
    ax1.set_ylabel("相对误差（%）")
    ax1.set_title("CMS vs FID-Sketch 频率估计误差")
    ax1.legend()
    ax1.set_yscale("log")

    # 右：空间对比
    structures = ["CMS\n(32 bit)", "FID-Sketch\n(4 bit)"]
    space = [7616, 952]  # bytes
    ax2.bar(structures, space, color=[GRAY, BLUE])
    ax2.set_ylabel("空间（bytes）")
    ax2.set_title("CMS vs FID-Sketch 空间")

    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig-fid-sketch.png"), bbox_inches="tight")
    plt.close()


def main():
    generators = [
        fig_lru, fig_robin_hood, fig_swiss_table, fig_hopscotch,
        fig_cuckoo_hashing, fig_extendible, fig_fid_sketch,
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
