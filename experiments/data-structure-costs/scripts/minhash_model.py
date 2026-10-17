# -*- coding: utf-8 -*-
"""
MinHash（最小哈希）理论性能模型

MinHash 用于估算两个集合的 Jaccard 相似度 J(A,B) = |A∩B| / |A∪B|。
核心性质：对集合做一个随机置换，取最小哈希值，两个集合的最小哈希相等的
概率恰好等于它们的 Jaccard 相似度。用 k 个独立置换得到 k 个最小哈希，
其一致比例就是 J 的无偏估计。

本模块只做闭式解计算，不依赖任何 ANN 库。核心公式：

1. 估计量
      Ĵ = (# 个最小哈希一致) / k
      E[Ĵ] = J,  Var[Ĵ] = J(1-J)/k
   这就是 MinHash 作为无偏估计的全部性质。

2. 标准误差
      σ = sqrt(J(1-J)/k)
   J=0.5 时方差最大，σ = 1/sqrt(4k) = 0.5/sqrt(k)。实践中常说的
   "σ = 1/√k" 是 J→0 或 J→1 极端情形的上界估计。

3. LSH band 技术（S-curve）
      P(候选对) = 1 - (1 - J^r)^b
   b 个 band，每 band r 行。两个集合在某 band 里所有 r 行都相同的概率是 J^r，
   b 个 band 做 OR。S-curve 是 sigmoid，把相似对与不相似对分开。

4. band 参数选择
      拐点条件：J^r = 1/b，即 b = 1/J^r。
      配合 b × r ≈ k，解 J^r = r/k，取使偏差最小的 r，b = round(k/r)。
      拐点处 P(候选) = 1 - (1-1/b)^b → 1 - 1/e ≈ 63.2%。

应用：文档去重、网页近似查重、推荐系统协同过滤、聚类。
"""
from dataclasses import dataclass
import math


@dataclass
class MinHashSpec:
    """MinHash 参数"""
    n: int = 1_000_000          # 集合数量
    avg_set_size: int = 100     # 平均集合大小（元素数）
    k: int = 128                # MinHash 签名长度（置换数）
    b: int = 0                  # band 数（0 表示自动标定到阈值）
    r: int = 0                  # 每 band 行数（0 表示自动）
    j_threshold: float = 0.8    # Jaccard 阈值（用于 band 拐点标定）


@dataclass
class MinHashMetrics:
    n: int                      # 集合数量
    avg_set_size: int           # 平均集合大小
    k: int                      # 签名长度
    b: int                      # band 数
    r: int                      # 每 band 行数
    j_threshold: float          # 标定阈值
    est_std: float              # 估计标准差 σ = sqrt(J(1-J)/k) @ J=threshold
    band_prob_at_j: float       # S-curve 在阈值处的概率（≈0.632 为拐点）
    sig_bytes: int              # 签名总字节数（k bits per set）
    compression_ratio: float    # 相对原始集合的压缩比（估算）
    pairwise_comparisons: int   # 暴力两两比较次数 = n(n-1)/2


def estimate_std(k: int, j: float) -> float:
    """MinHash 估计标准差。σ = sqrt(J(1-J)/k)。"""
    return math.sqrt(j * (1.0 - j) / k)


def band_prob(j: float, r: int, b: int) -> float:
    """S-curve：Jaccard 相似度 J 时成为候选对的概率。1 - (1 - J^r)^b。"""
    return 1.0 - (1.0 - j ** r) ** b


def compute_optimal_banding(k: int, j_threshold: float) -> tuple:
    """使 S-curve 拐点落在 J_threshold 的 (r, b)。

    拐点条件 J^r = 1/b，配合 b × r ≤ k（多余行直接不用）。
    枚举 b，取 r = k // b，选使 |J^r - 1/b| 最小的组合。
    约束 b×r ≤ k 很重要：否则最后一个 band 会被静默截断，
    该 band 的有效行数变少，候选概率被系统性抬高。
    """
    best = (1, 1, float("inf"))
    for b in range(1, k + 1):
        r = k // b
        if r < 1:
            break
        err = abs(j_threshold ** r - 1.0 / b)
        if err < best[2]:
            best = (r, b, err)
    return best[0], best[1]


def compute(spec: MinHashSpec) -> MinHashMetrics:
    """给定参数，递推全部指标。"""
    if spec.b == 0 or spec.r == 0:
        r, b = compute_optimal_banding(spec.k, spec.j_threshold)
    else:
        b, r = spec.b, spec.r

    est_std = estimate_std(spec.k, spec.j_threshold)
    band_prob_j = band_prob(spec.j_threshold, r, b)

    # 签名：k 比特/集合
    sig_bytes = spec.n * ((spec.k + 7) // 8)

    # 压缩比：假设元素用 4 字节整数存，原始 ≈ n × avg_set_size × 4 bytes
    raw_bytes = spec.n * spec.avg_set_size * 4
    compression = raw_bytes / sig_bytes

    return MinHashMetrics(
        n=spec.n,
        avg_set_size=spec.avg_set_size,
        k=spec.k,
        b=b,
        r=r,
        j_threshold=spec.j_threshold,
        est_std=est_std,
        band_prob_at_j=band_prob_j,
        sig_bytes=sig_bytes,
        compression_ratio=compression,
        pairwise_comparisons=spec.n * (spec.n - 1) // 2,
    )


def _selftest():
    """公式自洽性检查"""
    spec = MinHashSpec(n=1_000_000, avg_set_size=100, k=128, j_threshold=0.8)
    m = compute(spec)

    # 标准差随 k 增大而减小
    m_k256 = compute(MinHashSpec(n=1_000_000, avg_set_size=100, k=256, j_threshold=0.8))
    assert m_k256.est_std < m.est_std, "k 越大估计标准差应越小"

    # J=0.5 时方差最大（σ 最大）
    std_0_5 = estimate_std(128, 0.5)
    std_0_9 = estimate_std(128, 0.9)
    std_0_1 = estimate_std(128, 0.1)
    assert std_0_5 > std_0_9 and std_0_5 > std_0_1, "J=0.5 附近方差应最大"

    # S-curve 在 J=0 和 J=1 的行为
    assert band_prob(0.0, 4, 16) == 0.0, "J=0 时不应成为候选"
    assert band_prob(1.0, 4, 16) == 1.0, "J=1 时必须成为候选"

    # S-curve 单调递增
    assert band_prob(0.3, 4, 16) < band_prob(0.6, 4, 16) < band_prob(0.9, 4, 16), \
        "S-curve 应随 J 单调递增"

    # band 数在合理范围
    assert 1 <= m.b <= m.k, f"band 数应在 1 到 k 之间，实际 {m.b}"
    assert 1 <= m.r, "每 band 行数应 ≥ 1"

    # 拐点标定：自动 (r, b) 应使 S-curve 拐点落在 J_threshold 上
    assert abs(m.band_prob_at_j - (1.0 - 1.0 / math.e)) < 0.08, \
        f"拐点概率应接近 1-1/e=0.632，实际 {m.band_prob_at_j:.4f}"

    # r 越大 S-curve 越陡：阈值以下的 J 处候选概率应更低
    m_wide = compute(MinHashSpec(n=1_000_000, avg_set_size=100, k=128, r=8, b=16, j_threshold=0.8))
    assert band_prob(0.5, m.r, m.b) < band_prob(0.5, m_wide.r, m_wide.b), \
        "r 更大时阈值以下的候选概率应更低"

    print("selftest 全部通过")
    print(f"\nn={m.n:,}  avg_set={m.avg_set_size}  k={m.k}  b={m.b}  r={m.r}")
    print(f"σ(J={m.j_threshold})={m.est_std:.4f}  拐点概率={m.band_prob_at_j:.4f}  (1-1/e=0.6321)")
    print(f"J=0.5 处候选概率={band_prob(0.5, m.r, m.b):.6f}  (宽带 r=8: {band_prob(0.5, 8, 16):.6f})")
    print(f"签名={m.sig_bytes/1024/1024:.2f} MiB  压缩比≈{m.compression_ratio:.1f}x")
    print(f"暴力两两比较={m.pairwise_comparisons:,}")


if __name__ == "__main__":
    _selftest()
