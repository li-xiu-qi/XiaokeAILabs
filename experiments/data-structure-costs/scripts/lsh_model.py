# -*- coding: utf-8 -*-
"""
LSH（Locality-Sensitive Hashing，局部敏感哈希）理论性能模型

LSH 是近似最近邻（ANN）的经典算法族。核心思想：用一族哈希函数把高维向量
映射到桶里，使得相似向量以高概率落进同一个桶，从而把 O(n) 的线性扫描
降到 O(n^ρ)（ρ<1），其中 ρ 是 LSH 家族的内在指数，与具体参数无关。

本模块只做闭式解计算，不依赖任何 ANN 库。核心公式：

1. 单哈希碰撞概率（随机投影，SimHash 风格）
      p(s) = 1 - θ/π,  θ = arccos(s),  s 为余弦相似度
   推导：h(v) = sign(a·v)，a 为随机高斯向量。两向量夹角 θ 时，
   两个投影同号的概率 = 1 - θ/π。

2. S-curve（AND-OR 组合）
      P(candidate | s) = 1 - (1 - p(s)^k)^L
   k 个哈希做 AND（同 band 全中才算命中），L 个 band 做 OR（任一 band 命中即为候选）。
   S-curve 是 sigmoid 形状：相似度远离阈值时概率急剧跌向 0 或升向 1。

3. 查询复杂度指数 ρ
      ρ = ln(1/p1) / ln(1/p2)
   其中 (r1, r2, p1, p2)-sensitive：相似度 ≥ r1 时碰撞概率 ≥ p1，
   相似度 ≤ r2 时碰撞概率 ≤ r2。取 r1 = 阈值 s1，r2 = 对侧相似度 s2。
   ρ<1 才有亚线性收益。

4. 参数选择（Indyk-Motwani 经典标定）
      k = log(n) / log(1/p2)
      L = n^ρ
   这两个式子合起来满足拐点条件 p1^k = 1/L，即 S-curve 的陡升段正好
   落在相似度阈值 s1 上。此时：
     - 阈值处召回 = 1 - (1 - 1/L)^L → 1 - 1/e ≈ 63.2%（这是 LSH 的理论保证值）
     - 每次查询的期望误候选数 = n × P(candidate | s2) = n^(1-ρ)，亚线性

5. 加速比 = n^(1-ρ)，即相对线性扫描减少的计算量。
"""
from dataclasses import dataclass
import math


@dataclass
class LSHSpec:
    """LSH 参数"""
    n: int = 1_000_000          # 语料规模
    d: int = 128                # 向量维度
    s1: float = 0.80            # 目标相似度阈值（≥ s1 希望进候选集）
    s2: float = 0.40            # 对侧相似度（≤ s2 希望排除）
    k: int = 0                  # 每 band 的哈希数（0 表示按公式自动算）
    L: int = 0                  # band 数（0 表示按公式自动算）
    num_queries: int = 1_000    # 查询批大小（用于推导每秒查询能力）


@dataclass
class LSHMetrics:
    n: int                      # 语料规模
    k: int                      # 每 band 哈希数
    L: int                      # band 数
    rho: float                  # 复杂度指数 ρ
    p1: float                   # s1 处单哈希碰撞概率
    p2: float                   # s2 处单哈希碰撞概率
    prob_at_s1: float           # S-curve 在 s1 处的候选概率（召回）
    prob_at_s2: float           # S-curve 在 s2 处的候选概率
    total_hash_fns: int         # 总哈希函数数 = k × L
    table_bytes: int            # 哈希表组总字节数（n 项 × L 个 k 位签名）
    scan_speedup: float         # 相对线性扫描的加速比 = n^(1-ρ)
    false_candidates_per_query: float  # 每次查询的期望误候选数 ≈ n^(1-ρ)


def collision_prob(s: float) -> float:
    """随机投影 LSH 的单哈希碰撞概率。p = 1 - θ/π, θ = arccos(s)。"""
    return 1.0 - math.acos(max(-1.0, min(1.0, s))) / math.pi


def s_curve(p: float, k: int, L: int) -> float:
    """S-curve：给定单哈希碰撞概率 p，返回 AND-OR 组合后的候选概率。"""
    return 1.0 - (1.0 - p ** k) ** L


def compute_rho(p1: float, p2: float) -> float:
    """复杂度指数 ρ = ln(1/p1) / ln(1/p2)。p1 大 p2 小，ρ<1。"""
    return math.log(1.0 / p1) / math.log(1.0 / p2)


def compute_k(n: int, p2: float) -> int:
    """每 band 哈希数。k = log(n) / log(1/p2)。"""
    return max(1, round(math.log(n) / math.log(1.0 / p2)))


def compute_L(n: int, rho: float) -> int:
    """band 数。L = n^ρ。"""
    return max(1, int(round(n ** rho)))


def compute(spec: LSHSpec) -> LSHMetrics:
    """给定参数，递推全部指标。"""
    p1 = collision_prob(spec.s1)
    p2 = collision_prob(spec.s2)
    rho = compute_rho(p1, p2)

    if spec.k == 0:
        k = compute_k(spec.n, p2)
    else:
        k = spec.k

    if spec.L == 0:
        L = compute_L(spec.n, rho)
    else:
        L = spec.L

    prob_at_s1 = s_curve(p1, k, L)
    prob_at_s2 = s_curve(p2, k, L)

    # 每个向量在每个 band 里存一个 k 位签名，共 L 个 band
    table_bytes = spec.n * L * ((k + 7) // 8)

    # 期望误候选数：以 s2（对侧相似度）处的候选概率 × n 估计，
    # 理论值为 n^(1-ρ)。这是每次查询需要真正算距离的向量数。
    false_candidates = spec.n * prob_at_s2

    return LSHMetrics(
        n=spec.n,
        k=k,
        L=L,
        rho=rho,
        p1=p1,
        p2=p2,
        prob_at_s1=prob_at_s1,
        prob_at_s2=prob_at_s2,
        total_hash_fns=k * L,
        table_bytes=table_bytes,
        scan_speedup=spec.n ** (1.0 - rho),
        false_candidates_per_query=false_candidates,
    )


def _selftest():
    """公式自洽性检查"""
    spec = LSHSpec(n=1_000_000, d=128, s1=0.8, s2=0.4)
    m = compute(spec)

    # 碰撞概率单调递增，边界值正确
    assert collision_prob(0.0) < collision_prob(0.9), "相似度越高碰撞概率应越高"
    assert abs(collision_prob(1.0) - 1.0) < 1e-9, "完全相同向量碰撞概率应为 1"
    assert abs(collision_prob(0.0) - 0.5) < 1e-9, "正交向量碰撞概率应为 0.5"

    # ρ < 1（亚线性收益）
    assert 0 < m.rho < 1, f"ρ 应在 0 和 1 之间，实际 {m.rho}"

    # S-curve 在阈值附近陡峭切换
    assert m.prob_at_s1 > 0.5 > m.prob_at_s2, "S-curve 应在阈值附近陡峭切换"

    # band 数随 n 增长
    m_big = compute(LSHSpec(n=10_000_000, d=128, s1=0.8, s2=0.4))
    assert m_big.L > m.L, "n 越大 band 数应越多"

    # 拐点条件：自动参数应使 S-curve 的陡升段正好落在 s1 上（p1^k ≈ 1/L）
    knee_err = abs(m.p1 ** m.k - 1.0 / m.L) / (1.0 / m.L)
    assert knee_err < 0.05, f"自动参数应满足拐点条件 p1^k = 1/L，实际偏差 {knee_err:.2%}"

    # 阈值处召回应接近经典 LSH 保证值 1 - 1/e
    assert abs(m.prob_at_s1 - (1.0 - 1.0 / math.e)) < 0.01, \
        f"阈值处召回应接近 1-1/e=0.632，实际 {m.prob_at_s1:.4f}"

    # 期望误候选数应接近 n^(1-ρ)
    ratio = m.false_candidates_per_query / spec.n ** (1.0 - m.rho)
    assert 0.5 < ratio < 2.0, f"误候选数应接近 n^(1-ρ)，实际比值 {ratio:.3f}"

    # k 增大（保持拐点）会减少误候选数：n × (p2/p1)^k 随 k 递减
    m_k_small = compute(LSHSpec(n=1_000_000, d=128, s1=0.8, s2=0.4, k=20, L=round(1.0 / m.p1 ** 20)))
    assert m.false_candidates_per_query < m_k_small.false_candidates_per_query, \
        "k 更大时误候选数应更少"

    print("selftest 全部通过")
    print(f"\nn={m.n:,}  d={spec.d}  k={m.k}  L={m.L}")
    print(f"ρ={m.rho:.4f}  (线性扫描加速比 {m.scan_speedup:.1f}x)")
    print(f"拐点条件 p1^k={m.p1**m.k:.6f}  1/L={1.0/m.L:.6f}")
    print(f"s1={spec.s1} 处候选概率(召回)={m.prob_at_s1:.4f}  (经典值 1-1/e=0.6321)")
    print(f"s2={spec.s2} 处候选概率={m.prob_at_s2:.6f}")
    print(f"期望误候选数={m.false_candidates_per_query:,.0f}  (n^(1-ρ)={spec.n**(1-m.rho):,.0f})")
    print(f"总哈希函数数={m.total_hash_fns:,}  哈希表字节={m.table_bytes/1024/1024:.2f} MiB")


if __name__ == "__main__":
    _selftest()
