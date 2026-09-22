# -*- coding: utf-8 -*-
"""
SimHash（相似度哈希）理论性能模型

SimHash 用 d 个随机超平面把 d 维向量压成 d 位签名，海明距离近似反映余弦
相似度。它是 Google 网页去重的原始算法（Charikar 2002），也是 LSH 家族里
随机投影那一支的实现方式。

本模块只做闭式解计算，不依赖任何 ANN 库。核心公式：

1. 单比特碰撞概率
      P(bit 相同) = 1 - θ/π,  θ 为两向量夹角
   等价地，P(bit 不同) = θ/π。

2. 期望海明距离
      E[h(x, y)] = d × θ/π
   海明距离与余弦相似度一一对应：夹角越大，海明距离越大。

3. 海明距离的方差
      Var[h] = d × (θ/π) × (1 - θ/π)
      σ = sqrt(Var[h])
   d 越大，相对波动越小，估计越准。

4. 候选集（桶半径 r）
      候选 = {y : h(x, y) ≤ r}
   给定目标相似度 s0（θ0 = arccos(s0)），半径 r = round(d × θ0/π) 时
   召回 ≈ P(Binomial(d, θ0/π) ≤ r)，可用正态近似估算。

5. 存储压缩比
      签名 = n × (num_bits/8) bytes
      原始 float32 = n × d × 4 bytes
      压缩比 = 32d / num_bits
   标准 SimHash 取 num_bits = d，压缩比恰好 32 倍（4 字节 float32 → 1 比特）。
"""
from dataclasses import dataclass
import math


@dataclass
class SimHashSpec:
    """SimHash 参数"""
    n: int = 1_000_000          # 向量数量
    d: int = 128                # 向量维度
    s0: float = 0.80            # 目标相似度阈值
    r: int = 0                  # 桶半径（0 表示自动 = round(num_bits × θ0/π)）
    num_bits: int = 64          # 签名位数（可与 d 不同，用于压缩场景）


@dataclass
class SimHashMetrics:
    n: int                      # 向量数量
    d: int                      # 向量维度
    num_bits: int               # 签名位数
    s0: float                   # 目标相似度
    theta0: float               # arccos(s0)（弧度）
    expected_hamming: float     # 期望海明距离 = num_bits × θ0/π
    hamming_std: float          # 海明距离标准差
    r: int                      # 实际桶半径
    recall_at_r: float          # 半径 r 内的召回率（正态近似）
    sig_bytes: int              # 签名总字节数
    raw_bytes: int              # 原始向量总字节数（float32）
    compression_ratio: float    # 压缩比（原始 / 签名）


def bit_flip_prob(theta: float) -> float:
    """单位比特不同的概率 = θ/π。"""
    return theta / math.pi


def expected_hamming(num_bits: int, theta: float) -> float:
    """期望海明距离 = num_bits × θ/π。"""
    return num_bits * theta / math.pi


def hamming_std(num_bits: int, theta: float) -> float:
    """海明距离标准差 = sqrt(num_bits × (θ/π) × (1 - θ/π))。"""
    p = theta / math.pi
    return math.sqrt(num_bits * p * (1.0 - p))


def recall_at_radius(num_bits: int, theta: float, r: int) -> float:
    """半径 r 内的召回率。海明距离 ~ Binomial(num_bits, θ/π)，用正态近似积分。"""
    mu = expected_hamming(num_bits, theta)
    sigma = hamming_std(num_bits, theta)
    if sigma < 1e-12:
        return 1.0 if r >= mu else 0.0
    z = (r + 0.5 - mu) / sigma  # 连续性校正
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def compute(spec: SimHashSpec) -> SimHashMetrics:
    """给定参数，递推全部指标。"""
    theta0 = math.acos(max(-1.0, min(1.0, spec.s0)))

    if spec.r == 0:
        r = max(1, round(expected_hamming(spec.num_bits, theta0)))
    else:
        r = spec.r

    sig_bytes = spec.n * ((spec.num_bits + 7) // 8)
    raw_bytes = spec.n * spec.d * 4
    compression = raw_bytes / sig_bytes

    return SimHashMetrics(
        n=spec.n,
        d=spec.d,
        num_bits=spec.num_bits,
        s0=spec.s0,
        theta0=theta0,
        expected_hamming=expected_hamming(spec.num_bits, theta0),
        hamming_std=hamming_std(spec.num_bits, theta0),
        r=r,
        recall_at_r=recall_at_radius(spec.num_bits, theta0, r),
        sig_bytes=sig_bytes,
        raw_bytes=raw_bytes,
        compression_ratio=compression,
    )


def _selftest():
    """公式自洽性检查"""
    spec = SimHashSpec(n=1_000_000, d=128, num_bits=64, s0=0.8)
    m = compute(spec)

    # θ/π 关系：θ=0（完全相同）时海明距离为 0，θ=π/2（正交）时为 num_bits/2
    assert abs(expected_hamming(64, 0.0)) < 1e-12, "θ=0 时期望海明距离应为 0"
    assert abs(expected_hamming(64, math.pi / 2) - 32.0) < 1e-12, "正交时期望海明距离应为 num_bits/2"

    # 相似度越高，期望海明距离越小
    m_low = compute(SimHashSpec(n=1_000_000, d=128, num_bits=64, s0=0.5))
    assert m.expected_hamming < m_low.expected_hamming, "相似度越高海明距离应越小"

    # 位数越多，相对标准差越小（估计越稳）
    m_128 = compute(SimHashSpec(n=1_000_000, d=128, num_bits=128, s0=0.8))
    assert m_128.hamming_std / 128 < m.hamming_std / 64, "位数越多相对标准差应越小"

    # 压缩比 = 32d / num_bits（d=128, bits=64 时为 64x；bits=d 时为 32x）
    assert abs(m.compression_ratio - 32.0 * spec.d / spec.num_bits) < 1e-6, \
        f"压缩比应为 32d/bits，实际 {m.compression_ratio}"
    m_std = compute(SimHashSpec(n=1_000_000, d=128, num_bits=128, s0=0.8))
    assert abs(m_std.compression_ratio - 32.0) < 1e-6, "num_bits=d 时压缩比应为 32x"

    # 自动半径应接近期望海明距离，召回率在合理区间
    assert abs(m.r - round(m.expected_hamming)) <= 1, "自动半径应接近期望海明距离"
    assert 0.4 < m.recall_at_r < 0.8, f"默认半径下召回率应在 0.4-0.8，实际 {m.recall_at_r}"

    print("selftest 全部通过")
    print(f"\nn={m.n:,}  d={m.d}  签名位数={m.num_bits}  阈值 s0={spec.s0}")
    print(f"θ0={math.degrees(m.theta0):.2f}°  期望海明距离={m.expected_hamming:.2f}  σ={m.hamming_std:.2f}")
    print(f"桶半径 r={m.r}  召回率≈{m.recall_at_r:.4f}")
    print(f"签名={m.sig_bytes/1024/1024:.2f} MiB  原始={m.raw_bytes/1024/1024:.2f} MiB  压缩比={m.compression_ratio:.1f}x")


if __name__ == "__main__":
    _selftest()
