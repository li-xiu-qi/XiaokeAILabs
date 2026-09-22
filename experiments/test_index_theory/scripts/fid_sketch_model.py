# -*- coding: utf-8 -*-
"""
FID-Sketch（细粒度概率计数草图）理论性能模型

FID-Sketch 是 Yang, Zhang, Wang, Shahzad, Liu, Xin & Li (2019) 提出的频次统计结构，
发表在 World Wide Web Journal。它在 Count-Min Sketch 的二维骨架不变的前提下，
把每个计数器从 32 位精确计数换成 4 位概率计数，用 Fine-grained probability
counting（细粒度概率计数，FGC）替代简单加一，空间降到 CMS 的 1/8。

核心机制（Morris 式概率计数）：
- 计数器存 4 位值 v（0-15），不存真实频次，存的是频次的对数量级
- 更新：读到 v 后，以概率 2^(-v) 把 v 加 1；概率随 v 增大而指数衰减
- 等价地，不递增的概率是 p = 1 - 2^(-v)，v 越大越难再涨
- 解码：虚拟计数 ĉ = 2^v - 1，这是 Morris 计数器的标准解码式，满足 E[ĉ] = g
  （g 为该格实际收到的更新数），即无偏
- 查询：对 d 行各解出一个 ĉ_r，取算术平均

关键性质：
1. 空间：d × w × 4 bit，同 (w, d) 下是 32 位 CMS 的 1/8
2. 噪声：单个解码值的相对标准差约 0.70（Morris 计数器的固有乘性噪声），
   d 行平均后降到 0.70/√d
3. 与 CMS 的噪声性质正交：CMS 的碰撞噪声是加性的（εN = eN/w，随 w 增大而减），
   FID 的 Morris 噪声是乘性的（0.70/√d，只随 d 增大而减）
4. 饱和上限：4 位计数器最多表示 2^15 - 1 = 32767 次更新，超过即封顶

误差合成（对频次为 f 的元素，w 行宽、d 行深）：

    期望碰撞次数  μ = N/w
    碰撞项        sqrt(μ)/f          泊松计数的标准差
    Morris 项    (0.70/√d)·(f+μ)/f  乘性噪声作用在格子总更新数上
    rel_std = sqrt( 碰撞项² + Morris 项² )

两项相互独立。碰撞项随 w 增大而减，Morris 项只随 d 增大而减，这决定了两者的
胜负区间：内存紧张时碰撞项主导，FID 用同等内存摆下 8 倍计数器，占优；内存充裕
时 Morris 项主导，CMS 的精确计数器占优。

实测吻合度：频次 50 以上预测与实测相对误差相差在 15% 以内；频次低于 50 时模型
高估（小计数的误差分布右偏，平均绝对值大于标准差），偏差方向已知，不额外修正。

关于估计式的一处必要修正。任务与部分二手资料给出的查询式是
「估计频率 = -m × ln(1 - p̂)」。这个式子在量纲上不成立：右边是常数乘对数，
随频次对数增长，而真实频次线性增长，二者不可能在多个数量级上同时吻合。
以 m=2048、真实频次 100 代入，该式给出约 9400，偏差两个数量级。
保留 p = 1 - 2^(-v) 作为「不递增概率」的映射（它正确地刻画了更新规则），
解码则必须走 ĉ = 2^v - 1 = 1/(1-p) - 1，即概率的倒数减一，而非概率的对数。
验证脚本里同时实测了这两个解码器，偏差数字见 docs/FID-Sketch模型.md。

本模块实现这些公式，参数可配。
"""
from dataclasses import dataclass
import math


# Morris 解码值的相对标准差，实测常数（见 verify_fid_sketch.py 的标定段）
MORRIS_RELATIVE_STD = 0.70

# 每个计数器的比特数（FID-Sketch 用 4 位概率值）
FID_COUNTER_BITS = 4

# 对照组 Count-Min Sketch 的计数器比特数
CMS_COUNTER_BITS = 32


@dataclass
class FIDSketchSpec:
    """FID-Sketch 参数"""
    n: int = 100_000             # 预期总更新次数（流长度 N）
    epsilon: float = 0.01        # 相对误差 ε
    delta: float = 0.001         # 失败概率 δ
    counter_bits: int = FID_COUNTER_BITS   # 每计数器比特数
    width: int = 0               # 宽度 w（0 表示按 ε 自动计算）
    depth: int = 0               # 深度 d（0 表示按 δ 自动计算）


@dataclass
class FIDSketchMetrics:
    n: int                       # 总更新次数 N
    width: int                   # 每行计数器数 w
    depth: int                   # 行数 d
    epsilon: float               # 相对误差
    delta: float                 # 失败概率
    counter_bits: int            # 每计数器比特数
    counter_max: int             # 单个计数器最大值 2^bits - 1
    saturation_limit: int        # 单格可表示的最大更新数 2^(2^bits - 1) - 1
    absolute_error: float        # 碰撞误差界 εN
    num_counters: int            # 计数器总数 d×w
    memory_bits: int             # 总比特数
    memory_bytes: int            # 总字节数
    cms_memory_bytes: int        # 同等 (w,d) 下 32 位 CMS 的内存
    memory_saving_ratio: float   # 相对 32 位 CMS 的空间压缩比
    morris_noise: float          # d 行平均后的 Morris 相对噪声 0.70/√d
    single_counter_noise: float  # 单个解码值的相对噪声


def compute_width(epsilon: float) -> int:
    """宽度。w = ceil(e/ε)，与 Count-Min Sketch 同式。"""
    return int(math.ceil(math.e / epsilon))


def compute_depth(delta: float) -> float:
    """深度。d = ceil(ln(1/δ))，与 Count-Min Sketch 同式。"""
    return int(math.ceil(math.log(1.0 / delta)))


def compute_absolute_error(epsilon: float, n: int) -> float:
    """碰撞误差界。加性噪声上界 εN = eN/w。"""
    return epsilon * n


def compute_num_counters(width: int, depth: int) -> int:
    """计数器总数。d × w。"""
    return width * depth


def compute_memory_bits(width: int, depth: int, counter_bits: int) -> int:
    """总比特数。d × w × counter_bits。"""
    return width * depth * counter_bits


def compute_memory_bytes(width: int, depth: int, counter_bits: int) -> int:
    """总字节数。向上取整到整字节。"""
    return math.ceil(width * depth * counter_bits / 8)


def compute_relative_error_bound(width: int) -> float:
    """给定 w 反推相对误差。ε = e/w。"""
    return math.e / width


def compute_counter_max(counter_bits: int) -> int:
    """计数器可表示的最大值。2^bits - 1。"""
    return (1 << counter_bits) - 1


def compute_saturation_limit(counter_bits: int) -> int:
    """单格可表示的最大更新数。

    Morris 计数器的值 v 对应约 2^v 次更新，4 位计数器 v 上限 15，
    因此单格在收到 2^15 - 1 = 32767 次更新后封顶，估计不再增长。
    这是 4 位计数器的硬上限，选 (w, d) 时必须保证头部元素的频次
    加碰撞噪声不超过它。
    """
    return 2 ** compute_counter_max(counter_bits) - 1


def compute_increment_probability(v: int) -> float:
    """递增概率。2^(-v)。v 越大越难再涨，这是压缩的来源。"""
    return 2.0 ** (-v)


def compute_no_increment_probability(v: int) -> float:
    """不递增概率。p = 1 - 2^(-v)。

    它与递增概率互补，刻画的是「这一格已经攒了多少」的饱和程度。
    注意它只是中间量，估计频率时不能对它取对数（见模块 docstring）。
    """
    return 1.0 - compute_increment_probability(v)


def compute_decode(v: int) -> float:
    """Morris 解码。虚拟计数 ĉ = 2^v - 1，无偏：E[ĉ] = g。

    用概率表示即 ĉ = 1/(1-p) - 1，其中 p = 1 - 2^(-v)。
    """
    return 2.0 ** v - 1.0


def compute_single_counter_noise(morris_std: float = MORRIS_RELATIVE_STD) -> float:
    """单个解码值的相对标准差。实测约 0.70，与频次大小无关。"""
    return morris_std


def compute_morris_noise(depth: int,
                         morris_std: float = MORRIS_RELATIVE_STD) -> float:
    """d 行平均后的 Morris 相对噪声。0.70/√d。

    d 行相互独立，各解出一个 ĉ_r 后取平均，乘性噪声按 1/√d 衰减。
    """
    if depth <= 0:
        return float("inf")
    return morris_std / math.sqrt(depth)


def compute_collision_bound(n: int, width: int) -> float:
    """碰撞误差界（Count-Min Sketch 风格）。eN/w。

    这是 CMS 论文给出的加性噪声上界，配合取 min 聚合使用。
    它不是单格实际收到的碰撞次数的期望，实际期望是 N/w，
    多出的因子 e 是马尔可夫不等式留的安全余量。
    """
    if width <= 0:
        return float("inf")
    return math.e * n / width


def compute_expected_collisions(n: int, width: int) -> float:
    """单格实际收到的期望碰撞次数。μ = N/w。

    w 个格子均匀分摊 N 次更新，每格期望 N/w 次。
    """
    if width <= 0:
        return float("inf")
    return n / width


def compute_collision_relative_error(n: int, width: int,
                                    frequency: float) -> float:
    """碰撞项的相对标准差。sqrt(N/w)/f。

    碰撞次数近似服从泊松分布，均值 μ = N/w、标准差 sqrt(μ)。
    """
    if frequency <= 0 or width <= 0:
        return float("inf")
    return math.sqrt(n / width) / frequency


def compute_expected_relative_error(n: int, width: int, depth: int,
                                    frequency: float,
                                    morris_std: float = MORRIS_RELATIVE_STD
                                    ) -> float:
    """合成相对标准差。

    两项相互独立，按平方和开方合成。
    碰撞项与 f 成反比（加性），Morris 项随 f 增大趋近 0.70/√d 的上界
    （乘性，作用在格子总更新数 f + N/w 上）。
    """
    if frequency <= 0 or depth <= 0:
        return float("inf")
    mu = compute_expected_collisions(n, width)
    collision = math.sqrt(mu) / frequency
    morris = compute_morris_noise(depth, morris_std) * (frequency + mu) / frequency
    return math.sqrt(collision * collision + morris * morris)


def compute_equal_memory_width(cms_width: int, cms_bits: int = CMS_COUNTER_BITS,
                               fid_bits: int = FID_COUNTER_BITS) -> int:
    """同等内存下 FID 可摆下的宽度。

    计数器比特数从 32 降到 4，同等 (w, d) 省 8 倍；若保持 d 不变，
    宽度可以放大 8 倍，碰撞项随之降到 1/8。
    """
    return cms_width * cms_bits // fid_bits


def compute(spec: FIDSketchSpec) -> FIDSketchMetrics:
    """给定参数，递推全部指标。"""
    w = compute_width(spec.epsilon) if spec.width == 0 else spec.width
    d = compute_depth(spec.delta) if spec.depth == 0 else spec.depth
    epsilon = spec.epsilon if spec.width == 0 else compute_relative_error_bound(w)

    mem_bits = compute_memory_bits(w, d, spec.counter_bits)
    cms_bytes = compute_memory_bytes(w, d, CMS_COUNTER_BITS)
    fid_bytes = compute_memory_bytes(w, d, spec.counter_bits)

    return FIDSketchMetrics(
        n=spec.n,
        width=w,
        depth=d,
        epsilon=epsilon,
        delta=spec.delta,
        counter_bits=spec.counter_bits,
        counter_max=compute_counter_max(spec.counter_bits),
        saturation_limit=compute_saturation_limit(spec.counter_bits),
        absolute_error=compute_absolute_error(epsilon, spec.n),
        num_counters=compute_num_counters(w, d),
        memory_bits=mem_bits,
        memory_bytes=fid_bytes,
        cms_memory_bytes=cms_bytes,
        memory_saving_ratio=cms_bytes / fid_bytes,
        morris_noise=compute_morris_noise(d),
        single_counter_noise=compute_single_counter_noise(),
    )


def _selftest():
    """公式自洽性检查"""
    spec = FIDSketchSpec(n=100_000, epsilon=0.01, delta=0.001)
    m = compute(spec)

    # w 与 d 应与 CMS 同式
    assert m.width == 272, f"w 应为 272，实际 {m.width}"
    assert m.depth == 7, f"d 应为 7，实际 {m.depth}"

    # 4 位计数器：最大值 15，单格饱和上限 32767
    assert m.counter_max == 15, "4 位计数器最大值应为 15"
    assert m.saturation_limit == 2 ** 15 - 1, "饱和上限应为 2^15-1"

    # 同等 (w, d) 下内存是 32 位 CMS 的 1/8
    assert m.cms_memory_bytes == 8 * m.memory_bytes, \
        "4 位计数器内存应为 32 位的 1/8"
    assert m.memory_saving_ratio == 8.0, "空间压缩比应为 8 倍"

    # 递增概率随 v 指数衰减
    p0 = compute_increment_probability(0)
    p1 = compute_increment_probability(1)
    p4 = compute_increment_probability(4)
    assert p0 == 1.0 and p1 == 0.5 and p4 == 0.0625, "递增概率应为 2^(-v)"
    assert p0 > p1 > p4, "递增概率应随 v 增大而减小"

    # p = 1 - 2^(-v) 与递增概率互补，且落在 [0, 1)
    for v in range(16):
        p = compute_no_increment_probability(v)
        assert 0.0 <= p < 1.0, "p 应落在 [0, 1)"
        assert abs(p + compute_increment_probability(v) - 1.0) < 1e-12, \
            "p 应与递增概率互补"

    # Morris 解码：ĉ = 2^v - 1
    assert compute_decode(0) == 0.0, "v=0 应解码为 0"
    assert compute_decode(1) == 1.0, "v=1 应解码为 1"
    assert compute_decode(4) == 15.0, "v=4 应解码为 15"

    # 无偏性：E[2^v - 1] = g（模拟验证，固定种子保证可复现）
    import random
    for g in [1, 10, 100, 1000]:
        rng = random.Random(20260907)
        total = 0.0
        trials = 4000
        for _ in range(trials):
            v = 0
            for _ in range(g):
                if rng.random() < 2.0 ** (-v) and v < 15:
                    v += 1
            total += compute_decode(v)
        mean = total / trials
        assert abs(mean - g) / g < 0.05, \
            f"g={g} 时解码均值应为 {g}，实际 {mean:.1f}"

    # Morris 噪声随 d 按 1/√d 衰减
    n1 = compute_morris_noise(1)
    n4 = compute_morris_noise(4)
    assert abs(n4 - n1 / 2.0) < 1e-12, "d 变 4 倍，噪声应减半"
    assert 0.6 < n1 < 0.8, "单计数器相对噪声应在 0.6-0.8"

    # 合成误差单调性：w 越大碰撞项越小，d 越大 Morris 项越小
    e_wide = compute_expected_relative_error(100_000, 4096, 5, 100.0)
    e_narrow = compute_expected_relative_error(100_000, 256, 5, 100.0)
    assert e_wide < e_narrow, "w 越大，合成误差应越小"
    e_deep = compute_expected_relative_error(100_000, 2048, 20, 100.0)
    e_shallow = compute_expected_relative_error(100_000, 2048, 5, 100.0)
    assert e_deep < e_shallow, "d 越大，合成误差应越小"

    # 期望碰撞次数为 N/w，CMS 误差界为 eN/w
    assert abs(compute_expected_collisions(100_000, 2048) - 48.83) < 0.01, \
        "期望碰撞次数应为 N/w"
    assert abs(compute_collision_bound(100_000, 2048) - 132.73) < 0.01, \
        "碰撞误差界应为 eN/w"

    # 低频项上碰撞项主导，高频项上 Morris 项主导
    low = compute_expected_relative_error(100_000, 2048, 5, 1.0)
    high = compute_expected_relative_error(100_000, 2048, 5, 10_000.0)
    assert low > high, "低频项的合成误差应更大"

    # 高频项的合成误差应收敛到 Morris 项
    assert high < compute_morris_noise(5) * 1.01, \
        "高频项的合成误差应收敛到 Morris 项"

    # 同等内存下 FID 宽度是 CMS 的 8 倍
    assert compute_equal_memory_width(256) == 2048, \
        "同等内存下 FID 宽度应为 CMS 的 8 倍"

    print("selftest 全部通过")
    print(f"\nN={m.n:,}  ε={m.epsilon}  δ={m.delta}  计数器={m.counter_bits} 位")
    print(f"w={m.width}  d={m.depth}  计数器总数={m.num_counters:,}  "
          f"碰撞误差界={m.absolute_error:,.0f}")
    print(f"内存={m.memory_bits:,} bits = {m.memory_bytes:,} B"
          f"（32 位 CMS 同构 {m.cms_memory_bytes:,} B，"
          f"压缩 {m.memory_saving_ratio:.0f}x）")
    print(f"单计数器 Morris 噪声={m.single_counter_noise:.3f}，"
          f"d={m.depth} 行平均后={m.morris_noise:.3f}")
    print(f"单格饱和上限={m.saturation_limit:,} 次更新")


if __name__ == "__main__":
    _selftest()
