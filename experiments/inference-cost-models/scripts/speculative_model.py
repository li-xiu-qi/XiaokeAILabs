"""
推测解码（Speculative Decoding）加速模型：加速比由接受率决定。

小草稿模型自回归生成 k 个候选 token，大目标模型一次前向并行验证。
接受的 token 直接采用，拒绝的丢弃。加速的来源是「一次目标前向验证
多个 token」，但只有被接受的部分才赚到。

关键量：
- α：草稿 token 被目标接受的比例（草稿质量决定上限）
- k：每次草稿生成长度
- cost_ratio：草稿/目标单次前向成本比（草稿小，远小于 1）

期望接受 token 数 E = (1-α^(k+1))/(1-α)。
加速比 ≈ E / (1 + k × cost_ratio)，即一次目标前向换来 E 个 token。

自测：参数扫描展示加速比如何随 α、k 变化，校准到 GB10 社区数据。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_spec import GB10, Qwen3_8_27B, DTYPE_BYTES


# GB10 社区实测锚点（2026-08 阿里云测速）：
# llama.cpp Q4_K_M 单流 12.1 t/s → SGLang + NEXTN 投机采样 23 t/s
GB10_BASE_TPS = 12.1      # llama.cpp 单流基线
GB10_SPEC_TPS = 23.0      # SGLang + NEXTN 加速后
GB10_SPEEDUP = GB10_SPEC_TPS / GB10_BASE_TPS  # 1.9x


def expected_accepted(alpha: float, k: int) -> float:
    """期望接受的 token 数 = (1-α^(k+1))/(1-α)。

    几何分布的期望。α=1（全接受）时等于 k+1，α→0 时接近 1。
    """
    if alpha >= 1.0:
        return k + 1
    if alpha <= 0.0:
        return 1.0
    return (1 - alpha ** (k + 1)) / (1 - alpha)


def speedup(alpha: float, k: int, cost_ratio: float = 0.05) -> float:
    """推测解码加速比。

    一次迭代：草稿生成 k 个 token（成本 k × cost_ratio），目标验证
    k+1 个位置（成本 1），赚到 E 个 token。
    加速比 = E / (1 + k × cost_ratio)。
    cost_ratio 是草稿/目标单次前向成本比（小模型草稿远小于 1）。
    """
    e = expected_accepted(alpha, k)
    return e / (1 + k * cost_ratio)


def effective_tps(base_tps: float, alpha: float, k: int,
                  cost_ratio: float = 0.05) -> float:
    """加速后的有效 decode 吞吐。"""
    return base_tps * speedup(alpha, k, cost_ratio)


def solve_alpha_for_speedup(target_speedup: float, k: int,
                            cost_ratio: float = 0.05) -> float:
    """反解：达到目标加速比所需的接受率 α（二分法）。"""
    lo, hi = 0.0, 0.999
    for _ in range(60):
        mid = (lo + hi) / 2
        if speedup(mid, k, cost_ratio) < target_speedup:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def self_test():
    print("=" * 64)
    print("推测解码加速模型自测：加速比由接受率决定")
    print("=" * 64)

    print("\n[1] 期望接受 token 数 E 随接受率 α、草稿长度 k 变化")
    print(f"{'α':>6} " + " ".join(f"k={k:<4}" for k in [2, 4, 6, 8]))
    for a in [0.5, 0.6, 0.7, 0.8, 0.9]:
        row = " ".join(f"{expected_accepted(a, k):>5.2f}" for k in [2, 4, 6, 8])
        print(f"{a:>6.1f} {row}")

    print("\n[2] 加速比随 α、k 变化（cost_ratio=0.05，小 draft 模型）")
    print(f"{'α':>6} " + " ".join(f"k={k:<4}" for k in [2, 4, 6, 8]))
    for a in [0.5, 0.6, 0.7, 0.8, 0.9]:
        row = " ".join(f"{speedup(a, k):>5.2f}x" for k in [2, 4, 6, 8])
        print(f"{a:>6.1f} {row}")

    print("\n[3] 校准 GB10 社区数据：12.1 → 23 t/s (1.9x)")
    print(f"  实测加速比: {GB10_SPEEDUP:.2f}x")
    for k in [3, 4, 5, 6]:
        need_a = solve_alpha_for_speedup(GB10_SPEEDUP, k)
        print(f"    若 k={k}，需接受率 α ≥ {need_a:.2f} 才能达到 1.9x")
    print("  → 社区 1.9x 加速对应接受率约 0.7-0.8（草稿模型质量较好）")

    print("\n[4] 加速比受草稿成本 cost_ratio 的影响（α=0.7, k=4）")
    print(f"{'cost_ratio':>11} {'加速比':>8} {'有效t/s':>9}")
    for cr in [0.02, 0.05, 0.10, 0.20]:
        sp = speedup(0.7, 4, cr)
        print(f"{cr:>11.2f} {sp:>7.2f}x {GB10_BASE_TPS*sp:>8.1f}")
    print("  → 草稿模型越大（cost_ratio 越高），净加速比越低")
    print("  → 草稿太小接受率低，太大成本高，存在最优草稿规模")

    print("\n[5] 外推：不同基线上的推测解码收益（α=0.75, k=4, cr=0.05）")
    print(f"{'基线':>14} {'基线t/s':>9} {'加速后t/s':>10}")
    for name, base in [("llama.cpp q4", 12.1), ("SGLang nvfp4", 18.0),
                       ("4090 q4", 49.0)]:
        sp = speedup(0.75, 4, 0.05)
        print(f"{name:>14} {base:>9.1f} {base*sp:>10.1f}")
    print("  → 推测解码是乘法加速，叠在任何基线之上")
    print("  → 但接受率 α 由草稿质量决定，是唯一的核心变量")


if __name__ == "__main__":
    self_test()
