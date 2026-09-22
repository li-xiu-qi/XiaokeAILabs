"""
Scaling Laws 模型：模型能力如何随规模外推（Kaplan 2020 / Chinchilla 2022）。

这条线管的是「更大的模型 / 更多数据 → 更强的能力」的幂律关系，用于：
- 判断某个模型规模的预期能力区间（外推，不只看显存和速度）
- 算力最优配比（Chinchilla：每个参数配多少训练 token）

与前面几个模型分工不同：roofline / 显存 / KV 管「跑不跑得动、多快」，
Scaling Laws 管「这个规模的模型大致有多强」。两者结合，才能回答
「这台机器能跑的模型里，哪个最值得跑」。

自测：复现 Chinchilla 最优配比，验证 Kaplan 幂律形式。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


# Kaplan et al. 2020 的幂律形式（论文 Table 3 的实证常数）：
#   L(N, D) = (N_c/N)^αN + (D_c/D)^αD
# 两项分别是「参数不够」和「数据不够」带来的损失，越小越强。
# N_c = 8.8e13 参数，D_c = 5.4e13 token，αN=0.076，αD=0.095。
# 这是 Kaplan 原文的 GPT-3 规模拟合，绝对损失值对现代模型会整体偏移，
# 但「规模越大损失越低、数据/参数要配比」的定性关系成立，用于趋势外推。
N_C = 8.8e13   # 参数尺度（个）
D_C = 5.4e13   # 数据尺度（token）
ALPHA_N = 0.076
ALPHA_D = 0.095


def kaplan_loss(params_b: float, data_t: float) -> float:
    """Kaplan 2020 幂律损失估计。params_b 十亿参数，data_t 万亿 token。"""
    n = params_b * 1e9
    d = data_t * 1e12
    return (N_C / n) ** ALPHA_N + (D_C / d) ** ALPHA_D


def chinchilla_optimal_tokens(params_b: float) -> float:
    """Chinchilla 2022 算力最优：每参数约 20 token。

    返回最优训练 token 数（单位：十亿 token = params_b * 20）。
    """
    return params_b * 20


def chinchilla_optimal_flops(params_b: float) -> float:
    """算力最优对应的总算力（FLOPs）≈ 6 × N × D。

    N 参数量，D token 数，系数 6 来自前向 2ND + 反向 4ND。
    """
    d = chinchilla_optimal_tokens(params_b) * 1e9
    return 6 * params_b * 1e9 * d


def flops_to_params_for_budget(flops: float, tokens_per_param: float = 20):
    """给定总算力预算，算力最优的参数量（十亿参数）。

    C = 6ND，D = tokens_per_param × N  →  C = 6 × tpp × N²
    → N = sqrt(C / (6 × tpp))，除以 1e9 换算成十亿。
    """
    import math
    return math.sqrt(flops / (6 * tokens_per_param)) / 1e9


def self_test():
    print("=" * 64)
    print("Scaling Laws 模型自测：Chinchilla 最优 + Kaplan 幂律")
    print("=" * 64)

    print("\n[1] Chinchilla 算力最优配比（每参数 20 token）")
    print(f"{'模型规模':>10} {'最优token':>12} {'最优算力(FLOPs)':>16}")
    for p in [1.5, 7, 27, 70, 320]:
        d = chinchilla_optimal_tokens(p)
        c = chinchilla_optimal_flops(p)
        print(f"{p:>8.1f}B {d:>10.1f}B {c:>16.2e}")

    print("\n[2] 验证 Chinchilla 论文经典结论：70B 模型最优约 1.4T token")
    d = chinchilla_optimal_tokens(70)
    print(f"  70B → {d:.0f}B = {d/1000:.1f}T token（Chinchilla 论文 1.4T ✓）")

    print("\n[3] Kaplan 幂律损失：同算力下，参数/数据配比如何影响损失")
    print(f"{'参数B':>7} {'数据T':>7} {'token/param':>12} {'预测损失':>9}")
    for p, d in [(1, 0.02), (1, 0.2), (7, 0.14), (7, 1.4), (70, 1.4)]:
        loss = kaplan_loss(p, d)
        tpp = d * 1e12 / (p * 1e9)
        print(f"{p:>7} {d:>7} {tpp:>12.0f} {loss:>9.3f}")

    print("\n[4] 反推：给定算力预算，最优模型规模")
    print(f"{'算力(FLOPs)':>14} {'最优参数':>10} {'最优token':>12}")
    for c in [1e22, 1e23, 1e24, 1e25]:
        p = flops_to_params_for_budget(c)
        d = chinchilla_optimal_tokens(p)
        print(f"{c:>14.0e} {p:>8.1f}B {d:>10.1f}B")

    print("\n[5] 与前面模型的分工说明")
    print("  Scaling Laws 回答「这个规模大致多强」（能力外推）")
    print("  roofline / 显存 / KV 回答「这台机器跑不跑得动、多快」（工程约束）")
    print("  → 两者结合：GB10 能跑的规模里，Qwen3.8-27B 这类是能力/成本甜点")


if __name__ == "__main__":
    self_test()
