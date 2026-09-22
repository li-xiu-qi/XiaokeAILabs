"""
注意力复杂度模型：上下文长度对算力和显存的二次压力，以及解法。

标准注意力的计算量和显存都随序列长度 S 平方增长：
- 注意力分数矩阵 QK^T/√d：S×S，算力 2×S²×d，显存 S²
- FlashAttention：算力仍是 O(S²)，但显存降到 O(S)（分块、不物化分数矩阵）
- 线性注意力 / SSM / Mamba：算力和显存都降到 O(S)

这个模型回答：上下文拉长时，哪部分是瓶颈，FlashAttention/线性注意力各解决什么。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_spec import Qwen3_8_27B, GLM5_3_Flash


def attention_flops(model, ctx_len: int) -> float:
    """一次前向里注意力部分的 FLOPs。

    QK^T：S×S×d 乘加 = 2×S²×d
    AV：S×S×d 乘加 = 2×S²×d
    合计每层 4×S²×d，乘层数。返回总 FLOPs。
    """
    per_layer = 4 * ctx_len**2 * model.head_dim
    return per_layer * model.num_layers


def projection_flops(model, ctx_len: int) -> float:
    """投影 + MLP 的 FLOPs（随 S 线性，用于对比注意力何时反超）。

    每 token 约 2×P FLOPs，乘 S。
    """
    return 2 * model.total_params_b * 1e9 * ctx_len


def attn_score_matrix_gb(model, ctx_len: int, batch: int = 1) -> float:
    """标准注意力分数矩阵显存 (GB)：每层 S×S，乘层数、头数、batch。

    这是标准注意力的显存瓶颈。FlashAttention 通过不物化它消除。
    """
    # 每层每头一个 S×S 矩阵，fp32/bf16
    per_layer = model.num_attention_heads * ctx_len * ctx_len * 2  # bf16
    return per_layer * model.num_layers * batch / 1e9


def attn_fraction(model, ctx_len: int) -> float:
    """注意力 FLOPs 占全部 FLOPs 的比例。

    短上下文：注意力占比小，投影/MLP 主导。
    长上下文：注意力 S² 反超线性部分，成为算力瓶颈。
    """
    a = attention_flops(model, ctx_len)
    p = projection_flops(model, ctx_len)
    return a / (a + p)


def crossover_ctx(model) -> int:
    """注意力算力反超投影/MLP 算力的上下文拐点（解 a = p）。

    4×L×S²×d = 2×P×S  →  S = P/(2×L×d)
    """
    return int(model.total_params_b * 1e9 / (2 * model.num_layers * model.head_dim))


def self_test():
    print("=" * 64)
    print("注意力复杂度模型自测：上下文拐点与显存压力")
    print("=" * 64)

    print("\n[1] 注意力算力 vs 投影/MLP 算力，找反超拐点")
    for m in (Qwen3_8_27B, GLM5_3_Flash):
        x = crossover_ctx(m)
        print(f"  {m.name}: 拐点 S ≈ {x} tokens（超过它注意力成为算力瓶颈）")

    print("\n[2] Qwen3.8-27B 各上下文下注意力占比与分数矩阵显存")
    print(f"{'上下文':>8} {'注意力占比':>10} {'分数矩阵 GB':>12} {'(batch=1)':>10}")
    for ctx in [1024, 8192, 32768, 131072, 262144]:
        frac = attn_fraction(Qwen3_8_27B, ctx)
        mat = attn_score_matrix_gb(Qwen3_8_27B, ctx)
        print(f"{ctx:>8} {frac*100:>9.1f}% {mat:>12.2f} {'':>10}")

    print("\n[3] 解法对比：标准注意力 vs FlashAttention vs 线性注意力")
    print("  上下文长度    标准算力/显存   Flash算力/显存   线性算力/显存")
    for ctx in [4096, 32768, 131072, 1048576]:
        flops_std = attention_flops(Qwen3_8_27B, ctx)
        mat_std = attn_score_matrix_gb(Qwen3_8_27B, ctx)
        # FlashAttention: 算力同标准（O(S²)），显存 O(S)
        mat_fa = (Qwen3_8_27B.num_layers * Qwen3_8_27B.num_attention_heads
                  * ctx * 2) / 1e9  # O(S) 行块
        # 线性注意力: 算力 O(S)，显存 O(S)
        flops_lin = projection_flops(Qwen3_8_27B, ctx)  # 量级 ~O(S)
        print(f"  {ctx:>8}  O(S²)={flops_std:.1e} / {mat_std:.1f}GB   "
              f"O(S²)={flops_std:.1e} / {mat_fa:.2f}GB   O(S) / O(S)")

    print("\n[4] 结论：1M 上下文时的显存墙")
    ctx = 1048576
    mat = attn_score_matrix_gb(Qwen3_8_27B, ctx)
    print(f"  标准注意力 1M 上下文分数矩阵: {mat:.0f} GB（远超 GB10 {121}GB）")
    print(f"  → 必须 FlashAttention（显存 O(S)）才能跑 1M 上下文")
    print(f"  → 社区 1M 上下文用 NVFP4 + KV 量化 + FlashAttention 才放得下")


if __name__ == "__main__":
    self_test()
