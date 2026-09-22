"""
KV Cache 显存模型：上下文长度的内存成本，以及它如何随 batch 缩放。

KV cache = 2 (K和V) × num_layers × num_kv_heads × head_dim × seq_len × bytes × batch

这是长上下文推理的内存瓶颈。GQA/MQA 把 num_kv_heads 压小（Qwen3.8-27B
只有 4 个 KV 头，注意力头 24 个），KV cache 比标准 MHA 小 6 倍。

自测：用真实配置算每 token 字节数，验证长上下文下的显存占用。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_spec import (GB10, Qwen3_8_27B, GLM5_3_Flash, DTYPE_BYTES,
                      kv_bytes_per_token)


def kv_cache_gb(model, ctx_len: int, batch: int = 1, dtype: str = "bf16") -> float:
    """KV cache 显存 (GB)。"""
    return kv_bytes_per_token(model, dtype) * ctx_len * batch / 1e9


def max_ctx_for_memory(model, mem_gb: float, dtype_weights: str = "q4_k_xl",
                       dtype_kv: str = "bf16", batch: int = 1,
                       weight_gb: float = None, overhead_gb: float = 2.0) -> int:
    """在给定显存下，权重 + KV cache 能撑到的最大上下文长度。

    总显存 = 权重 + KV cache + 开销
    → 可用给 KV cache 的 = 显存 - 权重 - 开销
    → max_ctx = 可用 / (每 token KV 字节 × batch)
    """
    if weight_gb is None:
        weight_gb = model.total_params_b * DTYPE_BYTES[dtype_weights]
    avail = mem_gb - weight_gb - overhead_gb
    if avail <= 0:
        return 0
    per_tok = kv_bytes_per_token(model, dtype_kv) * batch
    return int(avail * 1e9 / per_tok)


def total_inference_memory(model, ctx_len: int, batch: int = 1,
                           dtype_weights: str = "bf16", dtype_kv: str = "bf16",
                           include_activation: bool = True) -> float:
    """推理总显存 = 权重 + KV cache + activation + 开销（GB）。

    activation 与 batch、seq_len 近似线性：每层激活约 hidden×seq×batch
    个元素，多层累积。用 hidden_size 粗估，系数取经验值（含残差、norm、
    中间投影的临时张量）。长序列大 batch 时 activation 可达几十 GB。
    """
    w = model.total_params_b * DTYPE_BYTES[dtype_weights]
    kv = kv_cache_gb(model, ctx_len, batch, dtype_kv)
    if include_activation:
        # activation ≈ 层数 × 2(残差+临时) × hidden × seq × batch × 2字节
        # 每层约存 2 份 hidden 维张量（残差流 + 临时），bf16
        act = model.num_layers * 2 * model.hidden_size * ctx_len * batch * 2 / 1e9
    else:
        act = 0.0
    return w + kv + act + 2.0  # 2GB CUDA 上下文/框架开销


def self_test():
    print("=" * 64)
    print("KV Cache 显存模型自测")
    print("=" * 64)

    print("\n[1] 每 token KV cache 字节数（GQA 效应）")
    for m in (Qwen3_8_27B, GLM5_3_Flash):
        kb = kv_bytes_per_token(m) / 1024
        # 对比标准 MHA（KV 头数 = 注意力头数）会占多少
        mha = 2 * m.num_layers * m.num_attention_heads * m.head_dim * 2 / 1024
        print(f"  {m.name}: {kb:.1f} KB/token (GQA {m.num_key_value_heads} KV头) "
              f"vs MHA {mha:.1f} KB/token，省 {mha/kb:.1f}x")

    print("\n[2] Qwen3.8-27B 不同上下文长度下的 KV cache (bf16, batch=1)")
    print(f"{'上下文':>10} {'KV GB':>8} {'权重 GB':>8} {'总计 GB':>8} {'占121GB':>8}")
    for ctx in [1024, 8192, 32768, 131072, 262144]:
        kv = kv_cache_gb(Qwen3_8_27B, ctx)
        w = Qwen3_8_27B.total_params_b * DTYPE_BYTES["bf16"]
        tot = kv + w + 2.0
        print(f"{ctx:>10} {kv:>8.2f} {w:>8.1f} {tot:>8.1f} {tot/GB10.mem_gb*100:>7.0f}%")

    print("\n[3] batch 缩放：8K 上下文下 KV cache 随 batch 线性增长")
    print(f"{'batch':>6} {'KV GB':>8} {'权重 GB':>8} {'总计 GB':>8}")
    for b in [1, 4, 16, 64]:
        kv = kv_cache_gb(Qwen3_8_27B, 8192, batch=b)
        w = Qwen3_8_27B.total_params_b * DTYPE_BYTES["q4_k_xl"]
        print(f"{b:>6} {kv:>8.2f} {w:>8.1f} {kv+w+2.0:>8.1f}")

    print("\n[4] GB10 121GB 下的最大上下文（q4 权重 + bf16 KV）")
    for m in (Qwen3_8_27B, GLM5_3_Flash):
        maxc = max_ctx_for_memory(m, GB10.mem_gb, dtype_weights="q4_k_xl")
        # 超过原生最大上下文的标 *
        native = m.max_position_embeddings
        star = " (超原生,需YaRN外推)" if maxc > native else ""
        print(f"  {m.name}: 最大 ≈ {maxc} tokens{star}")

    print("\n[5] KV cache 量化（fp8 KV）对最大上下文的提升")
    for m in (Qwen3_8_27B,):
        maxc_bf16 = max_ctx_for_memory(m, GB10.mem_gb, dtype_kv="bf16")
        maxc_fp8 = max_ctx_for_memory(m, GB10.mem_gb, dtype_kv="fp8")
        print(f"  {m.name}: bf16 KV → {maxc_bf16} tokens, fp8 KV → {maxc_fp8} tokens "
              f"({maxc_fp8/maxc_bf16:.1f}x)")

    print("\n[6] activation 显存随 batch 和 seq 增长（bf16 权重）")
    print("    activation ≈ 层数×2×hidden×seq×batch×2字节，之前用固定2GB低估了")
    print(f"{'seq':>8} {'batch':>6} {'权重GB':>8} {'KV GB':>8} {'act GB':>8} {'总计GB':>8}")
    for ctx, b in [(1024, 1), (8192, 1), (8192, 8), (32768, 16), (131072, 32)]:
        tot = total_inference_memory(Qwen3_8_27B, ctx, batch=b, dtype_weights="bf16")
        kv = kv_cache_gb(Qwen3_8_27B, ctx, b)
        act = tot - 54.6 - kv - 2.0
        print(f"{ctx:>8} {b:>6} {54.6:>8.1f} {kv:>8.1f} {act:>8.1f} {tot:>8.1f}")
    print("  → 大 batch 长序列时 activation 可达几十 GB，不可忽略")
    print("  → 长上下文服务显存爆的第三个来源（权重、KV、activation）")


if __name__ == "__main__":
    self_test()
