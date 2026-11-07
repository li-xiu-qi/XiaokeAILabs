"""Simplified dense attention cost, not a latency or peak-memory predictor.

Counts QK^T and AV multiply-adds for full, non-causal matrices only.
GQA still has one score matrix per query head. Hybrid linear-attention layers,
projections, masks and kernel-specific intermediates are not included.
"""
from types import SimpleNamespace


def attention_flops(model, ctx_len: int, batch: int = 1) -> float:
    """Dense non-causal prefill FLOPs, full-attention layers only."""
    if ctx_len <= 0 or batch <= 0:
        raise ValueError("ctx_len and batch must be positive")
    return 4 * batch * ctx_len**2 * model.head_dim * model.num_attention_heads * model.kv_layers


def attn_score_matrix_gb(model, ctx_len: int, batch: int = 1, bytes_per_element: int = 2) -> float:
    """Materialized score matrix for ONE layer, in decimal GB, not peak memory."""
    if min(ctx_len, batch, bytes_per_element) <= 0:
        raise ValueError("dimensions and element size must be positive")
    return batch * model.num_attention_heads * ctx_len**2 * bytes_per_element / 1e9


def self_test():
    # Synthetic dimensions, not a claim about a particular published model.
    m = SimpleNamespace(head_dim=256, num_attention_heads=24, kv_layers=16)
    print("Synthetic example: 16 full-attention layers, 24 query heads, head dim 256")
    print("Dense non-causal attention estimate; excludes other layer types")
    for s in (1024, 8192, 32768):
        print(f"S={s:6d}: attention={attention_flops(m, s):.3e} FLOPs; "
              f"one-layer BF16 score matrix={attn_score_matrix_gb(m, s):.3f} GB")
    print("FlashAttention avoids materializing the full score matrix; actual peak memory depends on implementation.")


if __name__ == "__main__":
    self_test()
