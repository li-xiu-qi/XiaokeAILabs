#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""02_model.py — 模型定义自检。

llm_common.CausalLM 的形状、QK norm、RoPE 相对位置性、KV cache 与 naive 前向一致性、
softcap 截断、relu^2/SwiGLU 双路径、GQA、untied/tied 权重等自检。
直接运行：python scripts/02_model.py
"""

from __future__ import annotations

import torch

import llm_common as lc


def cfg(**kw) -> lc.ModelConfig:
    base = dict(
        vocab_size=128, d_model=64, n_layers=2, n_heads=4, n_kv_heads=4, d_ff=256, max_len=32
    )
    base.update(kw)
    return lc.ModelConfig(**base)


def check_shapes() -> None:
    model = lc.build_model(cfg(), seed=0)
    idx = torch.randint(0, 128, (2, 16))
    logits, loss = model(idx, targets=idx)
    assert logits.shape == (2, 16, 128)
    assert loss.dim() == 0 and loss.item() > 0
    print(f"[02] 形状 OK: logits={tuple(logits.shape)}, loss={loss.item():.4f}")


def check_qk_norm() -> None:
    """QK norm：把 q/k 投影权重整体放大，注意力输出应几乎不变（RoPE 后归一化）。"""
    torch.manual_seed(0)
    model = lc.build_model(cfg(qk_norm=True), seed=0)
    idx = torch.randint(0, 128, (1, 12))
    with torch.no_grad():
        out1, _ = model(idx)
        blk = model.blocks[0].attn
        blk.q_proj.weight.mul_(8.0)  # 放大 query 投影
        blk.k_proj.weight.mul_(8.0)  # 放大 key 投影
        out2, _ = model(idx)
    diff = (out1 - out2).abs().max().item()
    scale = out1.abs().max().item()
    assert diff < 1e-3 * max(scale, 1.0), f"QK norm 未生效，diff={diff}"
    print(f"[02] QK norm OK: q/k 缩放 8x 后 logits 最大变化 {diff:.2e}（阈值内）")


def check_rope_relative() -> None:
    """RoPE：位置 i 与 j 的 q·k 点积只依赖相对距离。"""
    head_dim = 16
    rope = lc.RotaryEmbedding(head_dim, max_len=64, base=10000.0)
    # 同一个 q/k 向量铺满所有位置：点积应只依赖两位置的相对距离
    qv = torch.randn(1, 1, 1, head_dim).repeat(1, 1, 16, 1)
    kv = torch.randn(1, 1, 1, head_dim).repeat(1, 1, 16, 1)
    cos, sin = rope.get_cos_sin(16)
    qr = lc.apply_rotary_emb(qv, cos, sin)
    kr = lc.apply_rotary_emb(kv, cos, sin)
    dot = torch.einsum("bhtd,bhsd->bhts", qr, kr).squeeze()
    # 相对位置性：dot[t, s] 只依赖 t-s，即沿对角线平移不变
    max_dev = (dot[1:, 1:] - dot[:-1, :-1]).abs().max().item()
    assert max_dev < 1e-3, f"RoPE 相对位置性被破坏，max_dev={max_dev}"
    print(f"[02] RoPE 相对位置性 OK: 偏差 {max_dev:.2e}")


def check_kv_cache_consistency() -> None:
    """KV cache 逐 token 生成必须与整段前向结果一致。"""
    c = cfg(n_layers=3, n_kv_heads=2, max_len=48, dropout=0.0)
    model = lc.build_model(c, seed=0)
    idx = torch.randint(0, 128, (1, 10))
    with torch.no_grad():
        out_cached, _ = lc.generate(
            model, idx.clone(), max_new=5, eos=0, temperature=0.0, use_cache=True
        )
        out_naive, _ = lc.generate(
            model, idx.clone(), max_new=5, eos=0, temperature=0.0, use_cache=False
        )
        full_logits, _ = model(idx)
        cache = lc.KVCache(c.n_layers, c.n_kv_heads, c.d_model // c.n_heads, c.max_len)
        prefill_logits, _ = model(idx[:, :-1], kv_cache=cache)
    assert torch.equal(out_cached, out_naive), "KV cache 与 naive 生成结果不一致"
    diff = (prefill_logits - full_logits[:, :-1]).abs().max().item()
    assert diff < 1e-4, f"KV cache prefill 与整段前向不一致，diff={diff}"
    # batch>1：同一 prompt 复制成 batch，cache 与 naive 仍须逐 token 一致
    idx_b = idx.repeat(3, 1)
    with torch.no_grad():
        oc, _ = lc.generate(model, idx_b.clone(), max_new=5, eos=0, temperature=0.0, use_cache=True)
        on, _ = lc.generate(
            model, idx_b.clone(), max_new=5, eos=0, temperature=0.0, use_cache=False
        )
    assert torch.equal(oc, on), "batch>1 时 KV cache 与 naive 不一致"
    print(
        f"[02] KV cache 一致性 OK: 生成逐 token 一致, prefill logits 偏差 {diff:.2e}, "
        f"batch=3 亦一致"
    )


def check_softcap() -> None:
    model = lc.build_model(cfg(softcap=15.0), seed=0)
    idx = torch.randint(0, 128, (1, 16))
    with torch.no_grad():
        logits, _ = model(idx)
    assert logits.abs().max().item() <= 15.0 + 1e-4, "softcap 未生效"
    print(f"[02] softcap OK: logits 范围 [{logits.min():.3f}, {logits.max():.3f}] ⊆ [-15, 15]")


def check_mlp_variants() -> None:
    for mlp in ("relu2", "swiglu"):
        model = lc.build_model(cfg(mlp_type=mlp), seed=0)
        idx = torch.randint(0, 128, (1, 12))
        _, loss = model(idx, targets=idx)
        loss.backward()
        print(f"[02] MLP={mlp} OK: loss={loss.item():.4f}, 反向传播正常")


def check_tie_weights() -> None:
    model = lc.build_model(cfg(n_layers=1, tie_weights=True), seed=0)
    assert model.lm_head.weight.data_ptr() == model.tok_emb.weight.data_ptr()
    print("[02] tie_weights OK: lm_head 与 tok_emb 共享存储")


def main() -> int:
    check_shapes()
    check_qk_norm()
    check_rope_relative()
    check_kv_cache_consistency()
    check_softcap()
    check_mlp_variants()
    check_tie_weights()

    cfg_default = lc.ModelConfig()
    model = lc.build_model(cfg_default, seed=0)
    groups = model.num_param_groups()
    n = model.num_params()
    print(f"[02] 默认配置参数量: {n:,} ({n / 1e6:.2f}M)")
    print(f"[02] 参数分组: {groups}")
    print(f"[02] config: {cfg_default.to_dict()}")
    print("[02] 全部自检通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
