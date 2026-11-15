#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""02_model.py — 模型定义自检（CPU / CUDA / DDP 三态）。

llm_common.CausalLM 的形状、QK norm、RoPE 相对位置性、KV cache 与 naive 前向一致性、
softcap 截断、relu^2/SwiGLU 双路径、GQA、untied/tied 权重等自检。
被 torchrun 拉起时（WORLD_SIZE>1）额外执行 DDP 自检：权重广播一致、同 batch 梯度一致、
checkpoint 无 module. 前缀且可跨设备加载。

用法：
    python scripts/02_model.py                    # auto 设备（有卡用卡）
    python scripts/02_model.py --device cpu       # 强制 CPU
    python scripts/02_model.py --device cuda      # 单卡自检
    torchrun --nproc_per_node=2 scripts/02_model.py --device cpu    # 2-rank gloo
    torchrun --nproc_per_node=2 scripts/02_model.py --device cuda   # 2-rank（单机双卡或共用一卡）

输出目录取 XL_OUT_DIR（未设置时默认写入 .work/out/<run_id>）。
"""

from __future__ import annotations

import argparse

import torch

import llm_common as lc


def cfg(**kw) -> lc.ModelConfig:
    base = dict(
        vocab_size=128, d_model=64, n_layers=2, n_heads=4, n_kv_heads=4, d_ff=256, max_len=32
    )
    base.update(kw)
    return lc.ModelConfig(**base)


def _p(*a) -> None:
    print(*a)


def check_shapes(device) -> None:
    model = lc.build_model(cfg(), seed=0).to(device)
    idx = torch.randint(0, 128, (2, 16), device=device)
    logits, loss = model(idx, targets=idx)
    assert logits.shape == (2, 16, 128)
    assert loss.dim() == 0 and loss.item() > 0
    _p(f"[02] 形状 OK: logits={tuple(logits.shape)}, loss={loss.item():.4f} (device={device})")


def check_qk_norm(device) -> None:
    """QK norm：把 q/k 投影权重整体放大，注意力输出应几乎不变（RoPE 后归一化）。"""
    torch.manual_seed(0)
    model = lc.build_model(cfg(qk_norm=True), seed=0).to(device)
    idx = torch.randint(0, 128, (1, 12), device=device)
    with torch.no_grad():
        out1, _ = model(idx)
        blk = model.blocks[0].attn
        blk.q_proj.weight.mul_(8.0)  # 放大 query 投影
        blk.k_proj.weight.mul_(8.0)  # 放大 key 投影
        out2, _ = model(idx)
    diff = (out1 - out2).abs().max().item()
    scale = out1.abs().max().item()
    assert diff < 1e-3 * max(scale, 1.0), f"QK norm 未生效，diff={diff}"
    _p(f"[02] QK norm OK: q/k 缩放 8x 后 logits 最大变化 {diff:.2e}（阈值内）")


def check_rope_relative(device) -> None:
    """RoPE：位置 i 与 j 的 q·k 点积只依赖相对距离。"""
    head_dim = 16
    rope = lc.RotaryEmbedding(head_dim, max_len=64, base=10000.0).to(device)
    # 同一个 q/k 向量铺满所有位置：点积应只依赖两位置的相对距离
    qv = torch.randn(1, 1, 1, head_dim, device=device).repeat(1, 1, 16, 1)
    kv = torch.randn(1, 1, 1, head_dim, device=device).repeat(1, 1, 16, 1)
    cos, sin = rope.get_cos_sin(16)
    qr = lc.apply_rotary_emb(qv, cos, sin)
    kr = lc.apply_rotary_emb(kv, cos, sin)
    dot = torch.einsum("bhtd,bhsd->bhts", qr, kr).squeeze()
    # 相对位置性：dot[t, s] 只依赖 t-s，即沿对角线平移不变
    max_dev = (dot[1:, 1:] - dot[:-1, :-1]).abs().max().item()
    assert max_dev < 1e-3, f"RoPE 相对位置性被破坏，max_dev={max_dev}"
    # 缓存扩容路径也要保持设备对齐（模型 .to(device) 后再建 cache 不落 CPU）
    cos2, sin2 = rope.get_cos_sin(16, offset=60)
    assert cos2.device.type == device.type and sin2.device.type == device.type, "RoPE 缓存扩容后设备漂移"
    _p(f"[02] RoPE 相对位置性 OK: 偏差 {max_dev:.2e}, 缓存扩容设备对齐 {device}")


def check_kv_cache_consistency(device) -> None:
    """KV cache 逐 token 生成必须与整段前向结果一致（当前设备上）。"""
    c = cfg(n_layers=3, n_kv_heads=2, max_len=48, dropout=0.0)
    model = lc.build_model(c, seed=0).to(device)
    idx = torch.randint(0, 128, (1, 10), device=device)
    with torch.no_grad():
        out_cached, _ = lc.generate(
            model, idx.clone(), max_new=5, eos=0, temperature=0.0, use_cache=True
        )
        out_naive, _ = lc.generate(
            model, idx.clone(), max_new=5, eos=0, temperature=0.0, use_cache=False
        )
        full_logits, _ = model(idx)
        cache = lc.KVCache(
            c.n_layers, c.n_kv_heads, c.d_model // c.n_heads, c.max_len, device=device
        )
        prefill_logits, _ = model(idx[:, :-1], kv_cache=cache)
    # CUDA 上 SDPA 对不同形状可能选不同 kernel，贪心 argmax 仍应一致；logits 允许小误差
    ids_equal = torch.equal(out_cached, out_naive)
    diff = (prefill_logits - full_logits[:, :-1]).abs().max().item()
    assert ids_equal or (out_cached == out_naive).float().mean().item() > 0.99, (
        "KV cache 与 naive 生成结果不一致"
    )
    assert diff < 1e-3, f"KV cache prefill 与整段前向不一致，diff={diff}"
    # batch>1：同一 prompt 复制成 batch，cache 与 naive 仍须逐 token 一致
    idx_b = idx.repeat(3, 1)
    with torch.no_grad():
        oc, _ = lc.generate(model, idx_b.clone(), max_new=5, eos=0, temperature=0.0, use_cache=True)
        on, _ = lc.generate(model, idx_b.clone(), max_new=5, eos=0, temperature=0.0, use_cache=False)
    assert torch.equal(oc, on), "batch>1 时 KV cache 与 naive 不一致"
    # 温度采样：Generator 与输入同设备，CUDA 上必须不报错且可复现
    s1, _ = lc.generate(model, idx_b.clone(), 4, 0, temperature=0.9, seed=7)
    s2, _ = lc.generate(model, idx_b.clone(), 4, 0, temperature=0.9, seed=7)
    assert torch.equal(s1, s2), "同种子温度采样不可复现"
    _p(
        f"[02] KV cache 一致性 OK: 生成逐 token 一致, prefill logits 偏差 {diff:.2e}, "
        f"batch=3 亦一致, 温度采样可复现 (device={device})"
    )


def check_softcap(device) -> None:
    model = lc.build_model(cfg(softcap=15.0), seed=0).to(device)
    idx = torch.randint(0, 128, (1, 16), device=device)
    with torch.no_grad():
        logits, _ = model(idx)
    assert logits.abs().max().item() <= 15.0 + 1e-4, "softcap 未生效"
    _p(f"[02] softcap OK: logits 范围 [{logits.min():.3f}, {logits.max():.3f}] ⊆ [-15, 15]")


def check_mlp_variants(device) -> None:
    for mlp in ("relu2", "swiglu"):
        model = lc.build_model(cfg(mlp_type=mlp), seed=0).to(device)
        idx = torch.randint(0, 128, (1, 12), device=device)
        _, loss = model(idx, targets=idx)
        loss.backward()
        _p(f"[02] MLP={mlp} OK: loss={loss.item():.4f}, 反向传播正常 (device={device})")


def check_tie_weights() -> None:
    model = lc.build_model(cfg(n_layers=1, tie_weights=True), seed=0)
    assert model.lm_head.weight.data_ptr() == model.tok_emb.weight.data_ptr()
    _p("[02] tie_weights OK: lm_head 与 tok_emb 共享存储")


def check_amp(device) -> None:
    """AMP：bf16（以及 CUDA 上的 fp16）autocast 前向，loss 有限且与 fp32 接近。"""
    torch.manual_seed(0)
    model = lc.build_model(cfg(), seed=0).to(device)
    idx = torch.randint(0, 128, (2, 16), device=device)
    with torch.no_grad():
        logits_fp32, loss_fp32 = model(idx, targets=idx)
    modes = ["bf16"] + (["fp16"] if device.type == "cuda" else [])
    for amp in modes:
        amp_ctx = lc.AmpCtx(amp, device)
        with torch.no_grad(), amp_ctx.autocast():
            logits_amp, loss_amp = model(idx, targets=idx)
        assert torch.isfinite(loss_amp), f"amp={amp} loss 非有限"
        d = (logits_amp.float() - logits_fp32).abs().max().item()
        assert d < 0.5, f"amp={amp} logits 与 fp32 偏差过大: {d}"
        _p(f"[02] amp={amp} OK: loss={loss_amp.item():.4f}, logits 偏差 {d:.2e}")
    try:
        lc.AmpCtx("fp16", torch.device("cpu"))
        raise AssertionError("CPU 上 fp16 应当被拒绝")
    except RuntimeError:
        _p("[02] amp=fp16 在 CPU 上按预期拒绝")


def check_muon_device(device) -> None:
    """Muon 在目标设备上完成一次正交化更新（CUDA/DDP 兼容性的前置）。"""
    model = lc.build_model(cfg(n_layers=1), seed=0).to(device)
    idx = torch.randint(0, 128, (2, 16), device=device)
    _, loss = model(idx, targets=idx)
    opts, _ = lc.build_optimizer(model, model.cfg, lr=1e-3, optimizer="muon")
    for opt in opts:
        opt.zero_grad(set_to_none=True)
    loss.backward()
    for opt in opts:
        opt.step()
    _, loss2 = model(idx, targets=idx)
    assert torch.isfinite(loss2), "Muon 更新后 loss 非有限"
    assert all(p.device == device for p in model.parameters()), "Muon 更新后参数设备漂移"
    _p(f"[02] Muon OK: loss {loss.item():.4f} -> {loss2.item():.4f} (device={device})")


def check_ddp(ctx) -> None:
    """DDP 自检（仅 torchrun 多进程时执行）：权重同步 / 梯度一致 / checkpoint 无前缀。"""
    if not ctx.enabled:
        _p("[02] DDP 自检跳过（单进程）")
        return
    device = ctx.device
    torch.manual_seed(1234)  # 各 rank 同初始化，DDP 构造时从 rank0 广播
    model = lc.build_model(cfg(n_layers=2), seed=0).to(device)
    net = lc.wrap_ddp(model, ctx)

    # 1) 权重一致性：构造后所有 rank 参数应完全相同（DDP 广播 rank0 初始值）
    with torch.no_grad():
        flat = torch.cat([p.detach().reshape(-1).double() for p in model.parameters()])
        local = torch.stack([flat.sum(), flat.square().sum()]).to(device)
    glob = lc.all_reduce_sum(local, ctx)
    spread = (glob - ctx.world_size * local).abs().max().item()
    assert spread < 1e-6, f"DDP 权重广播不一致: spread={spread}"
    _p(f"[02] DDP 权重广播一致 OK: 参数和/平方和跨 rank 极差 {spread:.2e}")

    # 2) 梯度一致性：各 rank 用同一 batch 前向反向，DDP all_reduce 后梯度必须相同
    g = torch.Generator().manual_seed(999)
    x, y = lc.get_batch(
        torch.randint(0, 128, (512,)), 4, 16, g, device=device
    )
    _, loss = net(x, targets=y)
    loss.backward()
    grad_spread = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        s = lc.all_reduce_sum(p.grad.detach().abs().sum(), ctx)
        local = p.grad.detach().abs().sum()
        grad_spread = max(grad_spread, abs(float(s) - ctx.world_size * float(local)))
    assert grad_spread < 1e-4, f"DDP 梯度不一致: spread={grad_spread}"
    _p(f"[02] DDP 梯度一致 OK: all_reduce 前后梯度极差 {grad_spread:.2e}")

    # 3) checkpoint：DDP 包装保存无 module. 前缀，且可跨设备加载
    if ctx.is_rank0():
        path = lc.out_dir() / "checkpoints" / "_ddp_selftest.pt"
        lc.save_checkpoint(net, model.cfg, path, stage="ddp_selftest", world_size=ctx.world_size)
        import torch as _t

        state = _t.load(path, map_location="cpu", weights_only=False)["model_state"]
        assert not any(k.startswith("module.") for k in state), "checkpoint 仍带 module. 前缀"
        m2, _ = lc.load_checkpoint(path, device=torch.device("cpu"))
        m2 = m2.to(device)
        with torch.no_grad():
            same = all(
                _t.equal(a, b) for a, b in zip(model.parameters(), m2.parameters())
            )
        assert same, "checkpoint 加载后参数不一致"
        path.unlink()
    _p("[02] DDP checkpoint OK: 无 module. 前缀，跨设备加载参数一致")


def main() -> int:
    ap = argparse.ArgumentParser(description="模型自检（CPU / CUDA / DDP）")
    lc.add_device_args(ap)
    args = ap.parse_args()

    ctx = lc.init_distributed(args.device, args.backend)
    device = ctx.device
    if ctx.is_rank0():
        _p(
            f"[02] 设备={device}, 分布式={ctx.enabled}, world_size={ctx.world_size}, "
            f"backend={ctx.backend}"
        )
    try:
        check_shapes(device)
        check_qk_norm(device)
        check_rope_relative(device)
        check_kv_cache_consistency(device)
        check_softcap(device)
        check_mlp_variants(device)
        check_tie_weights()
        check_amp(device)
        check_muon_device(device)
        check_ddp(ctx)

        if ctx.is_rank0():
            cfg_default = lc.ModelConfig()
            model = lc.build_model(cfg_default, seed=0)
            groups = model.num_param_groups()
            n = model.num_params()
            _p(f"[02] 默认配置参数量: {n:,} ({n / 1e6:.2f}M)")
            _p(f"[02] 参数分组: {groups}")
            _p(f"[02] config: {cfg_default.to_dict()}")
            _p("[02] 全部自检通过")
            # 阶段指标（证据白名单可拉回）
            import json

            metrics = {
                "stage": "model_selftest",
                "run_id": lc.run_id(),
                "checks": [
                    "shapes", "qk_norm", "rope_relative", "kv_cache_consistency",
                    "softcap", "mlp_variants", "tie_weights", "amp", "muon_device", "ddp",
                ],
                "all_passed": True,
                "params": n,
                "param_groups": groups,
                "config": cfg_default.to_dict() | {"dist": lc.dist_info(ctx)},
            }
            mfile = lc.out_dir() / "model_metrics.json"
            mfile.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
            _p(f"[02] 阶段指标已写入: {mfile}")
    finally:
        lc.cleanup_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
