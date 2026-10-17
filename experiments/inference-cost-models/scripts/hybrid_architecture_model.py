"""
hybrid_architecture_model.py

混合架构（linear_attention + full_attention）的显存与吞吐外推模型。

逆向自 llmfit（AlexsJones/llmfit, MIT, 35.1k star）的核心公式，
参考实现在 _reference/projects/infra-and-backend/llmfit/。

## 与标准 Transformer 模型的区别

标准 Transformer 每层都有 context-scaled KV Cache。
混合架构（Qwen3.5/3.8、Jamba、Zamba 等）只有 full attention 层有 KV Cache，
linear/state-space 层只有固定大小的循环状态，不计入 per-token KV Cache。

## 核心公式（来自 llmfit）

### KV Cache（models.rs kv_cache_gb）

只对 full attention 层计算：

    kv_bytes = 2 * n_kv_heads * head_dim * ctx * dtype_bytes * full_layers
    kv_gb = kv_bytes / 2^30

linear 层的固定循环状态计入固定 overhead（0.5 GB）。

### 显存（models.rs estimate_memory_gb）

    model_mem = params_b * bpp
    kv_cache = kv_cache_gb(ctx, kv_quant)
    overhead = 0.5  # 包含 CUDA context + 混合架构的 fixed recurrent state
    total = model_mem + kv_cache + overhead

### Decode TPS（fit.rs estimate_tps，带宽路径）

    raw_tps = bandwidth_GB_s / (params_b * bytes_per_param)
    tps = raw_tps * efficiency * run_mode_factor

efficiency 默认 0.55，run_mode_factor 默认 1.0（GPU 模式）。

### Prefill TPS（fit.rs estimate_prefill，算力路径）

    flops_per_token = 2 * params
    usable_flops = tflops * 1e12 * 0.35  # PREFILL_COMPUTE_UTILIZATION
    prefill_tps = usable_flops / flops_per_token

## 与本实验的对比

llmfit 的 efficiency=0.55 是通用默认值。
本实验在 GB10 上用 llama.cpp 实测标定的 decode 效率是 73-86%。

llmfit 的 prefill 用 PREFILL_COMPUTE_UTILIZATION=0.35，
比 LLM-Viewer 的 100% 假设更接近实际，但口径本身是错的。
2026-09-13 用 LLM-Viewer 交叉验证后改为带宽路径：
GB10 的 prefill 有效带宽是 60.7 GB/s（峰值的 22.2%），
见 GB10_PREFILL_EFFECTIVE_BANDWIDTH。
"""

import json
import glob
import math

# GB10 硬件常数（与本实验其他模型一致）
PEAK_FLOPS_BF16 = 89.3e12   # 89.3 TFLOPS（实测）
MEMORY_BANDWIDTH = 273e9    # 273 GB/s（LPDDR5X 256-bit）

# llmfit 的默认参数
LLMFIT_EFFICIENCY = 0.55           # decode 效率（通用默认）
PREFILL_COMPUTE_UTILIZATION = 0.35  # prefill 算力利用率（llmfit 经验常数）
LLMFIT_OVERHEAD_GB = 0.5            # 固定 overhead

# GB10 实测标定的 prefill 有效带宽（2026-09-13 LLM-Viewer 交叉验证反推）
#
# 关键：bs=1/seq=128 下 LLM-Viewer 逐层 roofline 显示所有层都是 memory bound
# （q_proj 算术强度 122.4 FLOP/byte < 平衡点 327），所以它的 prefill 预测
# （730.5 tok/s）是带宽下界，不是算力上界。用它反推「算力利用率」是标签错误。
#
# 正确的归因是带宽。实测：152.04 tok/s 跑完 128 token 需 841.9 ms，
# 期间读 51.1 GB 权重，得有效带宽 60.7 GB/s = 峰值的 22.2%。
#
# 对比 decode 的有效带宽（224.8 GB/s = 82.4% 峰值），prefill 只有 decode 的
# 四分之一。原因是 prefill 每字节配十倍计算（AI 55-100 vs decode 的 1-10），
# 在 GB10 这种低算力机器上 kernel 藏不住访存延迟，再叠加 linear attention
# 循环状态更新、激活读写、softmax 这些 LLM-Viewer 未建模的开销。
#
# 4.80 倍偏差分解：权重字节 1.14x（LLM-Viewer 算 44.9 GB，实际 51.1 GB）
#                  × 有效带宽 4.50x（273 vs 60.7 GB/s）
#
# 注意：60.7 GB/s 只标定于 bs=1/seq=128/vLLM full compile 一个点，
# 换引擎或换 batch 都需重标；batch 增大后部分层转 compute-bound，
# 届时要切回算力路径。不能与 decode 的 224.8 GB/s 共用。
GB10_PREFILL_EFFECTIVE_BANDWIDTH = 60.7e9  # 60.7 GB/s
GB10_DECODE_EFFECTIVE_BANDWIDTH = 224.8e9  # 224.8 GB/s（四引擎均值反推）
# 各量化的实际权重字节（GB）。BF16 与 Q4_K_XL 是实测文件大小，
# Q4_0 无实测文件，按 27B x 0.5 bpp 理论值 13.5 GB。
# 注意 BF16 理论值 54.0 但实测只有 51.1（27B 是约数），
# Q4_K_XL 理论值 15.19 但实测 17.5（bpp 高于 0.5625，已知待办）。
ACTUAL_WEIGHT_GB = {"bf16": 51.1, "q4_k_xl": 17.5, "q4_0": 13.5}


def estimate_prefill_tps_bandwidth(quant, weight_gb=None,
                                   eff_bw=GB10_PREFILL_EFFECTIVE_BANDWIDTH,
                                   seqlen=128):
    """
    Prefill TPS（带宽路径）。

    口径依据：bs=1/seq=128 下 LLM-Viewer 逐层 roofline 显示所有层都是
    memory bound（q_proj 算术强度 122.4 < 平衡点 327），所以 prefill
    受有效带宽限制，不是受峰值算力限制。

        prefill_tps = seqlen / (权重字节 / 有效带宽)

    局限：60.7 GB/s 只标定于 BF16 / bs=1 / seq=128 / vLLM full compile
    一个点。换量化、换引擎、换 batch 都需重标——量化的算术强度更高
    （Q4_K_XL 是 3.55 FLOP/byte），虽仍在平衡点以下属带宽瓶颈，
    但有效带宽本身会变。当前直接沿用 60.7 是外推，不是标定值。
    """
    if weight_gb is None:
        weight_gb = ACTUAL_WEIGHT_GB.get(quant, 54.0)
    return seqlen / (weight_gb / (eff_bw / 1e9))

# 本实验在 GB10 上实测标定的效率（来自 Roofline 模型）
GB10_DECODE_EFFICIENCY_BF16 = 0.86  # 86%（235/273 GB/s）
GB10_DECODE_EFFICIENCY_Q4 = 0.73    # 73%（198/273 GB/s）

# 量化格式的每参数字节（bpp）
QUANT_BPP = {
    "fp32": 4.0, "bf16": 2.0, "fp16": 2.0, "fp8": 1.0,
    "q8_0": 1.0, "q6_k": 0.75, "q5_k_m": 0.625, "q4_k_m": 0.5625,
    "q4_k_xl": 0.5625, "q4_0": 0.5, "q3_k_m": 0.4375, "q2_k": 0.3125,
}

# KV Cache 量化格式的每元素字节
KV_BYTES = {"fp16": 2.0, "fp8": 1.0, "fp4": 0.5}


def kv_cache_gb(n_full_layers, n_kv_heads, head_dim, ctx, kv_quant="fp16"):
    """
    KV Cache 显存（GB），只对 full attention 层计算。

    公式（llmfit models.rs）：
        kv_bytes = 2 * n_kv_heads * head_dim * ctx * dtype_bytes * full_layers
        kv_gb = kv_bytes / 2^30

    与标准 Transformer 的区别：标准模型 full_layers = n_layers，
    混合架构 full_layers < n_layers（只有 full attention 层）。
    """
    dtype_bytes = KV_BYTES.get(kv_quant, 2.0)
    kv_bytes = 2 * n_kv_heads * head_dim * ctx * dtype_bytes * n_full_layers
    return kv_bytes / (1024 ** 3)


def estimate_memory_gb(params_b, bpp, kv_gb, overhead=LLMFIT_OVERHEAD_GB):
    """
    总显存（GB）= 权重 + KV Cache + 固定 overhead。

    overhead 包含 CUDA context 和混合架构的 fixed recurrent state。
    """
    model_mem = params_b * bpp
    return model_mem + kv_gb + overhead


def estimate_decode_tps(params_b, bpp, bandwidth_gbps=MEMORY_BANDWIDTH / 1e9,
                        efficiency=LLMFIT_EFFICIENCY, run_mode_factor=1.0):
    """
    Decode TPS（带宽瓶颈）。

    公式（llmfit fit.rs）：
        raw_tps = bandwidth_GB_s / (params_b * bytes_per_param)
        tps = raw_tps * efficiency * run_mode_factor
    """
    model_bytes_gb = params_b * bpp
    raw_tps = bandwidth_gbps / model_bytes_gb
    return raw_tps * efficiency * run_mode_factor


def estimate_prefill_tps(params_b, tflops=PEAK_FLOPS_BF16 / 1e12,
                         utilization=PREFILL_COMPUTE_UTILIZATION,
                         run_mode_factor=1.0):
    """
    Prefill TPS（算力瓶颈）。

    公式（llmfit fit.rs）：
        flops_per_token = 2 * params
        usable_flops = tflops * 1e12 * utilization
        prefill_tps = usable_flops / flops_per_token
    """
    flops_per_token = 2.0 * params_b * 1e9  # 2 FLOP per param per token
    usable_flops = tflops * 1e12 * utilization
    return (usable_flops / flops_per_token) * run_mode_factor


def qwen3_8_27b_analysis(quant="bf16", ctx=128, batch_size=1,
                          use_gb10_efficiency=True):
    """
    Qwen3.8-27B 在 GB10 上的完整分析（混合架构建模）。

    模型参数（从 config.json 读取）：
        hidden_size=5120, head_dim=256, num_attn_heads=24, num_kv_heads=4
        num_hidden_layers=64, full_attention=16, linear_attention=48
        intermediate_size=17408, vocab_size=248320
    """
    # 模型元数据
    params_b = 27.0  # 27B
    n_layers = 64
    n_full_layers = 16   # 混合架构：只有 16 层 full attention
    n_linear_layers = 48
    n_kv_heads = 4
    head_dim = 256
    hidden_size = 5120
    vocab_size = 248320

    bpp = QUANT_BPP.get(quant, 2.0)

    # KV Cache（只算 full attention 层）
    kv_gb = kv_cache_gb(n_full_layers, n_kv_heads, head_dim, ctx, "fp16")

    # 总显存
    total_mem_gb = estimate_memory_gb(params_b, bpp, kv_gb)

    # Decode TPS
    if use_gb10_efficiency:
        eff = GB10_DECODE_EFFICIENCY_BF16 if bpp >= 2.0 else GB10_DECODE_EFFICIENCY_Q4
    else:
        eff = LLMFIT_EFFICIENCY
    decode_tps = estimate_decode_tps(params_b, bpp, efficiency=eff)

    # Prefill TPS
    prefill_tps = estimate_prefill_tps(params_b)

    return {
        "quant": quant,
        "bpp": bpp,
        "ctx": ctx,
        "kv_cache_gb": round(kv_gb, 4),
        "weight_mem_gb": round(params_b * bpp, 2),
        "total_mem_gb": round(total_mem_gb, 2),
        "decode_tps": round(decode_tps, 2),
        "prefill_tps": round(prefill_tps, 1),
        "decode_efficiency": eff,
        "n_full_layers": n_full_layers,
        "n_linear_layers": n_linear_layers,
    }


def main():
    print("=" * 70)
    print("混合架构模型：Qwen3.8-27B / GB10（逆向自 llmfit）")
    print("=" * 70)
    print(f"硬件：bandwidth={MEMORY_BANDWIDTH/1e9:.0f} GB/s, "
          f"peak_FLOPS={PEAK_FLOPS_BF16/1e12:.1f} TFLOPS")
    print(f"混合架构：64 层 = 16 full_attention + 48 linear_attention")
    print()

    # BF16 和 Q4 两种量化
    for quant in ["bf16", "q4_k_xl", "q4_0"]:
        r = qwen3_8_27b_analysis(quant=quant, ctx=128)

        print(f"--- {quant.upper()} (bpp={r['bpp']}) ---")
        print(f"  权重显存: {r['weight_mem_gb']:.2f} GB")
        print(f"  KV Cache: {r['kv_cache_gb']:.4f} GB (16 full 层, ctx=128)")
        print(f"  总显存: {r['total_mem_gb']:.2f} GB")
        print(f"  Decode TPS: {r['decode_tps']:.2f} tok/s (效率 {r['decode_efficiency']:.0%})")
        print(f"  Prefill TPS: {r['prefill_tps']:.1f} tok/s (llmfit 35% 算力利用率，机理错误口径)")
        prefill_bw = estimate_prefill_tps_bandwidth(quant)
        print(f"  Prefill TPS: {prefill_bw:.1f} tok/s (GB10 标定 60.7 GB/s 有效带宽)")
        print()

    # 与实测对比
    print("=" * 70)
    print("与本实验实测对比（GB10, bs=1, seq=128）")
    print("=" * 70)
    print(f"{'指标':<20} {'BF16 预测':>12} {'BF16 实测':>12} {'Q4 预测':>12} {'Q4 实测':>12}")
    print("-" * 70)

    bf16 = qwen3_8_27b_analysis(quant="bf16")
    q4 = qwen3_8_27b_analysis(quant="q4_k_xl")

    actual_bf16 = {"decode_tps": 4.3, "weight_mem_gb": 50.9}
    actual_q4 = {"decode_tps": 12.1, "weight_mem_gb": 17.5}

    print(f"{'decode tok/s':<20} {bf16['decode_tps']:>10.2f}  {actual_bf16['decode_tps']:>10.2f}  "
          f"{q4['decode_tps']:>10.2f}  {actual_q4['decode_tps']:>10.2f}")
    print(f"{'权重显存 GB':<20} {bf16['weight_mem_gb']:>10.2f}  {actual_bf16['weight_mem_gb']:>10.2f}  "
          f"{q4['weight_mem_gb']:>10.2f}  {actual_q4['weight_mem_gb']:>10.2f}")

    # 误差
    bf16_dec_err = abs(bf16['decode_tps'] - actual_bf16['decode_tps']) / actual_bf16['decode_tps'] * 100
    q4_dec_err = abs(q4['decode_tps'] - actual_q4['decode_tps']) / actual_q4['decode_tps'] * 100
    bf16_mem_err = abs(bf16['weight_mem_gb'] - actual_bf16['weight_mem_gb']) / actual_bf16['weight_mem_gb'] * 100
    q4_mem_err = abs(q4['weight_mem_gb'] - actual_q4['weight_mem_gb']) / actual_q4['weight_mem_gb'] * 100

    print()
    print(f"{'decode 误差':<20} {bf16_dec_err:>10.1f}%  {'':>12} {q4_dec_err:>10.1f}%")
    print(f"{'显存误差':<20} {bf16_mem_err:>10.1f}%  {'':>12} {q4_mem_err:>10.1f}%")

    # KV Cache 对比（标准 vs 混合）
    print()
    print("=" * 70)
    print("KV Cache：混合架构 vs 标准 Transformer（ctx=128）")
    print("=" * 70)
    hybrid_kv = kv_cache_gb(16, 4, 256, 128)  # 只算 16 full 层
    standard_kv = kv_cache_gb(64, 4, 256, 128)  # 全算 64 层
    print(f"  混合架构（16 full 层）: {hybrid_kv:.4f} GB")
    print(f"  标准 Transformer（64 层）: {standard_kv:.4f} GB")
    print(f"  差异: {standard_kv/hybrid_kv:.1f}x（LLM-Viewer 适配器高估的倍数）")

    # 四引擎实测对比（BF16）+ LLM-Viewer 独立锚点
    print()
    print("=" * 70)
    print("四引擎实测 + LLM-Viewer 独立锚点（GB10, BF16, bs=1/bs=32, seq=128）")
    print("=" * 70)
    print(f"{'指标':<28} {'模型预测':>10} {'实测':>10} {'误差':>10}")
    print("-" * 62)

    # decode：四引擎都在带宽瓶颈区，取均值 4.35
    engines_decode = {"llama.cpp": 4.30, "vLLM eager": 4.39,
                      "vLLM full": 4.44, "SGLang": 4.47}
    engines_prefill = {"llama.cpp": 25.8, "vLLM eager": 11.34,
                       "vLLM full": 152.04, "SGLang": 121.53}
    measured_decode = sum(engines_decode.values()) / len(engines_decode)
    best_prefill = max(engines_prefill.values())

    prefill_gb10 = estimate_prefill_tps_bandwidth("bf16")
    # LLM-Viewer 的预测是逐层 roofline（所有层 memory bound），
    # 不是简单的 2*params/峰值算力。730.5 是 LLM-Viewer 实际跑出来的值。
    LLMVIEWER_PREFILL_TPS = 730.5  # LLM-Viewer 预测（GB10, bs=1, seq=128）
    LLMVIEWER_WEIGHT_GB = 44.9     # LLM-Viewer 的权重字节口径（64 层 43.6 + lm_head 1.3）

    dec_err = (bf16['decode_tps'] - measured_decode) / measured_decode * 100
    pre_err_35 = (bf16['prefill_tps'] - best_prefill) / best_prefill * 100
    pre_err_gb10 = (prefill_gb10 - best_prefill) / best_prefill * 100
    pre_err_lv = (LLMVIEWER_PREFILL_TPS - best_prefill) / best_prefill * 100

    print(f"{'decode tok/s (实测均值)':<28} {bf16['decode_tps']:>9.2f}  {measured_decode:>9.2f}  {dec_err:>8.1f}%")
    print()
    print(f"{'prefill tok/s':<28} {'预测':>10} {'实测':>10} {'误差':>10}")
    print(f"{'  LLM-Viewer (逐层 roofline)':<28} {LLMVIEWER_PREFILL_TPS:>9.1f}  {best_prefill:>9.1f}  {pre_err_lv:>8.1f}%")
    print(f"{'  llmfit 35% 算力利用率':<28} {bf16['prefill_tps']:>9.1f}  {best_prefill:>9.1f}  {pre_err_35:>8.1f}%")
    print(f"{'  GB10 60.7 GB/s 带宽路径':<28} {prefill_gb10:>9.1f}  {best_prefill:>9.1f}  {pre_err_gb10:>8.1f}%")

    # 偏差分解
    ratio_total = LLMVIEWER_PREFILL_TPS / best_prefill
    ratio_bytes = ACTUAL_WEIGHT_GB['bf16'] / LLMVIEWER_WEIGHT_GB
    ratio_bw = (MEMORY_BANDWIDTH / 1e9) / (GB10_PREFILL_EFFECTIVE_BANDWIDTH / 1e9)
    print()
    print(f"  偏差分解：{ratio_total:.2f}x = 权重字节 {ratio_bytes:.2f}x × 有效带宽 {ratio_bw:.2f}x")
    print(f"  主导项是有效带宽。LLM-Viewer 少算 {(1-1/ratio_bytes)*100:.0f}% 权重")
    print(f"  （{LLMVIEWER_WEIGHT_GB} GB vs 实际 {ACTUAL_WEIGHT_GB['bf16']} GB），")
    print(f"  且假设 273 GB/s 而 prefill 实测只有 {(GB10_PREFILL_EFFECTIVE_BANDWIDTH/1e9):.1f} GB/s。")
    print()
    print(f"  关键不对称：prefill 有效带宽只有 decode 的")
    print(f"  {(GB10_PREFILL_EFFECTIVE_BANDWIDTH/GB10_DECODE_EFFECTIVE_BANDWIDTH)*100:.0f}%")
    print(f"  （{(GB10_PREFILL_EFFECTIVE_BANDWIDTH/1e9):.1f} vs {(GB10_DECODE_EFFECTIVE_BANDWIDTH/1e9):.1f} GB/s）。")
    print("  原因：prefill 每字节配十倍计算，GB10 低算力下 kernel 藏不住访存延迟。")

    print()
    print("说明：")
    print(f"  decode 在带宽瓶颈区，与引擎无关（四引擎 {min(engines_decode.values())}-{max(engines_decode.values())} tok/s），")
    print(f"  模型预测 {bf16['decode_tps']:.2f} 与实测均值 {measured_decode:.2f} 偏差 {dec_err:.1f}%，达标。")
    print(f"  prefill 由 torch.compile 决定：enforce-eager 仅 {engines_prefill['vLLM eager']} tok/s，")
    print(f"  完整编译后 {engines_prefill['vLLM full']} tok/s，差 {engines_prefill['vLLM full']/engines_prefill['vLLM eager']:.0f} 倍。")
    print()
    print("  归因（bs=1/seq=128 下所有层都是 memory bound，故按带宽口径而非算力口径）：")
    print(f"  用算力利用率建模的三个档位都系统性高估 prefill（{pre_err_gb10:.0f}%~{pre_err_lv:.0f}%），")
    print("  因为 llmfit 的 35% 和 LLM-Viewer 的逐层 roofline 都把带宽不足、激活开销、")
    print("  未建模算子（linear attention 循环状态更新、softmax）揉进一个常数。")
    print(f"  改走带宽路径（实测有效带宽 {(GB10_PREFILL_EFFECTIVE_BANDWIDTH/1e9):.1f} GB/s）后误差归零，")
    print(f"  但要注意这个有效带宽只标定于 bs=1/seq=128/vLLM full 一个点，")
    print("  换引擎或换 batch 都需重标；batch 增大后部分层会转 compute-bound，")
    print("  届时要切回算力路径。不能与 decode 的 224.8 GB/s 共用。")


if __name__ == '__main__':
    main()
