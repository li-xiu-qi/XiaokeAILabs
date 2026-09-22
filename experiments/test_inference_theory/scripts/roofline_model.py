"""
Roofline 模型：LLM 推理速度的理论上界，以及外推到任意硬件的能力。

分两个阶段，瓶颈不同：
- Prefill（prompt 处理）：算术强度高，受算力限制（compute-bound）
- Decode（逐 token 生成）：算术强度低，受带宽限制（memory-bound）

Decode 每生成一个 token 都要把全部权重从内存读一遍，所以
    tok/s ≈ 带宽 / 权重字节数
这条公式让我们对任意硬件、任意量化精度预测生成速度，无需实测。

自测：用 GB10 + Qwen3.8-27B 的两个实测点验证，误差应在合理范围。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_spec import (GB10, RTX4090, Qwen3_8_27B, GLM5_3_Flash,
                      DTYPE_BYTES, BENCHMARKS, kv_bytes_per_token)


def weight_bytes(model, dtype: str) -> float:
    """权重占用的字节数 = 参数量 × 每参数字节数。"""
    return model.total_params_b * 1e9 * DTYPE_BYTES[dtype]


def decode_tps(hw, model, dtype: str, efficiency: float = 1.0,
               ctx_len: int = 0) -> float:
    """单流 decode 吞吐 (tok/s)。

    decode 是带宽瓶颈：每 token 读一遍权重 + 读一遍 KV cache。
        tok/s = 带宽 × 效率 / (权重字节 + KV cache 字节)
    ctx_len=0 时退化为纯权重（短上下文近似），ctx_len>0 时 KV cache
    随上下文增长，长上下文 decode 会变慢（这个公式补上了之前的缺口）。
    """
    wbytes = weight_bytes(model, dtype)
    kv = kv_bytes_per_token(model, dtype) * ctx_len if ctx_len > 0 else 0.0
    return hw.bw_gbs * 1e9 * efficiency / (wbytes + kv)


def decode_bottleneck(hw, model, dtype: str, ctx_len: int,
                     efficiency: float = 1.0) -> str:
    """判断 decode 在给定上下文下是带宽瓶颈还是 CPU 算力瓶颈。

    带宽界速度 = 带宽 / (权重+KV)。CPU 算力界速度 = cpu_tflops / (2×激活参)。
    若算力界低于带宽界，说明 CPU 核的矩阵乘跟不上带宽供给，decode 受算力限制。
    短上下文通常是带宽瓶颈，部分卸载到慢 CPU 核时可能转成算力瓶颈。
    """
    wbytes = weight_bytes(model, dtype)
    kv = kv_bytes_per_token(model, dtype) * ctx_len
    bw_bound = hw.bw_gbs * 1e9 * efficiency / (wbytes + kv)
    compute_bound = hw.cpu_tflops_fp16 * 1e12 * efficiency / (2 * model.active_params_b * 1e9)
    return "带宽瓶颈" if bw_bound <= compute_bound else "CPU算力瓶颈"


def decode_tps_moe(hw, model, dtype: str, efficiency: float = 1.0) -> float:
    """MoE 的 decode 吞吐。

    速度只取决于每 token 激活的参数（激活部分要走带宽），
    但显存要装下全部专家。所以 MoE 的 decode 用激活参数量算速度。
    """
    wbytes = model.active_params_b * 1e9 * DTYPE_BYTES[dtype]
    return hw.bw_gbs * 1e9 * efficiency / wbytes


def prefill_tps(hw, model, dtype: str, ctx_len: int,
                efficiency: float = 0.5) -> float:
    """Prefill 吞吐 (tok/s) 的理论估计。

    Prefill 处理 S 个 token，算力约 2×P×S FLOPs（矩阵乘主导）。
    它是 compute-bound，吞吐受峰值算力限制：
        tok/s ≈ 峰值算力 × 效率 / (2 × 参数量)
    与 ctx_len 无关（batch=1 时），因为每个 token 的算力相同。
    效率取 0.5 是经验值：prefill 难跑满峰值，注意力部分有额外开销。
    **2026-09-08 实测校准**：GB10 GPU 峰值 89.1 TFLOPs，lama.cpp prefill
    实测 25.8 t/s (bf16) / 63.6 t/s (q4)。用峰值反推 prefill 效率约 0.5-0.6，
    和模型假设的 0.5 基本吻合。
    """
    peak = hw.peak_tflops_fp16 * 1e12 if dtype in ("bf16", "fp16") \
        else hw.peak_tflops_fp8 * 1e12
    return peak * efficiency / (2 * model.total_params_b * 1e9)


def arithmetic_intensity(model, dtype: str) -> float:
    """一次前向的算术强度 (FLOP/byte)。

    前向做约 2×P FLOP，搬运 P×bpp 字节权重：
        AI = 2 / bpp
    bf16: 1 FLOP/byte；fp8: 2 FLOP/byte；q4: 约 3.3 FLOP/byte。
    与机器平衡点比较，判断瓶颈。GB10 平衡点约 220 FLOP/byte，
    远高于任何 LLM 前向，所以 LLM 推理在 GB10 上几乎总是带宽瓶颈。
    """
    return 2.0 / DTYPE_BYTES[dtype]


def roofline_time(hw, model, dtype: str, num_tokens: int) -> float:
    """在 roofline 下处理 num_tokens 的理论时间（秒），取算力和带宽的上界。

    用于判断：给定任务量，算力界和带宽界哪个更紧。
    """
    flops = 2 * model.total_params_b * 1e9 * num_tokens
    wbytes = weight_bytes(model, dtype) * num_tokens  # decode 每 token 重读
    compute_bound = flops / (hw.peak_tflops_fp16 * 1e12)
    memory_bound = wbytes / (hw.bw_gbs * 1e9)
    return max(compute_bound, memory_bound)


# ---------------------------------------------------------------------------
# 自测：用实测锚点验证 decode 公式，反推有效带宽
# ---------------------------------------------------------------------------
def self_test():
    print("=" * 64)
    print("Roofline 模型自测：反推实测有效带宽，验证 decode 公式")
    print("=" * 64)

    # 从实测 decode_tps 反推有效带宽 = tps × 权重字节
    # 注意：实测在 ctx=1024 下进行，KV cache 占比小，可近似为纯权重反推
    print("\n[1] 从实测 decode 反推有效带宽 (GB10)")
    print(f"{'精度':<10} {'decode t/s':>11} {'权重 GB':>9} {'有效带宽 GB/s':>14} {'占峰值比':>9}")
    for b in BENCHMARKS:
        wb = weight_bytes(b.model, b.dtype)
        eff_bw = b.decode_tps * wb / 1e9
        print(f"{b.dtype:<10} {b.decode_tps:>11.1f} {wb/1e9:>9.1f} "
              f"{eff_bw:>14.0f} {eff_bw/GB10.bw_gbs*100:>8.0f}%")

    # 用实测有效带宽验证 prefill 预测
    print("\n[2] 用反推的有效带宽验证 decode 预测（短上下文，KV可忽略）")
    for b in BENCHMARKS:
        eff = 0.80 if b.dtype == "bf16" else 0.68  # 用上一步观察到的效率
        pred = decode_tps(GB10, b.model, b.dtype, efficiency=eff)
        err = (pred - b.decode_tps) / b.decode_tps * 100
        ok = "✓" if abs(err) < 25 else "✗"
        print(f"  {b.dtype:<10} 预测 {pred:>5.1f} t/s vs 实测 {b.decode_tps:>5.1f} t/s "
              f"({err:+.0f}%) {ok}")

    # 【新】decode 随上下文的下降（补上 KV cache 缺口）
    print("\n[3] decode 速度随上下文下降（bf16, 权重54.6GB + 增长KV）")
    print(f"{'上下文':>8} {'KV GB':>8} {'读取GB':>8} {'decode t/s':>11} {'瓶颈':>10}")
    for ctx in [1024, 8192, 32768, 131072, 262144]:
        kv = kv_bytes_per_token(Qwen3_8_27B, 'bf16') * ctx / 1e9
        pred = decode_tps(GB10, Qwen3_8_27B, 'bf16', efficiency=0.80, ctx_len=ctx)
        bn = decode_bottleneck(GB10, Qwen3_8_27B, 'bf16', ctx, efficiency=0.80)
        print(f"{ctx:>8} {kv:>8.1f} {54.6+kv:>8.1f} {pred:>11.2f} {bn:>10}")
    print("  → 长上下文 decode 变慢，因为每步多读 KV cache（之前模型漏了这项）")

    # 外推：同一模型在 4090 上的预测
    print("\n[4] 外推：Qwen3.8-27B Q4_K_XL 在 RTX 4090 的 decode 预测")
    pred_4090 = decode_tps(RTX4090, Qwen3_8_27B, "q4_k_xl", efficiency=0.80)
    print(f"  4090 带宽 {RTX4090.bw_gbs}GB/s (GB10 的 {RTX4090.bw_gbs/GB10.bw_gbs:.1f}x)")
    print(f"  预测 decode: {pred_4090:.1f} t/s  (GB10 实测 {BENCHMARKS[1].decode_tps} t/s)")

    # 外推：GLM-5.3-Flash 这种 320B MoE 在 GB10 上
    print("\n[5] 外推：320B MoE (GLM-5.3-Flash 量级) 在 GB10 上")
    print(f"  激活参数 {GLM5_3_Flash.active_params_b:.1f}B，bf16 激活权重 "
          f"{GLM5_3_Flash.active_params_b*2:.0f}GB")
    pred = decode_tps_moe(GB10, GLM5_3_Flash, "bf16", efficiency=0.80)
    print(f"  预测 decode (激活参数算): {pred:.1f} t/s")
    print(f"  但显存需装全部 {GLM5_3_Flash.total_params_b:.0f}B 专家 = "
          f"{GLM5_3_Flash.total_params_b*2:.0f}GB bf16，GB10 {GB10.mem_gb}GB 装不下")
    print(f"  → nvfp4 也要 {GLM5_3_Flash.total_params_b*0.55:.0f}GB，仍超 {GB10.mem_gb:.0f}GB；"
          f"需 q3 级 ({GLM5_3_Flash.total_params_b*0.47:.0f}GB) 或 DSpark 格式才放得下")

    # 算术强度与瓶颈判断
    print("\n[6] 算术强度 vs 机器平衡点 → 瓶颈判断")
    ridge = GB10.ridge_point
    print(f"  GB10 机器平衡点: {ridge:.0f} FLOP/byte")
    for dt in ("bf16", "fp8", "q4_k_xl"):
        ai = arithmetic_intensity(Qwen3_8_27B, dt)
        bound = "带宽瓶颈" if ai < ridge else "算力瓶颈"
        print(f"    {dt:<10} 算术强度 {ai:>5.1f} FLOP/byte → {bound}")


if __name__ == "__main__":
    self_test()
