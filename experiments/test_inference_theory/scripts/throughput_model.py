"""
吞吐规模化模型：单流 vs 服务吞吐，batch 如何把带宽利用率拉满。

单流 decode 只跑一个请求，权重读一遍只服务一个 token，带宽利用率低
（GB10 上实测只有峰值的 68~86%）。

服务场景把 B 个请求放一批：
- 权重读一遍，同时算 B 个 token 的下一步 → 带宽摊薄
- 吞吐 ≈ B × 单流 t/s，直到算力成为瓶颈

这个模型回答：同一块卡，服务吞吐比单流高多少，batch 加到多少会碰到算力墙。

自测：单流点用实测 12.1 t/s 锚定，服务吞吐按带宽利用率推算。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_spec import GB10, Qwen3_8_27B, DTYPE_BYTES


def single_stream_tps(hw, model, dtype: str, efficiency: float) -> float:
    """单流 decode 吞吐 (tok/s)。"""
    wbytes = model.total_params_b * 1e9 * DTYPE_BYTES[dtype]
    return hw.bw_gbs * 1e9 * efficiency / wbytes


def serving_throughput(hw, model, dtype: str, batch: int,
                       efficiency: float, compute_ceiling: bool = True) -> float:
    """服务总吞吐 (tok/s) = batch × 单流 t/s，但受算力天花板限制。

    带宽利用率随 batch 提高（更多请求摊薄权重读取），
    但总算力 2×P×batch FLOPs/token 不能超过峰值算力。
    """
    # batch 越大，带宽利用率越接近峰值
    eff = min(efficiency * (1 + 0.1 * (batch - 1)), 1.0) if batch > 1 else efficiency
    tps = single_stream_tps(hw, model, dtype, eff) * batch
    if compute_ceiling:
        # 算力天花板：每 token 2×P FLOPs，峰值算力限制总吞吐
        peak = hw.peak_tflops_fp16 * 1e12 if dtype in ("bf16", "fp16") else hw.peak_tflops_fp8 * 1e12
        max_tps = peak / (2 * model.total_params_b * 1e9)
        tps = min(tps, max_tps * batch)  # 每请求至少还能到单流的算力界
    return tps


def batch_where_compute_bound(hw, model, dtype: str, efficiency: float) -> int:
    """batch 加到多少会从带宽瓶颈转到算力瓶颈。

    带宽界吞吐 = B × 单流；算力界吞吐 = 峰值算力 / (2×P)。
    两者相等时的 B 就是拐点。
    """
    single = single_stream_tps(hw, model, dtype, efficiency)
    peak = hw.peak_tflops_fp16 * 1e12 if dtype in ("bf16", "fp16") else hw.peak_tflops_fp8 * 1e12
    max_tps = peak / (2 * model.total_params_b * 1e9)
    return int(max_tps / single)


def self_test():
    print("=" * 64)
    print("吞吐规模化模型自测：单流 vs 服务吞吐")
    print("=" * 64)

    # 单流锚点：q4_k_xl 实测 12.1 t/s → 反推效率
    meas = 12.1
    wbytes = Qwen3_8_27B.total_params_b * 1e9 * DTYPE_BYTES["q4_k_xl"]
    eff = meas * wbytes / (GB10.bw_gbs * 1e9)
    print(f"\n[1] 单流锚点：q4_k_xl 实测 {meas} t/s → 带宽利用率 {eff*100:.0f}%")

    print("\n[2] 服务吞吐随 batch 增长（q4_k_xl，GB10）")
    print(f"{'batch':>6} {'带宽利用率':>10} {'单流 t/s':>10} {'总吞吐 t/s':>11} {'相对单流':>9}")
    for b in [1, 2, 4, 8, 16, 32, 64]:
        eff_b = min(eff * (1 + 0.1 * (b - 1)), 1.0) if b > 1 else eff
        single = single_stream_tps(GB10, Qwen3_8_27B, "q4_k_xl", eff_b)
        total = single * b
        print(f"{b:>6} {eff_b*100:>9.0f}% {single:>10.1f} {total:>11.1f} {total/meas:>8.1f}x")

    print("\n[3] 带宽瓶颈转算力瓶颈的 batch 拐点")
    for dt in ["bf16", "q4_k_xl"]:
        x = batch_where_compute_bound(GB10, Qwen3_8_27B, dt, eff)
        print(f"  {dt}: batch ≈ {x} 时碰到算力天花板")

    print("\n[4] 结论：GB10 单卡服务吞吐的天花板")
    peak = GB10.peak_tflops_fp16 * 1e12
    max_tps = peak / (2 * Qwen3_8_27B.total_params_b * 1e9)
    print(f"  算力天花板（所有 batch 加起来）: {max_tps:.0f} t/s")
    print(f"  单流 q4_k_xl 实测: {meas} t/s")
    print(f"  → 理论上服务能把总吞吐拉到 {max_tps/meas:.0f}x 单流（受 KV cache 显存约束）")


if __name__ == "__main__":
    self_test()
