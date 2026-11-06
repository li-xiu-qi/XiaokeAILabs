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


def batch_efficiency(efficiency: float, batch: int) -> float:
    """当前经验假设：带宽利用率随 batch 提高，最高不超过 100%。"""
    if batch < 1:
        raise ValueError("batch must be >= 1")
    return min(efficiency * (1 + 0.1 * (batch - 1)), 1.0) if batch > 1 else efficiency


def compute_ceiling_tps(hw, model, dtype: str) -> float:
    """全部请求合计的算力上限 (tok/s)，每个输出 token 约需 2×P FLOPs。"""
    peak = hw.peak_tflops_fp16 * 1e12 if dtype in ("bf16", "fp16") else hw.peak_tflops_fp8 * 1e12
    return peak / (2 * model.total_params_b * 1e9)


def serving_throughput(
    hw,
    model,
    dtype: str,
    batch: int,
    efficiency: float,
    compute_ceiling: bool = True,
) -> float:
    """服务总吞吐 (tok/s) = batch × 单流 t/s，但受全 batch 算力上限约束。"""
    eff = batch_efficiency(efficiency, batch)
    bandwidth_tps = single_stream_tps(hw, model, dtype, eff) * batch
    if not compute_ceiling:
        return bandwidth_tps
    return min(bandwidth_tps, compute_ceiling_tps(hw, model, dtype))


def batch_where_compute_bound(hw, model, dtype: str, efficiency: float) -> int:
    """返回未 capped 的带宽吞吐首次达到算力上限的整数 batch。"""
    ceiling = compute_ceiling_tps(hw, model, dtype)

    def uncapped(batch: int) -> float:
        return serving_throughput(hw, model, dtype, batch, efficiency, compute_ceiling=False)

    if uncapped(1) >= ceiling:
        return 1
    lower, upper = 1, 2
    while uncapped(upper) < ceiling:
        lower = upper
        upper *= 2
        if upper > 1_000_000:
            raise RuntimeError("Could not bracket the compute-bound batch")
    while lower + 1 < upper:
        middle = (lower + upper) // 2
        if uncapped(middle) >= ceiling:
            upper = middle
        else:
            lower = middle
    return upper


def self_test():
    print("=" * 64)
    print("吞吐规模化模型自测：单流 vs 服务吞吐")
    print("=" * 64)

    # 单流锚点：q4_k_xl 实测 12.1 t/s → 反推效率
    meas = 12.1
    wbytes = Qwen3_8_27B.total_params_b * 1e9 * DTYPE_BYTES["q4_k_xl"]
    eff = meas * wbytes / (GB10.bw_gbs * 1e9)
    assert abs(single_stream_tps(GB10, Qwen3_8_27B, "q4_k_xl", eff) - meas) < 1e-9
    print(f"\n[1] 单流锚点：q4_k_xl 实测 {meas} t/s → 带宽利用率 {eff*100:.0f}%")

    print("\n[2] 服务吞吐随 batch 增长（q4_k_xl，GB10，仅带宽路径）")
    print(f"{'batch':>6} {'带宽利用率':>10} {'单流 t/s':>10} {'总吞吐 t/s':>11} {'相对单流':>9}")
    for batch in [1, 2, 4, 8, 16, 32, 64]:
        eff_b = batch_efficiency(eff, batch)
        single = single_stream_tps(GB10, Qwen3_8_27B, "q4_k_xl", eff_b)
        total = serving_throughput(
            GB10, Qwen3_8_27B, "q4_k_xl", batch, eff, compute_ceiling=False
        )
        assert abs(total - single * batch) < 1e-9
        print(f"{batch:>6} {eff_b*100:>9.0f}% {single:>10.1f} {total:>11.1f} {total/meas:>8.1f}x")

    print("\n[3] 带宽瓶颈转算力瓶颈的 batch 拐点与上限断言")
    for dtype in ["bf16", "q4_k_xl"]:
        batch = batch_where_compute_bound(GB10, Qwen3_8_27B, dtype, eff)
        ceiling = compute_ceiling_tps(GB10, Qwen3_8_27B, dtype)
        at_boundary = serving_throughput(
            GB10, Qwen3_8_27B, dtype, batch, eff, compute_ceiling=False
        )
        assert at_boundary >= ceiling
        if batch > 1:
            before_boundary = serving_throughput(
                GB10, Qwen3_8_27B, dtype, batch - 1, eff, compute_ceiling=False
            )
            assert before_boundary < ceiling
        for test_batch in [batch, batch + 1, batch + 64]:
            capped = serving_throughput(GB10, Qwen3_8_27B, dtype, test_batch, eff)
            uncapped = serving_throughput(
                GB10, Qwen3_8_27B, dtype, test_batch, eff, compute_ceiling=False
            )
            assert abs(capped - min(uncapped, ceiling)) < 1e-9
            assert capped <= ceiling
        print(f"  {dtype}: batch ≈ {batch} 到拐点，全 batch 上限 {ceiling:.1f} t/s")

    print("\n[4] 结论：GB10 单卡服务吞吐的天花板")
    max_tps = compute_ceiling_tps(GB10, Qwen3_8_27B, "bf16")
    print(f"  算力天花板（所有 batch 加起来）: {max_tps:.0f} t/s")
    print(f"  单流 q4_k_xl 实测: {meas} t/s")
    print(f"  → 理论上服务能把总吞吐拉到 {max_tps/meas:.0f}x 单流（受 KV cache 显存约束）")
    print("  → 公式与边界断言通过：算力上限不再随 batch 重复放大")


if __name__ == "__main__":
    raise SystemExit("UNVALIDATED historical demonstration disabled; see docs/使用边界与审计.md")
