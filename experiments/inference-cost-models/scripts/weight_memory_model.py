"""
权重显存模型：模型权重占用多少显存，以及量化如何压缩它。

权重显存 = 总参数量 × 每参数字节数。
这是推理的固定成本，与上下文长度、batch 无关（batch 只增 KV cache）。

量化把每参数字节数降下来，显存近似线性下降，decode 速度近似线性上升
（因为 decode 是带宽瓶颈，带宽搬运的字节数少了）。

自测：和实测权重文件大小对齐（bf16 50.9GB、q4_k_xl 16.4GB）。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_spec import (GB10, Qwen3_8_27B, GLM5_3_Flash, DTYPE_BYTES, BENCHMARKS)


def weight_gb(model, dtype: str) -> float:
    """权重显存 (GB)。"""
    return model.total_params_b * DTYPE_BYTES[dtype]


def fits_in_memory(model, dtype: str, mem_gb: float, overhead_gb: float = 2.0) -> bool:
    """权重 + 额外开销（KV cache、激活、框架）能否装进显存。

    overhead_gb 是上下文、激活值、CUDA 上下文等固定开销的估计。
    """
    return weight_gb(model, dtype) + overhead_gb <= mem_gb


def min_dtype_to_fit(model, mem_gb: float, overhead_gb: float = 2.0):
    """从高精度到低精度找第一个能装下的量化档位。"""
    order = ["fp32", "bf16", "fp8", "q8_0", "q6_k", "q5_k_m",
             "q4_k_xl", "q4_k_m", "q3_k_m", "q2_k"]
    for dt in order:
        if fits_in_memory(model, dt, mem_gb, overhead_gb):
            return dt
    return None


def self_test():
    print("=" * 64)
    print("权重显存模型自测：对齐实测文件大小")
    print("=" * 64)

    print("\n[1] Qwen3.8-27B 各量化档位显存 (GB10 121GB)")
    print(f"{'精度':<10} {'B/param':>8} {'权重 GB':>9} {'装得下?':>8}")
    for dt in ["fp32", "bf16", "fp8", "q8_0", "q6_k", "q5_k_m", "q4_k_xl", "q4_k_m", "q3_k_m", "q2_k"]:
        gb = weight_gb(Qwen3_8_27B, dt)
        fit = "✓" if fits_in_memory(Qwen3_8_27B, dt, GB10.mem_gb) else "✗"
        print(f"{dt:<10} {DTYPE_BYTES[dt]:>8.2f} {gb:>9.1f} {fit:>8}")

    print("\n[2] 对齐实测文件大小")
    for b in BENCHMARKS:
        pred = weight_gb(b.model, b.dtype)
        err = (pred - b.file_gb) / b.file_gb * 100
        ok = "✓" if abs(err) < 5 else "~"
        print(f"  {b.dtype:<10} 预测 {pred:>5.1f}GB vs 实测 {b.file_gb:>5.1f}GB ({err:+.0f}%) {ok}")

    print("\n[3] 稠密大模型 vs GB10 内存上限：量化到能装下的最低档")
    for m in (Qwen3_8_27B, GLM5_3_Flash):
        dt = min_dtype_to_fit(m, GB10.mem_gb)
        if dt:
            print(f"  {m.name}: 需 ≤ {dt} ({weight_gb(m, dt):.0f}GB) 才装得下 {GB10.mem_gb:.0f}GB")
        else:
            print(f"  {m.name}: {GB10.mem_gb:.0f}GB 装不下任何档位（最低 q2 也要 "
                  f"{weight_gb(m, 'q2_k'):.0f}GB）")

    print("\n[4] 量化对 decode 速度的影响（带宽瓶颈，速度≈反比于 B/param）")
    base = BENCHMARKS[1].decode_tps  # q4_k_xl 实测 12.1
    base_bpp = DTYPE_BYTES["q4_k_xl"]
    for dt in ["bf16", "fp8", "q8_0", "q6_k", "q5_k_m", "q4_k_xl"]:
        pred = base * base_bpp / DTYPE_BYTES[dt]
        print(f"  {dt:<10} {DTYPE_BYTES[dt]:.2f} B/param → 预测 decode ≈ {pred:.1f} t/s")


if __name__ == "__main__":
    self_test()
