"""UNVALIDATED legacy estimates: not for performance prediction or bottleneck attribution.
Weight and KV dtype are independent. See docs/使用边界与审计.md.
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
               ctx_len: int = 0, kv_dtype: str = "bf16") -> float:
    """单流 decode 吞吐 (tok/s)。

    假设每 token 从内存读取一遍权重和 KV cache，只计算带宽路径的玩具值；
    这不证明 decode 的实际瓶颈，也不是经验证的速度预测。
        tok/s = 带宽 × 效率 / (权重字节 + KV cache 字节)
    ctx_len=0 时退化为纯权重（短上下文近似），ctx_len>0 时 KV cache
    随上下文增长，长上下文 decode 会变慢（这个公式补上了之前的缺口）。
    """
    wbytes = weight_bytes(model, dtype)
    kv = kv_bytes_per_token(model, kv_dtype) * ctx_len if ctx_len > 0 else 0.0
    return hw.bw_gbs * 1e9 * efficiency / (wbytes + kv)


def decode_bottleneck(hw, model, dtype: str, ctx_len: int,
                     efficiency: float = 1.0, kv_dtype: str = "bf16") -> str:
    raise NotImplementedError("unvalidated method; see docs/使用边界与审计.md")


def decode_tps_moe(hw, model, dtype: str, efficiency: float = 1.0) -> float:
    raise NotImplementedError("unvalidated method; see docs/使用边界与审计.md")


def prefill_tps(hw, model, dtype: str, ctx_len: int,
                efficiency: float = 0.5) -> float:
    raise NotImplementedError("unvalidated method; see docs/使用边界与审计.md")


def arithmetic_intensity(model, dtype: str) -> float:
    """一次前向的算术强度 (FLOP/byte)。

    前向做约 2×P FLOP，搬运 P×bpp 字节权重：
        AI = 2 / bpp
    bf16: 1 FLOP/byte；fp8: 2 FLOP/byte；q4: 约 3.3 FLOP/byte。
    仅在假设每 token 重读全部权重、忽略激活与缓存流量时成立；
    不能据此推断真实模型或具体硬件的性能瓶颈。
    """
    return 2.0 / DTYPE_BYTES[dtype]


def roofline_time(hw, model, dtype: str, num_tokens: int) -> float:
    """假定每 token 读取全部权重并执行 2P 操作时的玩具算式。

    它不是实际任务延迟下界：缓存、批量权重复用及 MoE 路由会改变
    读取和计算量；其他开销也不在这个算式内。不得用于瓶颈归因。
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
    raise NotImplementedError("unvalidated method; see docs/使用边界与审计.md")


if __name__ == "__main__":
    raise SystemExit("UNVALIDATED: legacy roofline estimates are disabled; see docs/使用边界与审计.md")
