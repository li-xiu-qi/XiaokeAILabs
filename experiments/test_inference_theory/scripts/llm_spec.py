"""
LLM 推理性能模型的共享规格与实测锚点。

所有理论模型脚本（roofline_model / weight_memory_model / ...）都从这里取
硬件参数、模型配置和实测基准。改锚点只改这一个文件。

锚点来源：
- 硬件 DGX Spark (GB10) 实测，2026-09-07
- 模型 Qwen3.8-27B 配置取自 ~/models/Qwen3.8-27B-BF16/config.json
- 实测吞吐取自 llama.cpp cuBLAS 构建，见 DGX-Spark-llama.cpp部署实录.md
"""
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# 硬件规格
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Hardware:
    name: str
    mem_gb: float            # 统一内存总量 (GB, 十进制 1GB=1e9B)
    bw_gbs: float            # 内存带宽峰值 (GB/s, 十进制)
    peak_tflops_fp16: float  # FP16/BF16 峰值算力 (TFLOPs, 1T=1e12)
    peak_tflops_fp8: float   # FP8 峰值算力
    cpu_tflops_fp16: float = 2.0  # CPU 侧 FP16 算力（部分卸载时用）
    notes: str = ""

    @property
    def ridge_point(self) -> float:
        """机器平衡点 (FLOP/byte)：算力/带宽之比。
        算术强度超过它，kernel 受算力限制；低于它，受带宽限制。"""
        return (self.peak_tflops_fp16 * 1e12) / (self.bw_gbs * 1e9)


# DGX Spark / NVIDIA GB10
# 带宽 273 GB/s 是 LPDDR5X 统一内存的公认峰值（社区实测一致，见 NVIDIA 论坛）。
# 算力 2026-09-08 实测（matmul benchmark, torch 2.14.0+cu130）：
#   GPU bf16 89.1 TFLOPs, fp16 88.9 TFLOPs
#   CPU bf16 40.7 TFLOPs (20核Armv9.2合计), fp32 12.5 TFLOPs
# cpu_tflops_fp16 用于部分卸载时判断 decode 是否从带宽瓶颈转成 CPU 算力瓶颈。
GB10 = Hardware(
    name="NVIDIA GB10 (DGX Spark)",
    mem_gb=121.0,
    bw_gbs=273.0,
    peak_tflops_fp16=89.1,   # 2026-09-08 实测
    peak_tflops_fp8=178.2,   # 估为 fp16 两倍
    cpu_tflops_fp16=40.7,    # 2026-09-08 实测（20核合计）
    notes="统一内存，sm_121a，需 cuBLAS 编译 llama.cpp",
)

# 对照机：单卡 4090（24GB），用于外推示例
RTX4090 = Hardware(
    name="RTX 4090",
    mem_gb=24.0,
    bw_gbs=1008.0,
    peak_tflops_fp16=165.0,
    peak_tflops_fp8=330.0,
    cpu_tflops_fp16=2.0,
)


# ---------------------------------------------------------------------------
# 模型规格：Qwen3.8-27B（配置取自 config.json）
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelConfig:
    name: str
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int   # GQA/MQA 时 < num_attention_heads
    head_dim: int
    intermediate_size: int     # SwiGLU 每个 expert 的 intermediate
    vocab_size: int
    max_position_embeddings: int
    total_params_b: float      # 总参数量（十亿）。GGUF F16 报告 54.6GB/2 = 27.3B
    moe: bool = False
    num_experts: int = 1
    active_experts: int = 1    # MoE 每 token 激活的 expert 数

    @property
    def active_params_b(self) -> float:
        """MoE 每 token 实际参与计算的参数量。稠密模型等于总参数。"""
        if not self.moe:
            return self.total_params_b
        # 共享部分 + 激活的 expert 部分（粗略按比例折算）
        shared = self.total_params_b * 0.3
        routed = (self.total_params_b * 0.7) * self.active_experts / self.num_experts
        return shared + routed


Qwen3_8_27B = ModelConfig(
    name="Qwen3.8-27B",
    num_layers=64,
    hidden_size=5120,
    num_attention_heads=24,
    num_key_value_heads=4,
    head_dim=256,
    intermediate_size=17408,
    vocab_size=248320,
    max_position_embeddings=262144,
    total_params_b=27.3,
    moe=False,
)

# 用作对比的 MoE 示例（GLM-5.3-Flash 量级，供外推与 MoE 模型参考）
GLM5_3_Flash = ModelConfig(
    name="GLM-5.3-Flash (示例)",
    num_layers=48,
    hidden_size=6144,
    num_attention_heads=64,
    num_key_value_heads=8,
    head_dim=128,
    intermediate_size=12288,
    vocab_size=151552,
    max_position_embeddings=131072,
    total_params_b=320.0,
    moe=True,
    num_experts=64,
    active_experts=8,
)


# ---------------------------------------------------------------------------
# 量化精度：每参数字节数
# ---------------------------------------------------------------------------
DTYPE_BYTES = {
    "fp32": 4.0,
    "bf16": 2.0,
    "fp16": 2.0,
    "fp8": 1.0,
    "int8": 1.0,
    "q8_0": 1.06,      # GGUF 8-bit，含 scale
    "q6_k": 0.74,
    "q5_k_m": 0.64,
    "q4_k_m": 0.58,
    "q4_k_xl": 0.60,   # unsloth UD-Q4_K_XL，实测 16.4GB/27.3B = 0.60 B/param
    "q4_0": 0.56,
    "q3_k_m": 0.47,
    "q2_k": 0.38,
    "nvfp4": 0.55,     # NVFP4 含 scale，约 4.4 bit
}


# ---------------------------------------------------------------------------
# 实测锚点（llama.cpp cuBLAS 构建，GB10，2026-09-07）
# 用于自测：每个理论模型跑完要能和这些值对上。
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Benchmark:
    model: ModelConfig
    hw: Hardware
    dtype: str
    decode_tps: float      # 单流生成 tok/s
    prefill_tps: float     # prompt 处理 tok/s
    ctx_len: int
    file_gb: float         # 权重文件大小 (GB)
    notes: str = ""


BENCHMARKS = [
    Benchmark(
        model=Qwen3_8_27B, hw=GB10, dtype="bf16",
        decode_tps=4.3, prefill_tps=25.8, ctx_len=1024, file_gb=50.9,
        notes="F16 GGUF 全量",
    ),
    Benchmark(
        model=Qwen3_8_27B, hw=GB10, dtype="q4_k_xl",
        decode_tps=12.1, prefill_tps=63.6, ctx_len=1024, file_gb=16.4,
        notes="unsloth UD-Q4_K_XL 量化",
    ),
]


# ---------------------------------------------------------------------------
# KV cache 每 token 每 batch 的字节数（GQA/MQA 通用公式）
# KV = 2 (K和V) × num_layers × num_kv_heads × head_dim × bytes
# ---------------------------------------------------------------------------
def kv_bytes_per_token(model: ModelConfig, dtype: str = "bf16") -> float:
    bpp = DTYPE_BYTES[dtype]
    return 2 * model.num_layers * model.num_key_value_heads * model.head_dim * bpp


if __name__ == "__main__":
    print("=== 硬件 ===")
    for hw in (GB10, RTX4090):
        print(f"{hw.name}: {hw.mem_gb}GB, {hw.bw_gbs}GB/s, "
              f"机器平衡点 {hw.ridge_point:.1f} FLOP/byte")
    print("\n=== 模型 ===")
    for m in (Qwen3_8_27B, GLM5_3_Flash):
        print(f"{m.name}: {m.total_params_b}B 总参, {m.active_params_b:.1f}B 激活")
    print("\n=== KV cache (bf16, 每 token 每 batch) ===")
    for m in (Qwen3_8_27B, GLM5_3_Flash):
        kb = kv_bytes_per_token(m)
        print(f"{m.name}: {kb:.0f} B/token = {kb/1024:.2f} KB/token")
    print("\n=== 实测锚点 ===")
    for b in BENCHMARKS:
        print(f"{b.model.name} {b.dtype}: decode {b.decode_tps} t/s, "
              f"prefill {b.prefill_tps} t/s, file {b.file_gb}GB")
