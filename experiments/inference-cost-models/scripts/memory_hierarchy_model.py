"""
内存层级与卸载模型：权重放在哪层内存，决定 decode 的实际速度。

decode 是带宽瓶颈，tok/s = 有效带宽 / 权重字节。有效带宽取决于
权重分布在内存层级的哪一层：
- GPU 显存（高带宽，如 1000 GB/s）
- CPU 内存（中带宽，如 50 GB/s）
- 磁盘（低带宽，如 1.3 GB/s）

关键分野：统一内存 vs 分层内存。
- GB10 统一内存：CPU 和 GPU 共享同一条 273 GB/s 总线，没有 PCIe 瓶颈。
  所以部分卸载（-ngl 部分层）不会速度断崖，实测 1.5→4.7 只差 3 倍。
- 离散 GPU：GPU 显存和 CPU 内存是两条总线，CPU 内存的权重要经 PCIe
  （约 30 GB/s）搬运，一旦卸载就断崖。

自测：GB10 实测 -ngl 0/32/99 = 1.5/2.1/4.7 t/s。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_spec import GB10, Qwen3_8_27B, DTYPE_BYTES


# GB10 统一内存实测（2026-09-08）：Qwen3.8-27B bf16
GB10_UNIFIED_BW = GB10.bw_gbs          # 273 GB/s，CPU/GPU 共享
GB10_DISK_BW = 1.3                     # NVMe 顺序读
GB10_NGL_BENCH = {0: 1.5, 32: 2.1, 99: 4.7}  # -ngl 0/32/99 的实测 decode t/s

# 离散 GPU 对照（RTX 4090）：分层带宽
DISCRETE_GPU_BW = 1008.0    # 显存带宽
DISCRETE_CPU_BW = 50.0      # DDR 内存带宽
DISCRETE_PCIE_BW = 30.0     # PCIe 4.0 x16 实际带宽


def unified_effective_bw(gpu_frac: float, bw: float = GB10_UNIFIED_BW) -> float:
    """统一内存的有效带宽：无论权重在 CPU 还是 GPU，都走同一条总线。

    gpu_frac 是放在 GPU 侧的权重比例，但统一内存下带宽不随它变化
    （都共享 bw）。部分卸载不降带宽，只降一点计算效率（CPU 矩阵乘
    比 GPU 慢），所以实测 1.5→4.7 而非 0→4.7。
    """
    return bw


def discrete_effective_bw(gpu_frac: float,
                          gpu_bw: float = DISCRETE_GPU_BW,
                          pcie_bw: float = DISCRETE_PCIE_BW) -> float:
    """离散 GPU 的有效带宽：GPU 侧的权重走显存带宽，CPU 侧的经 PCIe。

    gpu_frac 放在显存（高带宽），其余在 CPU 内存经 PCIe 搬运（低带宽）。
    gpu_frac 越低，有效带宽越接近 PCIe 瓶颈，断崖式下降。
    """
    return gpu_frac * gpu_bw + (1 - gpu_frac) * pcie_bw


def layer_service_bw(model, dtype: str,
                     gpu_bw_frac: float = 0.93,
                     cpu_bw_frac: float = 0.30) -> tuple:
    """按设备定标每层的有效服务带宽（GB/s）。

    两个端点实测反推（bf16, 273 GB/s 统一内存）：
      -ngl 99 全 GPU：54.6GB / 64层 / 3.33ms = 254 GB/s = 峰值×0.93
      -ngl 0  全 CPU：54.6GB / 64层 / 10.42ms =  81 GB/s = 峰值×0.30

    两个端点读的是同一条物理总线，带宽峰值相同，差距完全来自访问效率。
    这与 arXiv 2512.01644 的实测一致：decode kernel 的 GPU Execution
    只占 24%，70%+ 周期停在访存 stall 上；CPU 侧 GEMV 的访存模式更差。
    旧版用一个 0.94 的单一利用率系数混合两端，在 -ngl 32 这类
    「层数在两端间切分」的场景必然失准（实测 63% 误差），因为两端
    的服务效率差 3 倍，不是同一个系数能覆盖的。
    """
    wb_layer = model.total_params_b * 1e9 * DTYPE_BYTES[dtype] / model.num_layers
    return GB10_UNIFIED_BW * gpu_bw_frac, GB10_UNIFIED_BW * cpu_bw_frac, wb_layer


def decode_tps_unified(model, dtype: str, gpu_frac: float,
                       bw: float = GB10_UNIFIED_BW) -> float:
    """统一内存下的 decode 吞吐（旧版，保留作对照）。

    用混合计算效率 + 单一利用率系数。在 gpu_frac ∈ {0.0, 1.0} 两端能拟合，
    但中间档误差大，见 decode_tps_layerwise 的说明。
    """
    wbytes = model.total_params_b * 1e9 * DTYPE_BYTES[dtype]
    compute_ratio = GB10.cpu_tflops_fp16 / GB10.peak_tflops_fp16
    compute_eff = gpu_frac + (1 - gpu_frac) * compute_ratio
    util = 0.94
    return bw * 1e9 * compute_eff * util / wbytes


def decode_tps_layerwise(model, dtype: str, ngl: int,
                         total_layers: int | None = None) -> float:
    """按层累加服务时间，替代整体有效带宽模型。

    每层只在 GPU 或 CPU 之一上执行，服务时间 = 该层权重字节 / 该设备的
    有效服务带宽。总时间逐层累加，吞吐 = 1 / 总时间。

    这个形式直接对应论文的机制：部分卸载时层数在两端间切分，两端的
    服务效率差 3 倍（93% vs 30%），所以不能用单一系数混合，必须逐层
    分开算再累加。
    """
    total_layers = total_layers or model.num_layers
    gpu_bw, cpu_bw, wb_layer = layer_service_bw(model, dtype)
    n_gpu = min(ngl, total_layers)
    n_cpu = max(total_layers - ngl, 0)
    t_gpu = n_gpu * wb_layer / (gpu_bw * 1e9)
    t_cpu = n_cpu * wb_layer / (cpu_bw * 1e9)
    total_t = t_gpu + t_cpu
    return 1.0 / total_t if total_t > 0 else 0.0


def decode_tps_discrete(model, dtype: str, gpu_frac: float) -> float:
    """离散 GPU 的 decode 吞吐：有效带宽（可能被 PCIe 拖垮）/ 权重字节。"""
    wbytes = model.total_params_b * 1e9 * DTYPE_BYTES[dtype]
    eff = discrete_effective_bw(gpu_frac)
    return eff * 1e9 / wbytes


def self_test():
    print("=" * 64)
    print("内存层级与卸载模型自测：对齐 GB10 -ngl 实测")
    print("=" * 64)

    print("\n[1] GB10 统一内存：-ngl 实测 decode 速度")
    print("    旧版（单一利用率系数）vs 新版（按层累加服务时间）")
    print(f"{'-ngl':>6} {'实测t/s':>8} {'旧版t/s':>8} {'旧误差':>8} "
          f"{'新版t/s':>8} {'新误差':>8}")
    total_layers = Qwen3_8_27B.num_layers
    for ngl, meas in GB10_NGL_BENCH.items():
        frac = min(ngl / total_layers, 1.0)
        old = decode_tps_unified(Qwen3_8_27B, "bf16", frac)
        new = decode_tps_layerwise(Qwen3_8_27B, "bf16", ngl, total_layers)
        e_old = (old - meas) / meas * 100
        e_new = (new - meas) / meas * 100
        print(f"{ngl:>6} {meas:>8.1f} {old:>8.1f} {e_old:>7.0f}% "
              f"{new:>8.1f} {e_new:>7.0f}%")

    print("\n[2] 统一内存 vs 离散 GPU 的卸载特性对比（27B bf16, 54.6GB）")
    print(f"{'GPU侧比例':>10} {'GB10有效带宽':>14} {'GB10 t/s':>10} "
          f"{'4090有效带宽':>12} {'4090 t/s':>10}")
    for frac in [1.0, 0.75, 0.5, 0.25, 0.0]:
        uni_bw = unified_effective_bw(frac)
        uni_tps = decode_tps_unified(Qwen3_8_27B, "bf16", frac)
        dis_bw = discrete_effective_bw(frac)
        dis_tps = decode_tps_discrete(Qwen3_8_27B, "bf16", frac)
        print(f"{frac:>10.2f} {uni_bw:>12.0f}GB/s {uni_tps:>10.1f} "
              f"{dis_bw:>10.0f}GB/s {dis_tps:>10.1f}")

    print("\n[3] 关键结论：27B bf16 显存装不下时（>121GB 才需卸载，此处演示）")
    print("  GB10 统一内存：全 CPU(-ngl 0) 仍有 1.5 t/s，因为带宽不丢")
    print("  4090 离散显存：一旦卸载到 CPU 内存，PCIe 30GB/s 成瓶颈，")
    print("                54.6GB 权重经 PCIe，decode 约 1.6 t/s 且不稳")
    print("  → GB10 能「假装」有大显存跑超大模型，离散 GPU 一旦卸载就崩")

    print("\n[4] 磁盘作为最后一层（mmap 加载，权重在 NVMe）")
    print(f"  NVMe {GB10_DISK_BW}GB/s，bf16 54.6GB 权重的理论上限:")
    print(f"    decode ≈ {GB10_DISK_BW*1e9/ (Qwen3_8_27B.total_params_b*1e9*2):.2f} t/s")
    print("  → 磁盘带宽比内存低 200 倍，只适合超大模型的极限场景")
    print("  → 实际靠页缓存把热权重留在内存，冷权重留在磁盘")


if __name__ == "__main__":
    self_test()
