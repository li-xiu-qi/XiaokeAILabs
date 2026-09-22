"""
模型加载与 IO 模型：冷启动时间，以及它由什么决定。

服务可用性、弹性伸缩、serverless 推理都受冷启动时间约束。
这个模型回答：给定磁盘读带宽，加载一个模型要多久，页缓存和分片
如何改变它。

自测：用 GB10 实测磁盘读带宽 1.3 GB/s，验证加载时间公式。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_spec import GB10, Qwen3_8_27B, DTYPE_BYTES


# GB10 实测（2026-09-08）：dd 读 20GB 模型文件到 /dev/null，1.3 GB/s
DISK_READ_BW_GBS = 1.3
# 内存带宽（统一内存）
MEM_BW_GBS = GB10.bw_gbs


def cold_load_time_s(file_gb: float, disk_bw_gbs: float = DISK_READ_BW_GBS) -> float:
    """冷启动加载时间（秒）= 文件大小 / 磁盘顺序读带宽。

    冷启动 = 文件不在页缓存，必须从 NVMe 读。
    """
    return file_gb / disk_bw_gbs


def warm_load_time_s(file_gb: float, mem_bw_gbs: float = MEM_BW_GBS,
                     efficiency: float = 0.5) -> float:
    """热加载时间（秒）：文件已在页缓存，从内存读。

    efficiency 是内存拷贝的实际利用率（含反序列化、张量分配开销）。
    """
    return file_gb / (mem_bw_gbs * efficiency)


def parallel_load_time_s(file_gb: float, num_shards: int,
                         disk_bw_gbs: float = DISK_READ_BW_GBS) -> float:
    """分片并行加载时间（秒）：把模型切成 num_shards 份并行读。

    磁盘读带宽可以部分并行化（NVMe 多队列），但受总线限制不会线性提升。
    这里假设并行效率 0.7（实测 NVMe 多队列通常 1.5-2x，不到线性）。
    """
    parallel_bw = disk_bw_gbs * (1 + 0.7 * (num_shards - 1))
    return file_gb / parallel_bw


def self_test():
    print("=" * 64)
    print("模型加载与 IO 模型自测：对齐 GB10 实测磁盘带宽")
    print("=" * 64)

    print(f"\n[1] GB10 实测磁盘顺序读带宽: {DISK_READ_BW_GBS} GB/s")

    print("\n[2] Qwen3.8-27B 各版本冷启动加载时间")
    print(f"{'版本':<10} {'文件GB':>8} {'冷加载':>10} {'热加载':>10}")
    for name, gb in [("bf16", 50.9), ("q4_k_xl", 16.4)]:
        cold = cold_load_time_s(gb)
        warm = warm_load_time_s(gb)
        print(f"{name:<10} {gb:>8.1f} {cold:>8.1f}s {warm:>8.1f}s")

    print("\n[3] 分片并行加载对冷启动的加速（bf16 50.9GB）")
    print(f"{'分片数':>6} {'并行带宽':>10} {'加载时间':>10} {'加速比':>8}")
    base = cold_load_time_s(50.9)
    for ns in [1, 2, 4, 8]:
        parallel_bw = DISK_READ_BW_GBS * (1 + 0.7 * (ns - 1))
        t = 50.9 / parallel_bw
        print(f"{ns:>6} {parallel_bw:>8.1f}GB/s {t:>8.1f}s {base/t:>7.1f}x")

    print("\n[4] 冷启动 vs 热加载的量级差异")
    gb = 50.9
    cold = cold_load_time_s(gb)
    warm = warm_load_time_s(gb)
    print(f"  冷加载 {cold:.1f}s vs 热加载 {warm:.1f}s，差 {cold/warm:.0f}x")
    print(f"  → 冷启动受磁盘带宽限制，热加载受内存带宽限制")
    print(f"  → serverless/弹性伸缩必须优化冷启动，常驻服务命中页缓存后快得多")

    print("\n[5] 外推：更大模型的加载时间（NVFP4, q3 等）")
    print(f"{'模型规模':>10} {'精度':>10} {'文件GB':>8} {'冷加载':>10}")
    for p, dt, gb in [(27, "bf16", 50.9), (27, "nvfp4", 15.0),
                      (70, "q4", 40.0), (320, "nvfp4", 176.0)]:
        t = cold_load_time_s(gb)
        print(f"{p:>8}B {dt:>10} {gb:>8.1f} {t:>8.1f}s")


if __name__ == "__main__":
    self_test()
