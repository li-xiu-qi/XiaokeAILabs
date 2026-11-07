# -*- coding: utf-8 -*-
"""
向量索引理论性能模型（核心计算器）

纯闭式解，不依赖 faiss。把向量索引的存储、延迟、召回三个维度统一到一个模型里，
在简化假设下估算存储和计算成本；召回率必须实测。

核心假设：
- 向量数 n，维度 d，原始 float32 基线 = n × d × 4 字节
- 查询 top-k，相似度阈值 r
- 各索引的剪枝结构和压缩策略决定访问量和存储

支持的索引：
- FLAT：暴力扫描，无剪枝无压缩
- SQ8：标量量化，4:1 压缩
- PQ：乘积量化，m 子空间 nbits 比特
- IVF：倒排聚类，nlist 簇 nprobe 探测
- HNSW：多层近邻图，M 连接 efSearch 搜索宽度

本模块给出闭式解公式，参数可配。
"""
from dataclasses import dataclass
import math


@dataclass
class VectorIndexSpec:
    """向量索引参数"""
    n: int = 1_000_000           # 向量数
    d: int = 384                 # 维度
    # PQ
    pq_m: int = 8                # PQ 子空间数
    pq_nbits: int = 8            # 每子空间比特数
    # IVF
    ivf_nlist: int = 4096        # IVF 聚类数
    ivf_nprobe: int = 32         # IVF 探测簇数
    # HNSW
    hnsw_m: int = 16             # HNSW 每节点连接数
    hnsw_ef: int = 64            # HNSW 搜索宽度
    # 硬件常数（需实测标定）。compute_q 是参考维度下的距离计算次数/秒，
    # 换算成 FLOP/s 时必须乘 2 × compute_q_dim，不能混用两种单位。
    compute_q: float = 36_000_000  # 次距离计算/秒（float32 SIMD）
    compute_q_dim: int = 128       # compute_q 的标定维度
    bandwidth_mbps: float = 1928.0  # 顺序读带宽（MB/s）


@dataclass
class VectorIndexMetrics:
    n: int                       # 向量数
    d: int                       # 维度
    # 存储（字节）
    flat_bytes: int
    sq8_bytes: int
    pq_bytes: int
    ivf_flat_bytes: int
    ivf_pq_bytes: int
    hnsw_bytes: int
    # 延迟（毫秒）
    flat_latency_ms: float
    ivf_latency_ms: float
    hnsw_latency_ms: float


def compute_storage(spec: VectorIndexSpec) -> dict:
    """各索引的存储字节（闭式解）"""
    if spec.n <= 0 or spec.d <= 0 or spec.pq_m <= 0 or spec.pq_nbits <= 0:
        raise ValueError("n, d, pq_m and pq_nbits must be positive")
    if spec.d % spec.pq_m:
        raise ValueError("d must be divisible by pq_m")
    raw = spec.n * spec.d * 4  # float32 基线

    # FLAT：原始向量
    flat = raw

    # SQ8：每维 1 字节
    sq8 = spec.n * spec.d

    # PQ：压缩码 + 码本
    pq_codes = spec.n * ((spec.pq_m * spec.pq_nbits + 7) // 8)  # 向上取整到字节
    pq_codebook = spec.pq_m * (2 ** spec.pq_nbits) * (spec.d // spec.pq_m) * 4
    pq = pq_codes + pq_codebook

    # IVF-Flat：聚类中心 + 倒排列表
    ivf_centroids = spec.ivf_nlist * spec.d * 4
    ivf_ids = spec.n * 8  # 每条向量 8 字节 ID
    ivf_flat = ivf_centroids + ivf_ids + raw

    # IVF-PQ：聚类中心 + PQ 码本 + PQ 码 + ID
    ivf_pq = ivf_centroids + pq_codebook + pq_codes + ivf_ids

    # HNSW：原始向量 + 图边（int32，每节点 2M 条 level0 边）
    hnsw_edges = spec.n * 2 * spec.hnsw_m * 4  # 2M 条 level0 边，每条 4 字节
    hnsw_overhead = spec.n * 8  # levels/offset 数组
    hnsw = raw + hnsw_edges + hnsw_overhead

    return {
        "raw": raw,
        "flat": flat,
        "sq8": sq8,
        "pq": pq,
        "ivf_flat": ivf_flat,
        "ivf_pq": ivf_pq,
        "hnsw": hnsw,
    }


def compute_latency(spec: VectorIndexSpec) -> dict:
    """各索引的查询延迟（毫秒），距离次数先统一换算为 FLOP。"""
    if spec.compute_q <= 0 or spec.compute_q_dim <= 0:
        raise ValueError("compute_q and compute_q_dim must be positive")
    flops_per_s = spec.compute_q * 2 * spec.compute_q_dim

    # FLAT：访问 n 个向量，每次 2d 浮点运算
    flat_ops = spec.n * 2 * spec.d
    flat_ms = flat_ops / flops_per_s * 1000

    # IVF：访问 nprobe × n/nlist 个向量，每次 2d 浮点运算
    ivf_access = spec.ivf_nprobe * spec.n // spec.ivf_nlist
    ivf_ops = ivf_access * 2 * spec.d
    ivf_ms = ivf_ops / flops_per_s * 1000

    # HNSW：访问 O(ef × log n) 个节点，每次 2d 浮点运算
    log_n = max(1, int(math.log2(spec.n)))
    hnsw_access = spec.hnsw_ef * log_n
    hnsw_ops = hnsw_access * 2 * spec.d
    hnsw_ms = hnsw_ops / flops_per_s * 1000

    return {
        "flat_ms": flat_ms,
        "ivf_ms": ivf_ms,
        "hnsw_ms": hnsw_ms,
        "flat_access": spec.n,
        "ivf_access": ivf_access,
        "hnsw_access": hnsw_access,
    }


def compute(spec: VectorIndexSpec) -> VectorIndexMetrics:
    """给定参数，递推全部指标。"""
    storage = compute_storage(spec)
    latency = compute_latency(spec)

    return VectorIndexMetrics(
        n=spec.n,
        d=spec.d,
        flat_bytes=storage["flat"],
        sq8_bytes=storage["sq8"],
        pq_bytes=storage["pq"],
        ivf_flat_bytes=storage["ivf_flat"],
        ivf_pq_bytes=storage["ivf_pq"],
        hnsw_bytes=storage["hnsw"],
        flat_latency_ms=latency["flat_ms"],
        ivf_latency_ms=latency["ivf_ms"],
        hnsw_latency_ms=latency["hnsw_ms"],
    )


def _selftest():
    """公式自洽性检查"""
    spec = VectorIndexSpec()
    m = compute(spec)

    # FLAT 存储 = 原始向量
    assert m.flat_bytes == spec.n * spec.d * 4

    # SQ8 存储 = 原始/4
    assert m.sq8_bytes == spec.n * spec.d

    # PQ 存储 << FLAT
    assert m.pq_bytes < m.flat_bytes / 10, "PQ 应压缩 10 倍以上"

    # IVF-Flat > FLAT（多了聚类中心和 ID）
    assert m.ivf_flat_bytes > m.flat_bytes

    # HNSW > FLAT（多了图边），且与验证脚本共用 int32 邻居 id 口径
    expected_hnsw = spec.n * spec.d * 4 + spec.n * 2 * spec.hnsw_m * 4 + spec.n * 8
    assert m.hnsw_bytes == expected_hnsw

    # 延迟单位：d=128 的 36M 距离/s 等效为 9.216G FLOP/s
    flops_per_s = spec.compute_q * 2 * spec.compute_q_dim
    assert flops_per_s == 9_216_000_000
    expected_flat_ms = spec.n * 2 * spec.d / flops_per_s * 1000
    assert abs(m.flat_latency_ms - expected_flat_ms) < 1e-12
    assert abs(expected_flat_ms - 83.3333333333) < 1e-6

    # 低比特编码每条向量也需按字节向上取整
    packed = VectorIndexSpec(n=10, d=12, pq_m=3, pq_nbits=4)
    expected_codes = 10 * 2  # ceil(3 * 4 / 8)
    expected_codebook = 4 * 12 * (2 ** 4)
    assert compute_storage(packed)["pq"] == expected_codes + expected_codebook
    try:
        compute_storage(VectorIndexSpec(d=10, pq_m=3))
    except ValueError:
        pass
    else:
        raise AssertionError("invalid PQ dimensions must be rejected")

    # IVF 延迟 < FLAT
    assert m.ivf_latency_ms < m.flat_latency_ms

    # HNSW 延迟 << FLAT
    assert m.hnsw_latency_ms < m.flat_latency_ms / 100

    print("selftest 全部通过")
    # 演示
    print(f"\nn={m.n:,}  d={m.d}")
    print(f"存储: FLAT={m.flat_bytes/1024/1024:.1f} MiB  "
          f"SQ8={m.sq8_bytes/1024/1024:.1f} MiB  "
          f"PQ={m.pq_bytes/1024/1024:.1f} MiB  "
          f"HNSW={m.hnsw_bytes/1024/1024:.1f} MiB")
    print(f"估算延迟（简化假设，非实测）: FLAT={m.flat_latency_ms:.1f} ms  "
          f"IVF={m.ivf_latency_ms:.2f} ms  "
          f"HNSW={m.hnsw_latency_ms:.2f} ms")
    print("召回率：需要使用实际数据测量")


if __name__ == "__main__":
    _selftest()
