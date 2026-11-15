#!/usr/bin/env python3
"""Benchmark ColBERT multi-vector representation: storage compression & retrieval latency.

Compares:
  - Storage: FP32, FP16, ColBERTv2 residual quantization (1-bit and 2-bit scalar quantization with centroid codebook)
  - Retrieval: brute-force MaxSim vs PLAID-style 3-stage centroid pruning

Synthetic timing estimates are not an end-to-end retrieval benchmark. Recall is not measured.
The default output path is never overwritten.
"""
import json
import math
import os
import random
import statistics
import time
from dataclasses import dataclass, asdict
from typing import List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DIM = 128
NUM_DOCS = 5_000
TOKENS_PER_DOC = 180         # average tokens per document (e.g. 200 words passage)
NUM_QUERIES = 50            # queries for latency benchmark
NUM_CENTROIDS = 1024        # codebook size for benchmark scale (ColBERTv2 uses 32k for full MS MARCO)
PLAID_TOPK_CENTROIDS = 16   # top-k centroids probed per query token


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class CompressionResult:
    method: str
    bytes_per_token: float
    total_bytes_mb: float
    compression_ratio: float
    reconstruction_cosine: float

    def to_dict(self):
        return asdict(self)


@dataclass
class RetrievalResult:
    method: str
    avg_latency_ms: float
    speedup: float
    candidate_pruning_rate: float
    recall_at_10: float | None

    def to_dict(self):
        return asdict(self)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _kmeans_fast(X: np.ndarray, n_clusters: int, max_iter: int = 15, seed: int = 42) -> np.ndarray:
    """Fast spherical k-means for normalized embeddings."""
    rng = np.random.default_rng(seed)
    N, D = X.shape
    init_indices = rng.choice(N, size=n_clusters, replace=False)
    centroids = X[init_indices].copy()

    for _ in range(max_iter):
        sims = X @ centroids.T
        labels = np.argmax(sims, axis=1)
        new_centroids = np.zeros_like(centroids)
        counts = np.bincount(labels, minlength=n_clusters)
        for k in range(n_clusters):
            if counts[k] > 0:
                new_centroids[k] = X[labels == k].sum(axis=0)
            else:
                new_centroids[k] = centroids[k]
        # Normalize centroids
        norms = np.linalg.norm(new_centroids, axis=1, keepdims=True) + 1e-12
        new_centroids /= norms
        if np.allclose(centroids, new_centroids, atol=1e-4):
            break
        centroids = new_centroids
    return centroids


# ---------------------------------------------------------------------------
# 1. Storage Compression Benchmark
# ---------------------------------------------------------------------------
def run_compression_benchmark(seed: int = 42) -> List[CompressionResult]:
    np.random.seed(seed)
    random.seed(seed)

    num_tokens = NUM_DOCS * TOKENS_PER_DOC
    print(f"       Generating {num_tokens:,} normalized {DIM}-dim token embeddings...")
    raw = np.random.randn(min(100_000, num_tokens), DIM).astype(np.float32)
    raw /= np.linalg.norm(raw, axis=1, keepdims=True) + 1e-12

    # 1. FP32
    fp32_bytes = DIM * 4
    fp32_mb = (num_tokens * fp32_bytes) / (1024 * 1024)

    # 2. FP16
    fp16_bytes = DIM * 2
    fp16_mb = (num_tokens * fp16_bytes) / (1024 * 1024)
    fp16_data = raw.astype(np.float16).astype(np.float32)
    fp16_recon = float(np.mean([_cosine(raw[i], fp16_data[i]) for i in range(1000)]))

    # 3. Fit Centroids
    print(f"       Training {NUM_CENTROIDS} spherical centroids...")
    train_sample = raw[:20_000]
    centroids = _kmeans_fast(train_sample, n_clusters=NUM_CENTROIDS, max_iter=15, seed=seed)

    # Assign to nearest centroid
    sims = raw @ centroids.T
    assignments = np.argmax(sims, axis=1)
    assigned_centroids = centroids[assignments]
    residuals = raw - assigned_centroids

    # 4. ColBERTv2 2-bit Residual Quantization
    # Centroid ID: 2 bytes (uint16 supports up to 65,536 centroids)
    # 2-bit per dimension: 128 * 2 = 256 bits = 32 bytes
    # Total per token = 2 + 32 = 34 bytes
    # Quantization bins: 4 levels per dimension based on std/quantiles
    res_std = np.std(residuals, axis=0, keepdims=True) + 1e-8
    res_normed = residuals / res_std
    # 2-bit uniform scalar quantizer in [-2, 2] -> 4 bins: -1.5, -0.5, 0.5, 1.5
    q_levels = np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float32)
    quant_indices = np.clip(np.digitize(res_normed, bins=np.array([-1.0, 0.0, 1.0])) - 1, 0, 3)
    res_2bit_recon = q_levels[quant_indices] * res_std
    recon_2bit = assigned_centroids + res_2bit_recon
    recon_2bit /= np.linalg.norm(recon_2bit, axis=1, keepdims=True) + 1e-12
    cos_2bit = float(np.mean([_cosine(raw[i], recon_2bit[i]) for i in range(2000)]))

    bytes_2bit = 2 + (DIM * 2 // 8)  # 34 bytes
    mb_2bit = (num_tokens * bytes_2bit + NUM_CENTROIDS * DIM * 4) / (1024 * 1024)

    # 5. ColBERTv2 1-bit Residual Quantization
    # Centroid ID: 2 bytes
    # 1-bit per dimension (sign only): 128 * 1 = 128 bits = 16 bytes
    # Total per token = 2 + 16 = 18 bytes
    res_1bit_recon = np.where(residuals >= 0, 1.0, -1.0).astype(np.float32) * (np.mean(np.abs(residuals), axis=0, keepdims=True))
    recon_1bit = assigned_centroids + res_1bit_recon
    recon_1bit /= np.linalg.norm(recon_1bit, axis=1, keepdims=True) + 1e-12
    cos_1bit = float(np.mean([_cosine(raw[i], recon_1bit[i]) for i in range(2000)]))

    bytes_1bit = 2 + (DIM * 1 // 8)  # 18 bytes
    mb_1bit = (num_tokens * bytes_1bit + NUM_CENTROIDS * DIM * 4) / (1024 * 1024)

    results = [
        CompressionResult(
            method="FP32 原生浮点 (未压缩)",
            bytes_per_token=float(fp32_bytes),
            total_bytes_mb=round(fp32_mb, 2),
            compression_ratio=1.0,
            reconstruction_cosine=1.0,
        ),
        CompressionResult(
            method="FP16 半精度浮点",
            bytes_per_token=float(fp16_bytes),
            total_bytes_mb=round(fp16_mb, 2),
            compression_ratio=round(fp32_mb / fp16_mb, 2),
            reconstruction_cosine=round(fp16_recon, 6),
        ),
        CompressionResult(
            method="ColBERTv2 残差量化 (2-bit)",
            bytes_per_token=float(bytes_2bit),
            total_bytes_mb=round(mb_2bit, 2),
            compression_ratio=round(fp32_mb / mb_2bit, 2),
            reconstruction_cosine=round(cos_2bit, 4),
        ),
        CompressionResult(
            method="ColBERTv2 残差量化 (1-bit 极低位宽)",
            bytes_per_token=float(bytes_1bit),
            total_bytes_mb=round(mb_1bit, 2),
            compression_ratio=round(fp32_mb / mb_1bit, 2),
            reconstruction_cosine=round(cos_1bit, 4),
        ),
    ]
    return results


# ---------------------------------------------------------------------------
# 2. Retrieval Latency Benchmark (PLAID Centroid Pruning)
# ---------------------------------------------------------------------------
def run_retrieval_benchmark(seed: int = 42) -> List[RetrievalResult]:
    np.random.seed(seed)
    random.seed(seed)

    print(f"       Setting up corpus for {NUM_DOCS} docs ({TOKENS_PER_DOC} tokens/doc)...")
    Lq = 32
    Ld = TOKENS_PER_DOC

    # Train centroids on sample
    sample_raw = np.random.randn(20_000, DIM).astype(np.float32)
    sample_raw /= np.linalg.norm(sample_raw, axis=1, keepdims=True) + 1e-12
    centroids = _kmeans_fast(sample_raw, n_clusters=NUM_CENTROIDS, max_iter=10, seed=seed)

    # Document token assignments and inverted index: centroid -> list of doc_ids
    # For speed and consistency in benchmark, synthesize documents with clustered structure
    doc_centroid_sets = []
    inverted_index = {c: [] for c in range(NUM_CENTROIDS)}

    # Each doc contains tokens mapped to ~15-20 distinct centroids
    for doc_id in range(NUM_DOCS):
        active_centroids = np.random.choice(NUM_CENTROIDS, size=18, replace=False)
        doc_centroid_sets.append(set(active_centroids))
        for c in active_centroids:
            inverted_index[c].append(doc_id)

    # Document dense tokens (for exact verification of latency on candidates)
    docs_embs = [
        np.random.randn(Ld, DIM).astype(np.float32) / math.sqrt(DIM)
        for _ in range(min(500, NUM_DOCS))
    ]

    # Queries
    queries = [
        np.random.randn(Lq, DIM).astype(np.float32) / math.sqrt(DIM)
        for _ in range(NUM_QUERIES)
    ]

    # ---- 1. Brute-force MaxSim ----
    # Evaluate cost of computing MaxSim against candidate docs
    # On full corpus, brute force computes MaxSim across all 5000 docs
    print(f"       Running Brute-force MaxSim baseline ({NUM_QUERIES} queries)...")
    bf_latencies = []
    for q in queries:
        t0 = time.perf_counter()
        # MaxSim across sample candidate docs scaled to full corpus
        sample_scores = []
        for d in docs_embs[:100]:
            sim_matrix = q @ d.T  # (Lq, Ld)
            maxsim = np.sum(np.max(sim_matrix, axis=1))
            sample_scores.append(maxsim)
        t1 = time.perf_counter()
        # Extrapolate to 5000 docs
        extrapolated_time = (t1 - t0) * (NUM_DOCS / 100.0)
        bf_latencies.append(extrapolated_time * 1000.0)

    avg_bf_ms = statistics.mean(bf_latencies)

    # ---- 2. PLAID Centroid Pruning ----
    # Stage 1: Query token to centroid dot-product (Lq x K) -> fast matrix multiplication
    # Stage 2: Inverted list lookup & candidate accumulation
    # Stage 3: Exact MaxSim on pruned candidates
    print(f"       Running PLAID 3-stage centroid pruning ({NUM_QUERIES} queries)...")
    plaid_latencies = []
    pruning_rates = []

    for q in queries:
        t0 = time.perf_counter()
        # Stage 1: Query x Centroids (32 x 1024)
        q_c_sim = q @ centroids.T  # (32, 1024)
        # Top-k centroids per query token
        top_centroids = np.argpartition(q_c_sim, -PLAID_TOPK_CENTROIDS, axis=1)[:, -PLAID_TOPK_CENTROIDS:]
        unique_centroids = np.unique(top_centroids)

        # Stage 2: Inverted index union / candidate collection
        candidate_counts = {}
        for c in unique_centroids:
            for doc_id in inverted_index[c]:
                candidate_counts[doc_id] = candidate_counts.get(doc_id, 0) + 1

        # Keep candidates that match at least threshold query centroids
        threshold = 3
        pruned_candidates = [doc_id for doc_id, cnt in candidate_counts.items() if cnt >= threshold]
        if len(pruned_candidates) < 50:
            pruned_candidates = sorted(candidate_counts.keys(), key=lambda x: candidate_counts[x], reverse=True)[:100]

        cand_len = len(pruned_candidates)
        prune_rate = 1.0 - (cand_len / NUM_DOCS)
        pruning_rates.append(prune_rate)

        # Stage 3: Exact MaxSim only on pruned candidates (typically ~100-200 docs)
        for cand_id in pruned_candidates[:100]:
            d = docs_embs[cand_id % len(docs_embs)]
            sim_matrix = q @ d.T
            _ = np.sum(np.max(sim_matrix, axis=1))

        t1 = time.perf_counter()
        plaid_latencies.append((t1 - t0) * 1000.0)

    avg_plaid_ms = statistics.mean(plaid_latencies)
    avg_pruning = statistics.mean(pruning_rates)
    speedup = avg_bf_ms / avg_plaid_ms if avg_plaid_ms > 0 else 1.0

    retrieval_results = [
        RetrievalResult(
            method="全量暴力 MaxSim (未剪枝)",
            avg_latency_ms=round(avg_bf_ms, 2),
            speedup=1.0,
            candidate_pruning_rate=0.0,
            recall_at_10=None,
        ),
        RetrievalResult(
            method="PLAID 三级剪枝流水线",
            avg_latency_ms=round(avg_plaid_ms, 2),
            speedup=round(speedup, 2),
            candidate_pruning_rate=round(avg_pruning, 4),
            recall_at_10=None,
        ),
    ]
    return retrieval_results


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "..", "evidence")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "results_residual_compression.json")
    if os.path.exists(out_path):
        raise FileExistsError(f"Refusing to overwrite existing results: {out_path}. Run in a separate copy.")

    print("=" * 80)
    print("ColBERT Residual Compression & PLAID Retrieval Benchmark")
    print("=" * 80)
    print(f"\n[1/2] 运行存储压缩基准 (Dim={DIM}, Docs={NUM_DOCS}, Tokens/Doc={TOKENS_PER_DOC})...")
    comp = run_compression_benchmark()
    print(f"      完成。评估了 {len(comp)} 种存储方案。")

    print(f"\n[2/2] 运行检索剪枝延迟基准 (Queries={NUM_QUERIES}, Probing Centroids={PLAID_TOPK_CENTROIDS})...")
    ret = run_retrieval_benchmark()
    print(f"      完成。评估了 {len(ret)} 种检索路径。")

    payload = {
        "config": {
            "dim": DIM,
            "num_docs": NUM_DOCS,
            "tokens_per_doc": TOKENS_PER_DOC,
            "num_centroids": NUM_CENTROIDS,
            "plaid_topk_centroids": PLAID_TOPK_CENTROIDS,
            "num_queries": NUM_QUERIES,
        },
        "limitations": "Synthetic vectors; random centroid assignments do not correspond to scored document vectors; brute-force time extrapolated from 100 documents; pruned path computes at most 100 candidates; recall is not measured.",
        "compression": [c.to_dict() for c in comp],
        "retrieval": [r.to_dict() for r in ret],
    }

    with open(out_path, "x", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("合成示例汇总 (SYNTHETIC ESTIMATES; RECALL NOT MEASURED)")
    print("=" * 80)

    print("\n--- 1. 多向量存储压缩对比 (以 5,000 篇文档、共 90 万 Token 向量计算) ---")
    print(f"{'存储表示方案':<32} {'单 Token 字节':>12} {'总存储体积 (MB)':>16} {'相对 FP32 压缩比':>16} {'重建余弦相似度':>16}")
    print("-" * 96)
    for c in comp:
        print(f"{c.method:<32} {c.bytes_per_token:>12.1f} B {c.total_bytes_mb:>14.2f} MB {c.compression_ratio:>15.2f}x {c.reconstruction_cosine:>16.4f}")

    print("\n--- 2. 合成检索计时估算 (采样及外推，非全库检索) ---")
    print(f"{'检索执行方案':<28} {'单查询延迟 (ms)':>16} {'相对加速比':>12} {'候选剪枝率':>12} {'Top-10 召回保留率':>18}")
    print("-" * 90)
    for r in ret:
        print(f"{r.method:<28} {r.avg_latency_ms:>16.2f} ms {r.speedup:>11.2f}x {r.candidate_pruning_rate*100:>11.2f}% {'not measured':>18}")

    print(f"\n[OK] 示例结果已保存至: {os.path.abspath(out_path)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
