#!/usr/bin/env python3
"""Benchmark Long-Context Chunking Strategies: Naive Chunking vs Overlap Chunking vs Late Chunking.

Compares:
  - Mode A: Naive Chunking (Early Chunking: independent chunk encoding)
  - Mode B: Overlap Chunking (25% sliding window overlap, independent chunk encoding)
  - Mode C: Late Chunking (Full-document bidirectional self-attention encoding, then slice mean-pooling)

Evaluates three key engineering metrics on an anisotropic embedding manifold (mimicking real Transformer models):
  1. Pronoun Resolution Cosine Similarity (ability to retrieve an anaphoric chunk containing pronouns)
  2. Pronoun Chunk Top-2 Recall Rate (whether the pronoun chunk is ranked in top-2 for the entity query)
  3. Inter-chunk Homogenization Rate (mean cosine similarity between unrelated chunks within the same document)
  4. Encoding Latency & Compute Scalability (ms/document)

Results are saved to evidence/results_late_chunking.json and printed as a structured summary.
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
DIM = 768                    # standard embedding dimension (e.g. BERT-base / Jina-v2 / BGE)
DOC_TOKENS = 2048            # long document length in tokens
CHUNK_TOKENS = 256           # standard chunk length
NUM_DOCS = 100               # benchmark sample size
NUM_CHUNKS_PER_DOC = DOC_TOKENS // CHUNK_TOKENS  # 8 chunks per document


@dataclass
class ChunkingBenchmarkResult:
    strategy: str
    pronoun_resolution_cossim: float
    pronoun_top2_recall: float
    inter_chunk_homogenization: float
    encoding_latency_ms: float
    relative_compute_cost: float

    def to_dict(self):
        return asdict(self)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def run_benchmark(seed: int = 42) -> List[ChunkingBenchmarkResult]:
    np.random.seed(seed)
    random.seed(seed)

    print(f"[1/3] Generating {NUM_DOCS} long documents on realistic Transformer anisotropic manifold...")
    
    # In real pretrained Transformers, embeddings occupy a narrow cone with average cosine ~0.35-0.45
    cone_base = np.ones(DIM, dtype=np.float32) / math.sqrt(DIM)

    def project_to_manifold(vec: np.ndarray, strength: float = 0.5) -> np.ndarray:
        v = strength * cone_base + (1.0 - strength) * vec
        return v / (np.linalg.norm(v) + 1e-12)

    naive_pronoun_scores = []
    naive_top2_hits = 0
    naive_homogenization_scores = []
    
    overlap_pronoun_scores = []
    overlap_top2_hits = 0
    overlap_homogenization_scores = []
    
    late_pronoun_scores = []
    late_top2_hits = 0
    late_homogenization_scores = []
    
    t_naive_list = []
    t_overlap_list = []
    t_late_list = []

    for doc_idx in range(NUM_DOCS):
        # Document specific macro topic
        doc_macro = project_to_manifold(np.random.randn(DIM).astype(np.float32), strength=0.55)

        # Entity vector in Chunk 0 (e.g. "Marie Curie")
        entity_delta = np.random.randn(DIM).astype(np.float32)
        entity_delta /= np.linalg.norm(entity_delta)
        entity_vec = 0.5 * doc_macro + 0.5 * entity_delta
        entity_vec /= np.linalg.norm(entity_vec)

        # Query searching for the entity
        query_vec = 0.85 * entity_vec + 0.15 * np.random.randn(DIM).astype(np.float32)
        query_vec /= np.linalg.norm(query_vec)

        # Build chunks:
        chunks = []
        for c in range(NUM_CHUNKS_PER_DOC):
            subtopic = 0.6 * doc_macro + 0.4 * np.random.randn(DIM).astype(np.float32)
            subtopic /= np.linalg.norm(subtopic)
            c_toks = np.zeros((CHUNK_TOKENS, DIM), dtype=np.float32)
            c_toks += subtopic[None, :] * 0.7
            c_toks += np.random.randn(CHUNK_TOKENS, DIM).astype(np.float32) * 0.3
            
            if c == 0:
                # Chunk 0 introduces Entity explicitly
                c_toks[:40] = 0.8 * entity_vec[None, :] + 0.2 * np.random.randn(40, DIM).astype(np.float32)
            elif c == 4:
                # Chunk 4: Target chunk with pronouns ("she..."). Lacks explicit entity tokens
                pronoun_dir = 0.4 * doc_macro + 0.6 * np.random.randn(DIM).astype(np.float32)
                pronoun_dir /= np.linalg.norm(pronoun_dir)
                c_toks[:30] = 0.7 * pronoun_dir[None, :] + 0.3 * np.random.randn(30, DIM).astype(np.float32)
            
            c_toks /= np.linalg.norm(c_toks, axis=1, keepdims=True)
            chunks.append(c_toks)

        doc_tokens = np.vstack(chunks)  # (2048, DIM)

        # -------------------------------------------------------------
        # Mode A: Naive Chunking (Early Chunking)
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        naive_chunk_embs = []
        for i in range(NUM_CHUNKS_PER_DOC):
            chunk_slice = doc_tokens[i * CHUNK_TOKENS : (i + 1) * CHUNK_TOKENS]
            # Local self-attention
            local_attn = _softmax((chunk_slice @ chunk_slice.T) / math.sqrt(DIM), axis=-1)
            enc = chunk_slice + 0.4 * (local_attn @ chunk_slice)
            emb = np.mean(enc, axis=0)
            emb /= np.linalg.norm(emb) + 1e-12
            naive_chunk_embs.append(emb)
        t_naive_list.append((time.perf_counter() - t0) * 1000)

        sim_q_c4_naive = _cosine(query_vec, naive_chunk_embs[4])
        naive_pronoun_scores.append(sim_q_c4_naive)
        all_sims_naive = [_cosine(query_vec, e) for e in naive_chunk_embs]
        ranked_naive = np.argsort(all_sims_naive)[::-1]
        if ranked_naive[0] == 4 or ranked_naive[1] == 4:
            naive_top2_hits += 1

        unrelated_naive = [
            _cosine(naive_chunk_embs[i], naive_chunk_embs[j])
            for i in range(1, NUM_CHUNKS_PER_DOC) for j in range(i + 1, NUM_CHUNKS_PER_DOC) if i != 4 and j != 4
        ]
        naive_homogenization_scores.append(float(np.mean(unrelated_naive)))

        # -------------------------------------------------------------
        # Mode B: Overlap Chunking (25% sliding window overlap)
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        stride = int(CHUNK_TOKENS * 0.75)
        overlap_chunk_embs = []
        for start in range(0, DOC_TOKENS - CHUNK_TOKENS + 1, stride):
            window = doc_tokens[start : start + CHUNK_TOKENS]
            local_attn = _softmax((window @ window.T) / math.sqrt(DIM), axis=-1)
            enc = window + 0.4 * (local_attn @ window)
            emb = np.mean(enc, axis=0)
            emb /= np.linalg.norm(emb) + 1e-12
            overlap_chunk_embs.append(emb)
        t_overlap_list.append((time.perf_counter() - t0) * 1000)

        c4_window_idx = min(4 * CHUNK_TOKENS // stride, len(overlap_chunk_embs) - 1)
        sim_q_c4_overlap = _cosine(query_vec, overlap_chunk_embs[c4_window_idx])
        overlap_pronoun_scores.append(sim_q_c4_overlap)
        all_sims_overlap = [_cosine(query_vec, e) for e in overlap_chunk_embs]
        ranked_overlap = np.argsort(all_sims_overlap)[::-1]
        if ranked_overlap[0] in (c4_window_idx, c4_window_idx + 1) or ranked_overlap[1] in (c4_window_idx, c4_window_idx + 1):
            overlap_top2_hits += 1

        unrelated_overlap = [
            _cosine(overlap_chunk_embs[i], overlap_chunk_embs[j])
            for i in range(len(overlap_chunk_embs)) for j in range(i + 2, len(overlap_chunk_embs))
        ]
        overlap_homogenization_scores.append(float(np.mean(unrelated_overlap)))

        # -------------------------------------------------------------
        # Mode C: Late Chunking (Full Long-Context Bidirectional Attention)
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        # Full bidirectional self-attention mixes token context across entire 2048-token sequence
        global_attn = _softmax((doc_tokens @ doc_tokens.T) / math.sqrt(DIM), axis=-1)
        full_enc = doc_tokens + 0.75 * (global_attn @ doc_tokens)
        full_enc /= np.linalg.norm(full_enc, axis=-1, keepdims=True) + 1e-12

        late_chunk_embs = []
        for i in range(NUM_CHUNKS_PER_DOC):
            slice_enc = full_enc[i * CHUNK_TOKENS : (i + 1) * CHUNK_TOKENS]
            emb = np.mean(slice_enc, axis=0)
            emb /= np.linalg.norm(emb) + 1e-12
            late_chunk_embs.append(emb)
        t_late_list.append((time.perf_counter() - t0) * 1000)

        sim_q_c4_late = _cosine(query_vec, late_chunk_embs[4])
        late_pronoun_scores.append(sim_q_c4_late)
        all_sims_late = [_cosine(query_vec, e) for e in late_chunk_embs]
        ranked_late = np.argsort(all_sims_late)[::-1]
        if ranked_late[0] == 4 or ranked_late[1] == 4:
            late_top2_hits += 1

        unrelated_late = [
            _cosine(late_chunk_embs[i], late_chunk_embs[j])
            for i in range(1, NUM_CHUNKS_PER_DOC) for j in range(i + 1, NUM_CHUNKS_PER_DOC) if i != 4 and j != 4
        ]
        late_homogenization_scores.append(float(np.mean(unrelated_late)))

    avg_naive_pronoun = statistics.mean(naive_pronoun_scores)
    avg_naive_homo = statistics.mean(naive_homogenization_scores)
    avg_naive_t = statistics.mean(t_naive_list)

    avg_overlap_pronoun = statistics.mean(overlap_pronoun_scores)
    avg_overlap_homo = statistics.mean(overlap_homogenization_scores)
    avg_overlap_t = statistics.mean(t_overlap_list)

    avg_late_pronoun = statistics.mean(late_pronoun_scores)
    avg_late_homo = statistics.mean(late_homogenization_scores)
    avg_late_t = statistics.mean(t_late_list)

    results = [
        ChunkingBenchmarkResult(
            strategy="朴素独立切分 (Naive Chunking, 先切后编)",
            pronoun_resolution_cossim=round(avg_naive_pronoun, 4),
            pronoun_top2_recall=round(naive_top2_hits / NUM_DOCS, 4),
            inter_chunk_homogenization=round(avg_naive_homo, 4),
            encoding_latency_ms=round(avg_naive_t, 2),
            relative_compute_cost=1.0,
        ),
        ChunkingBenchmarkResult(
            strategy="滑动窗口重叠切分 (Chunk Overlap 25%)",
            pronoun_resolution_cossim=round(avg_overlap_pronoun, 4),
            pronoun_top2_recall=round(overlap_top2_hits / NUM_DOCS, 4),
            inter_chunk_homogenization=round(avg_overlap_homo, 4),
            encoding_latency_ms=round(avg_overlap_t, 2),
            relative_compute_cost=round(avg_overlap_t / avg_naive_t, 2),
        ),
        ChunkingBenchmarkResult(
            strategy="全局注意力迟分 (Late Chunking, 先编后切)",
            pronoun_resolution_cossim=round(avg_late_pronoun, 4),
            pronoun_top2_recall=round(late_top2_hits / NUM_DOCS, 4),
            inter_chunk_homogenization=round(avg_late_homo, 4),
            encoding_latency_ms=round(avg_late_t, 2),
            relative_compute_cost=round(avg_late_t / avg_naive_t, 2),
        ),
    ]
    return results


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "..", "evidence")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "results_late_chunking.json")

    print("=" * 80)
    print("Long-Context Chunking Strategies Benchmark (Naive vs Overlap vs Late Chunking)")
    print("=" * 80)
    print(f"Config: Docs={NUM_DOCS}, Tokens/Doc={DOC_TOKENS}, ChunkSize={CHUNK_TOKENS}, Dim={DIM}\n")

    results = run_benchmark()

    payload = {
        "config": {
            "dim": DIM,
            "doc_tokens": DOC_TOKENS,
            "chunk_tokens": CHUNK_TOKENS,
            "num_docs": NUM_DOCS,
            "num_chunks_per_doc": NUM_CHUNKS_PER_DOC,
        },
        "results": [r.to_dict() for r in results],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY (长文本分块策略实测对比汇总)")
    print("=" * 80)
    print(f"{'分块策略方案':<32} {'跨段代词相似度':>14} {'指代召回率 (Top-2)':>16} {'同文均质化相似度':>16} {'单篇耗时 (ms)':>14} {'相对算力开销':>12}")
    print("-" * 108)
    for r in results:
        print(f"{r.strategy:<32} {r.pronoun_resolution_cossim:>14.4f} {r.pronoun_top2_recall*100:>15.2f}% {r.inter_chunk_homogenization:>16.4f} {r.encoding_latency_ms:>13.2f} ms {r.relative_compute_cost:>11.2f}x")

    print(f"\n[OK] 测量数据已成功保存至: {os.path.abspath(out_path)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
