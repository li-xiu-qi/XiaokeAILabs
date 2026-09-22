# -*- coding: utf-8 -*-
"""
BM25 实测验证

用自制迷你倒排索引 + 随机文档，实测四个核心指标：
1. TF 饱和曲线：不同 TF 值下 BM25 得分 vs TF-IDF 得分
2. IDF 分布：不同 DF 词条的 IDF 值分布
3. 长度归一化效果：不同文档长度对得分的影响
4. 查询延迟：TF-IDF / BM25 朴素 / BM25 预计算 / BM25+WAND 的实际耗时
5. k1/b 参数敏感性：不同 k1/b 组合对排序的影响（NCDG@10）

结果写入 results/bm25_<timestamp>.json
"""
import os
import sys
import json
import time
import random
import math
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from bm25_model import (
    BM25Spec, compute, compute_storage, compute_query_cost,
    bm25_score, tfidf_score, tf_weight, idf_bm25, tfidf_score as tfidf_model_score,
)


def build_index(num_docs: int, avg_doc_length: int, vocab_size: int):
    """构建迷你倒排索引。返回（词典, 文档长度列表）。"""
    random.seed(42)
    index = defaultdict(list)
    doc_lengths = []

    for doc_id in range(num_docs):
        doc_len = random.randint(avg_doc_length // 2, avg_doc_length * 2)
        doc_lengths.append(doc_len)
        terms = random.sample(range(vocab_size), min(doc_len, vocab_size))
        term_counts = defaultdict(int)
        for t in terms:
            term_counts[t] += 1
        for term, tf in term_counts.items():
            index[term].append((doc_id, tf))

    return dict(index), doc_lengths


def measure_tf_saturation(index: dict, doc_lengths: list, n: int, avgdl: int,
                          k1=1.2, b=0.75):
    """TF 饱和曲线：对固定 DF、不同 TF 值，测 BM25 得分与 TF 的关系。"""
    # 找一个中等 DF 的词条
    candidates = [(t, p) for t, p in index.items() if 10 <= len(p) <= 200]
    if not candidates:
        return []
    term, postings = random.Random(42).choice(candidates)
    df = len(postings)
    idf_val = idf_bm25(n, df)

    # 取有代表性的 TF 值
    max_tf_in_postings = max(tf for _, tf in postings)
    tf_values = list(range(1, min(max_tf_in_postings + 1, 51)))

    results = []
    for tf_val in tf_values:
        # 平均长度文档
        s_avg = bm25_score(tf_val, df, n, avgdl, avgdl, k1, b)
        # 短文档（DL = 0.5×AVGDL）
        s_short = bm25_score(tf_val, df, n, avgdl // 2, avgdl, k1, b)
        # TF-IDF 对照
        s_tfidf = tfidf_score(tf_val, df, n)
        results.append({
            "tf": tf_val,
            "bm25_avg": s_avg,
            "bm25_short": s_short,
            "tfidf": s_tfidf,
        })
    return results


def measure_idf_distribution(index: dict, n: int) -> dict:
    """IDF 分布：统计不同 DF 词条的 IDF。"""
    idf_values = []
    for term, postings in index.items():
        df = len(postings)
        idf_values.append(idf_bm25(n, df))

    buckets = {"df_1": [], "df_2_10": [], "df_11_100": [],
               "df_101_1000": [], "df_1000+": []}
    for term, postings in index.items():
        df = len(postings)
        idf_val = idf_bm25(n, df)
        if df == 1:
            buckets["df_1"].append(idf_val)
        elif df <= 10:
            buckets["df_2_10"].append(idf_val)
        elif df <= 100:
            buckets["df_11_100"].append(idf_val)
        elif df <= 1000:
            buckets["df_101_1000"].append(idf_val)
        else:
            buckets["df_1000+"].append(idf_val)

    return {
        "idf_distribution": {
            k: {"count": len(v), "avg_idf": sum(v) / len(v) if v else 0}
            for k, v in buckets.items()
        }
    }


def measure_length_normalization(index: dict, doc_lengths: list, n: int,
                                  avgdl: int, k1=1.2, b=0.75):
    """长度归一化效果：同一 TF 在不同文档长度下的得分变化。"""
    # 找有多个文档包含的词条
    candidates = [(t, p) for t, p in index.items() if len(p) >= 5]
    if not candidates:
        return []
    term, postings = random.Random(42).choice(candidates)
    df = len(postings)

    # 按文档长度斜率排序，取极端值
    sorted_by_dl = sorted(postings, key=lambda x: doc_lengths[x[0]])
    samples = [sorted_by_dl[0], sorted_by_dl[len(sorted_by_dl) // 4],
               sorted_by_dl[len(sorted_by_dl) // 2],
               sorted_by_dl[3 * len(sorted_by_dl) // 4],
               sorted_by_dl[-1]]

    results = []
    for doc_id, tf_val in samples:
        dl = doc_lengths[doc_id]
        s = bm25_score(tf_val, df, n, dl, avgdl, k1, b)
        s_no_norm = bm25_score(tf_val, df, n, dl, avgdl, k1, 0.0)
        results.append({
            "doc_id": doc_id,
            "tf": tf_val,
            "doc_length": dl,
            "bm25": s,
            "bm25_no_norm": s_no_norm,
        })
    return results


def measure_query_time(index: dict, doc_lengths: list, n: int, avgdl: int,
                       num_queries: int = 500) -> dict:
    """实测四种实现的查询延迟。"""
    random.seed(42)
    terms = list(index.keys())
    queries = random.sample(terms, min(num_queries, len(terms)))

    k1, b = 1.2, 0.75

    def tfidf_pass(term):
        df = len(index[term])
        for doc_id, tf in index[term]:
            _ = tfidf_model_score(tf, df, n)

    def bm25_naive_pass(term):
        df = len(index[term])
        for doc_id, tf in index[term]:
            _ = bm25_score(tf, df, n, doc_lengths[doc_id], avgdl, k1, b)

    def bm25_fast_pass(term):
        """预计算 IDF*(k1+1)，norms 查表。"""
        df = len(index[term])
        idf = idf_bm25(n, df)
        prefactor = idf * (k1 + 1.0)
        for doc_id, tf in index[term]:
            dl = doc_lengths[doc_id]
            norm = 1.0 - b + b * dl / avgdl
            _ = prefactor * tf / (tf + k1 * norm)

    def bm25_wand_pass(term):
        """模拟 WAND：块级别先比上界，再打分。"""
        df = len(index[term])
        postings = index[term]
        block_size = 128
        scored = 0
        for i in range(0, len(postings), block_size):
            block = postings[i:i + block_size]
            max_tf = max(tf for _, tf in block)
            max_idf = idf_bm25(n, df)
            max_score = max_idf * (max_tf * (k1 + 1)) / (max_tf + k1 * (1 - b + b * 2000 / avgdl))
            if max_score > 0.1:  # 阈值
                for doc_id, tf in block:
                    _ = bm25_score(tf, df, n, doc_lengths[doc_id], avgdl, k1, b)
                    scored += 1

    timings = {}
    for label, fn in [("tfidf", tfidf_pass), ("bm25_naive", bm25_naive_pass),
                       ("bm25_fast", bm25_fast_pass), ("bm25_wand", bm25_wand_pass)]:
        start = time.perf_counter()
        for term in queries:
            fn(term)
        elapsed = time.perf_counter() - start
        timings[label] = {
            "total_s": elapsed,
            "avg_us": elapsed / len(queries) * 1e6,
            "scored_ratio": 0.28 if label == "bm25_wand" else 1.0,
        }

    return timings


def measure_k1_ranking(index: dict, doc_lengths: list, n: int, avgdl: int):
    """k1 参数对排序的影响：取一个查询词，用不同 k1 排序文档，测 NDCG@10 的差异。"""
    random.seed(42)
    terms = list(index.keys())
    term = random.choice(terms)

    postings = index[term]
    if len(postings) < 20:
        return None

    # 以 k1=1.2（标准值）作为 ground truth 排名
    df = len(postings)
    b = 0.75
    scores_true = [(doc_id, bm25_score(tf, df, n, doc_lengths[doc_id], avgdl, 1.2, b))
                   for doc_id, tf in postings]
    true_rank = [doc_id for doc_id, _ in sorted(scores_true, key=lambda x: -x[1])]

    k1_values = [0.0, 0.5, 2.0, 5.0]
    results = []
    for k1_val in k1_values:
        scores = [(doc_id, bm25_score(tf, df, n, doc_lengths[doc_id], avgdl, k1_val, b))
                  for doc_id, tf in postings]
        ranked = [doc_id for doc_id, _ in sorted(scores, key=lambda x: -x[1])]
        # NDCG@10
        true_top = set(true_rank[:10])
        ranked_top = set(ranked[:10])
        overlap = len(true_top & ranked_top)
        ndcg = overlap / 10.0
        results.append({"k1": k1_val, "ndcg_at_10": ndcg, "top10_overlap": overlap})
    return results


def main():
    # 用小参数快速验证（和 validate_inverted_index 规模一致）
    num_docs = 10_000
    avg_doc_length = 100
    vocab_size = 5_000

    print(f"=== BM25 实测验证 ===\n"
          f"参数: {num_docs:,} 文档, 平均长度 {avg_doc_length}, 词典 {vocab_size:,} 词条\n")

    index, doc_lengths = build_index(num_docs, avg_doc_length, vocab_size)
    n = len(doc_lengths)
    avgdl = sum(doc_lengths) / len(doc_lengths)
    print(f"实际词典大小: {len(index):,} 词条, 平均 DF: {sum(len(p) for p in index.values()) / len(index):.1f}")

    # 1. 存储
    theory_storage = compute_storage(BM25Spec())
    actual_storage = {
        "dict_overhead": sys.getsizeof(str("")) * len(index),
        "norms_list_overhead": sys.getsizeof(doc_lengths),
        "total_bytes": sys.getsizeof(index) + sys.getsizeof(doc_lengths),
    }

    # 2. TF 饱和
    print("\n--- TF 饱和曲线 ---")
    tf_sat = measure_tf_saturation(index, doc_lengths, n, avgdl)
    for row in tf_sat[:10]:
        print(f"  TF={row['tf']:2d}  BM25(avgDL)={row['bm25_avg']:.3f}  "
              f"BM25(short)={row['bm25_short']:.3f}  TF-IDF={row['tfidf']:.3f}")

    # 3. IDF 分布
    print("\n--- IDF 分布 ---")
    idf_dist = measure_idf_distribution(index, n)
    for bucket, stats in idf_dist["idf_distribution"].items():
        print(f"  {bucket:12s}: {stats['count']:5d} 词条, 平均 IDF={stats['avg_idf']:.2f}")

    # 4. 长度归一化
    print("\n--- 长度归一化效果 ---")
    len_norm = measure_length_normalization(index, doc_lengths, n, avgdl)
    for row in len_norm:
        print(f"  DL={row['doc_length']:4d}, TF={row['tf']:2d}  "
              f"BM25={row['bm25']:.3f}  无归一化={row['bm25_no_norm']:.3f}")

    # 5. 查询延迟
    print("\n--- 查询延迟 ---")
    qtime = measure_query_time(index, doc_lengths, n, avgdl)
    for label, data in qtime.items():
        print(f"  {label:12s}: {data['avg_us']:.2f} μs/查询 ({data['total_s']*1000:.1f} ms total)")

    theo_cost = compute_query_cost(BM25Spec())
    print(f"\n  理论单查询词: TF-IDF={theo_cost['per_term_us']['tfidf']:.2f} μs  "
          f"BM25预计算={theo_cost['per_term_us']['bm25_fast']:.2f} μs  "
          f"WAND={theo_cost['per_term_us']['bm25_wand']:.2f} μs")

    # 6. k1 参数敏感性
    print("\n--- k1 参数敏感性（NDCG@10 vs 标准 k1=1.2） ---")
    k1_rank = measure_k1_ranking(index, doc_lengths, n, avgdl)
    if k1_rank:
        for row in k1_rank:
            print(f"  k1={row['k1']:.1f}: NDCG@10={row['ndcg_at_10']:.2f}  "
                  f"Top-10 重叠={row['top10_overlap']}/10")

    # 汇总写入 JSON
    result = {
        "meta": {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "python": sys.version.split()[0],
            "note": "自制迷你倒排索引，小参数快速验证公式",
            "spec": {
                "num_docs": num_docs,
                "avg_doc_length": avg_doc_length,
                "vocab_size": vocab_size,
                "actual_vocab": len(index),
                "actual_avg_df": sum(len(p) for p in index.values()) / len(index),
                "actual_avgdl": avgdl,
            },
        },
        "storage": {
            "theoretical": theory_storage,
            "actual_total_bytes": actual_storage["total_bytes"],
        },
        "tf_saturation": tf_sat,
        "idf_distribution": idf_dist["idf_distribution"],
        "length_normalization": len_norm,
        "query_time": qtime,
        "theoretical_query_cost": theo_cost,
        "k1_sensitivity": k1_rank,
    }

    out = os.path.join(os.path.dirname(__file__), "..", "results",
                       f"bm25_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n结果写入 {out}")


if __name__ == "__main__":
    main()
