# -*- coding: utf-8 -*-
"""
倒排索引实测验证

用纯 Python 自制迷你倒排索引，实测三个核心指标：
1. 存储占用：构建索引后的实际内存/磁盘占用
2. 查询延迟：TF-IDF 和 BM25 查询的实际耗时
3. IDF 分布：不同 DF 的词条的 IDF 值分布

方法：生成模拟文档（随机词条），构建倒排索引，实测指标。
结果写入 results/inverted_index_<timestamp>.json
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
from inverted_index_model import (
    InvertedIndexSpec, compute_storage, tfidf_score, bm25_score, compute_query_time
)


def build_index(num_docs: int, avg_doc_length: int, vocab_size: int) -> dict:
    """
    构建迷你倒排索引。
    返回：词典（term -> [(doc_id, tf), ...]）和文档长度。
    """
    random.seed(42)
    index = defaultdict(list)  # term -> [(doc_id, tf), ...]
    doc_lengths = []

    for doc_id in range(num_docs):
        doc_len = random.randint(avg_doc_length // 2, avg_doc_length * 2)
        doc_lengths.append(doc_len)
        # 随机选择词条
        terms = random.sample(range(vocab_size), min(doc_len, vocab_size))
        term_counts = defaultdict(int)
        for t in terms:
            term_counts[t] += 1
        for term, tf in term_counts.items():
            index[term].append((doc_id, tf))

    return dict(index), doc_lengths


def measure_query_time(index: dict, doc_lengths: list, num_queries: int = 1000) -> dict:
    """实测查询延迟。随机选择查询词，计算 TF-IDF 和 BM25 分数。"""
    random.seed(42)
    terms = list(index.keys())
    queries = random.sample(terms, min(num_queries, len(terms)))

    n = len(doc_lengths)
    avgdl = sum(doc_lengths) / len(doc_lengths)

    # TF-IDF 查询
    start = time.perf_counter()
    for term in queries:
        df = len(index[term])
        for doc_id, tf in index[term]:
            _ = tfidf_score(tf, df, n)
    tfidf_elapsed = time.perf_counter() - start

    # BM25 查询
    start = time.perf_counter()
    for term in queries:
        df = len(index[term])
        for doc_id, tf in index[term]:
            dl = doc_lengths[doc_id]
            _ = bm25_score(tf, df, n, dl, avgdl)
    bm25_elapsed = time.perf_counter() - start

    return {
        "num_queries": len(queries),
        "tfidf_total_time_s": tfidf_elapsed,
        "bm25_total_time_s": bm25_elapsed,
        "tfidf_avg_time_us": tfidf_elapsed / len(queries) * 1e6,
        "bm25_avg_time_us": bm25_elapsed / len(queries) * 1e6,
    }


def measure_storage(index: dict) -> dict:
    """实测存储占用。用 sys.getsizeof 估算字典和列表的字节数。"""
    # 词典大小：每个词条一个 key（字符串）+ 一个 value（列表）
    dict_bytes = 0
    posting_bytes = 0
    for term, postings in index.items():
        dict_bytes += sys.getsizeof(str(term))  # 词条
        dict_bytes += sys.getsizeof(postings)   # 列表对象本身
        posting_bytes += len(postings) * 8      # 每个 (doc_id, tf) 元组约 8 字节（简化）

    return {
        "dict_bytes": dict_bytes,
        "posting_bytes": posting_bytes,
        "total_bytes": dict_bytes + posting_bytes,
    }


def measure_idf_distribution(index: dict, n: int) -> dict:
    """实测 IDF 分布。统计不同 DF 的词条的 IDF 值。"""
    idf_values = []
    for term, postings in index.items():
        df = len(postings)
        idf = math.log(n / df) if df > 0 else 0
        idf_values.append((df, idf))

    # 按 DF 分桶统计
    buckets = {
        "df_1": [],
        "df_2_10": [],
        "df_11_100": [],
        "df_101_1000": [],
        "df_1000+": [],
    }
    for df, idf in idf_values:
        if df == 1:
            buckets["df_1"].append(idf)
        elif df <= 10:
            buckets["df_2_10"].append(idf)
        elif df <= 100:
            buckets["df_11_100"].append(idf)
        elif df <= 1000:
            buckets["df_101_1000"].append(idf)
        else:
            buckets["df_1000+"].append(idf)

    return {
        "idf_distribution": {
            k: {
                "count": len(v),
                "avg_idf": sum(v) / len(v) if v else 0,
            }
            for k, v in buckets.items()
        }
    }


def main():
    print("=== 倒排索引实测验证 ===\n")

    # 用小参数快速验证
    spec = InvertedIndexSpec(
        num_docs=10_000,
        avg_doc_length=100,
        vocab_size=5_000,
        avg_postings_per_term=200,  # 会因随机性而不同，这里给个参考
    )

    print(f"参数: {spec.num_docs:,} 文档, 平均长度 {spec.avg_doc_length}, "
          f"词典 {spec.vocab_size:,} 词条\n")

    # 构建索引
    print("--- 构建索引 ---")
    index, doc_lengths = build_index(spec.num_docs, spec.avg_doc_length, spec.vocab_size)
    print(f"实际词典大小: {len(index):,} 词条")
    print(f"实际平均 DF: {sum(len(p) for p in index.values()) / len(index):.1f}")

    # 实测存储
    print("\n--- 存储占用 ---")
    storage = measure_storage(index)
    theo_dict, theo_posting, theo_compressed = compute_storage(spec)
    print(f"实测: 词典 {storage['dict_bytes']/1024:.1f} KB + "
          f"倒排列表 {storage['posting_bytes']/1024:.1f} KB = "
          f"总计 {storage['total_bytes']/1024:.1f} KB")
    print(f"理论: 词典 {theo_dict/1024:.1f} KB + "
          f"倒排列表 {theo_compressed/1024:.1f} KB (压缩) = "
          f"总计 {(theo_dict + theo_compressed)/1024:.1f} KB")

    # 实测查询延迟
    print("\n--- 查询延迟 ---")
    query_time = measure_query_time(index, doc_lengths, num_queries=1000)
    print(f"TF-IDF: {query_time['tfidf_avg_time_us']:.2f} μs/查询 "
          f"({query_time['num_queries']} 次查询, 总计 {query_time['tfidf_total_time_s']*1000:.1f} ms)")
    print(f"BM25:   {query_time['bm25_avg_time_us']:.2f} μs/查询 "
          f"({query_time['num_queries']} 次查询, 总计 {query_time['bm25_total_time_s']*1000:.1f} ms)")

    theo_tfidf, theo_bm25 = compute_query_time(spec)
    print(f"理论:   TF-IDF {theo_tfidf:.2f} μs, BM25 {theo_bm25:.2f} μs")

    # 实测 IDF 分布
    print("\n--- IDF 分布 ---")
    idf_dist = measure_idf_distribution(index, spec.num_docs)
    for bucket, stats in idf_dist["idf_distribution"].items():
        print(f"{bucket:12s}: {stats['count']:5d} 词条, 平均 IDF = {stats['avg_idf']:.2f}")

    # 汇总写入 JSON
    result = {
        "meta": {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "python": sys.version.split()[0],
            "note": "自制迷你倒排索引，小参数快速验证公式",
            "spec": {
                "num_docs": spec.num_docs,
                "avg_doc_length": spec.avg_doc_length,
                "vocab_size": spec.vocab_size,
            },
            "actual": {
                "vocab_size": len(index),
                "avg_df": sum(len(p) for p in index.values()) / len(index),
            },
        },
        "storage": {
            "actual": storage,
            "theoretical": {
                "dict_bytes": theo_dict,
                "compressed_posting_bytes": theo_compressed,
                "total_bytes": theo_dict + theo_compressed,
            },
        },
        "query_time": {
            "actual": query_time,
            "theoretical": {
                "tfidf_us": theo_tfidf,
                "bm25_us": theo_bm25,
            },
        },
        "idf_distribution": idf_dist["idf_distribution"],
    }

    out = os.path.join(os.path.dirname(__file__), "..", "results",
                       f"inverted_index_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n结果写入 {out}")


if __name__ == "__main__":
    main()
