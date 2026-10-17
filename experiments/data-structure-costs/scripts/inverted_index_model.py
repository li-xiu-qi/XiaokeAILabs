# -*- coding: utf-8 -*-
"""
倒排索引理论性能模型（核心计算器）

纯闭式解，不依赖真实搜索引擎。倒排索引的核心结构：
- 词典（Dictionary）：唯一词条 -> 倒排列表指针
- 倒排列表（Posting List）：每个词条在哪些文档中出现，出现频率

核心指标：
1. 存储占用：词典大小 + 倒排列表大小（含压缩）
2. 查询延迟：给定查询词，计算 TF-IDF 或 BM25 分数的时间
3. 召回率：返回的文档中包含相关文档的比例

标准 TF-IDF 打分：
- TF(t,d) = 词条 t 在文档 d 中的频率
- IDF(t) = log(N / DF(t))，N 是总文档数，DF(t) 是包含 t 的文档数
- Score(t,d) = TF(t,d) * IDF(t)

BM25 打分（更现代的变体）：
- IDF(t) = log((N - DF(t) + 0.5) / (DF(t) + 0.5) + 1)
- Score(t,d) = IDF(t) * (TF(t,d) * (k1 + 1)) / (TF(t,d) + k1 * (1 - b + b * DL(d) / AVGDL))
  - k1 典型 1.2，b 典型 0.75
  - DL(d) 是文档 d 的长度，AVGDL 是平均文档长度

本模块实现这些公式，参数可配。
"""
from dataclasses import dataclass
import math


@dataclass
class InvertedIndexSpec:
    """倒排索引参数"""
    num_docs: int = 1_000_000          # 总文档数 N
    avg_doc_length: int = 500          # 平均文档长度（词条数）
    vocab_size: int = 100_000          # 词典大小（唯一词条数）
    avg_postings_per_term: int = 50    # 每个词条的平均倒排条目数（DF）
    doc_id_bytes: int = 4              # docID 占字节数（uint32）
    tf_bytes: int = 2                  # TF 占字节数（uint16）
    score_dtype_bytes: int = 4         # 分数占字节数（float32）


@dataclass
class InvertedIndexMetrics:
    n: int                       # 文档数
    dict_size_bytes: int         # 词典大小（字节）
    posting_list_bytes: int      # 倒排列表大小（未压缩）
    compressed_posting_bytes: int  # 倒排列表大小（压缩后，估算）
    total_index_bytes: int       # 总索引大小
    tfidf_query_time_us: float   # TF-IDF 查询延迟（微秒）
    bm25_query_time_us: float    # BM25 查询延迟（微秒）


def compute_storage(spec: InvertedIndexSpec) -> tuple:
    """
    存储占用估算。
    词典：vocab_size * (term_bytes + postings_ptr_bytes)
    倒排列表：总条目数 * (docID + TF) 字节
    """
    term_bytes = 20  # 假设词条平均 20 字节（UTF-8 中文）
    postings_ptr_bytes = 8  # 指针占 8 字节
    dict_bytes = spec.vocab_size * (term_bytes + postings_ptr_bytes)

    # 总倒排条目数 = 词条数 * 每个词条的平均 DF
    total_postings = spec.vocab_size * spec.avg_postings_per_term
    posting_bytes = total_postings * (spec.doc_id_bytes + spec.tf_bytes)

    # 压缩估算：docID 差分编码 + varint，压缩比约 3-5x
    compression_ratio = 4.0
    compressed_posting_bytes = int(posting_bytes / compression_ratio)

    return dict_bytes, posting_bytes, compressed_posting_bytes


def tfidf_score(tf: int, df: int, n: int) -> float:
    """TF-IDF 打分"""
    if df == 0:
        return 0.0
    idf = math.log(n / df)
    return tf * idf


def bm25_score(tf: int, df: int, n: int, dl: int, avgdl: int,
               k1: float = 1.2, b: float = 0.75) -> float:
    """BM25 打分"""
    if df == 0:
        return 0.0
    idf = math.log((n - df + 0.5) / (df + 0.5) + 1)
    tf_component = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
    return idf * tf_component


def compute_query_time(spec: InvertedIndexSpec, avg_terms_per_query: int = 3) -> tuple:
    """
    查询延迟估算。
    每个查询词：查词典（O(1) 哈希）+ 遍历倒排列表（DF 个条目）+ 打分。
    TF-IDF 打分比 BM25 简单，所以更快。
    """
    # 词典查找：哈希表 O(1)，约 50 ns
    dict_lookup_ns = 50

    # 遍历倒排列表 + 打分
    # TF-IDF：每个条目 1 次乘法 + 1 次 log（但 IDF 只算一次），约 10 ns/条目
    # BM25：每个条目更多运算，约 20 ns/条目
    tfidf_per_posting_ns = 10
    bm25_per_posting_ns = 20

    # 合并结果：3 个查询词的结果合并（假设查询有 3 个词）
    merge_overhead_ns = 100

    avg_df = spec.avg_postings_per_term
    tfidf_ns = (dict_lookup_ns + avg_df * tfidf_per_posting_ns) * avg_terms_per_query + merge_overhead_ns
    bm25_ns = (dict_lookup_ns + avg_df * bm25_per_posting_ns) * avg_terms_per_query + merge_overhead_ns

    return tfidf_ns / 1000, bm25_ns / 1000  # 转换为微秒


def compute(spec: InvertedIndexSpec) -> InvertedIndexMetrics:
    """给定参数，递推全部指标。"""
    dict_bytes, posting_bytes, compressed_bytes = compute_storage(spec)
    total_bytes = dict_bytes + compressed_bytes
    tfidf_us, bm25_us = compute_query_time(spec)

    return InvertedIndexMetrics(
        n=spec.num_docs,
        dict_size_bytes=dict_bytes,
        posting_list_bytes=posting_bytes,
        compressed_posting_bytes=compressed_bytes,
        total_index_bytes=total_bytes,
        tfidf_query_time_us=tfidf_us,
        bm25_query_time_us=bm25_us,
    )


def _selftest():
    """公式自洽性检查"""
    spec = InvertedIndexSpec()

    # 存储应为正
    m = compute(spec)
    assert m.dict_size_bytes > 0
    assert m.total_index_bytes > 0

    # 压缩后应小于未压缩
    assert m.compressed_posting_bytes < m.posting_list_bytes

    # BM25 应比 TF-IDF 慢
    assert m.bm25_query_time_us > m.tfidf_query_time_us

    # 文档越多，IDF 越大
    idf_small = tfidf_score(tf=1, df=10, n=1000)
    idf_large = tfidf_score(tf=1, df=10, n=100000)
    assert idf_large > idf_small, "文档越多，IDF 应越大"

    # BM25 应惩罚长文档
    score_short = bm25_score(tf=1, df=10, n=100000, dl=100, avgdl=500)
    score_long = bm25_score(tf=1, df=10, n=100000, dl=2000, avgdl=500)
    assert score_short > score_long, "BM25 应惩罚长文档"

    print("selftest 全部通过")
    # 演示
    print(f"\n文档数={m.n:,}  词典大小={m.dict_size_bytes/1024/1024:.1f} MiB  "
          f"总索引={m.total_index_bytes/1024/1024:.1f} MiB")
    print(f"TF-IDF 查询延迟={m.tfidf_query_time_us:.2f} μs  "
          f"BM25 查询延迟={m.bm25_query_time_us:.2f} μs")


if __name__ == "__main__":
    _selftest()
