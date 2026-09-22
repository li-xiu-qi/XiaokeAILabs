# -*- coding: utf-8 -*-
"""
BM25 理论性能模型（核心计算器）

纯闭式解，不依赖真实搜索引擎。BM25 是 Okapi BM25 的简称（Robertson & Zaragoza, 2009），
倒排索引检索的事实标准打分函数，Elasticsearch、Lucene、Whoosh、SQLite FTS5 的默认排序。

本模块把 BM25 拆成四个可独立递推的维度：

1. 打分形状：IDF 曲线、TF 饱和曲线、文档长度归一化曲线。回答「同一篇文档换 k1/b 排名怎么变」。
2. 存储：倒排列表 + norms 文档长度数组。回答「比 TF-IDF 多存什么」。
3. 查询延迟：每条目的打分代价（朴素实现 vs 预计算实现），以及 WAND / Block-Max WAND 剪枝后的实际代价。
4. 规模外推：给定 N、词典、平均 DF，递推延迟与存储。

打分公式（Okapi BM25）：

    IDF(t)      = log((N - DF(t) + 0.5) / (DF(t) + 0.5) + 1)
    Score(t, d) = IDF(t) * [ TF(t,d) * (k1 + 1) ] / [ TF(t,d) + k1 * (1 - b + b * DL(d) / AVGDL) ]

k1 控制 TF 饱和速度（0 = 二值匹配，越大越接近原始 TF），b 控制长度归一化强度（0 = 不归一，1 = 完全归一）。

延迟模型的关键在于每条目的常数：朴素实现每条目含一次除法；把 IDF*(k1+1) 提到词条级别、
把 k1*(1-b+b*DL/AVGDL) 预计算进 norms 数组后，每条目退化成几次乘加，这是 Lucene 的实际做法。

本模块实现这些公式，参数可配。
"""
from dataclasses import dataclass
import math


# ── 参数与指标 ──────────────────────────────────────────

@dataclass
class BM25Spec:
    """BM25 与倒排索引的参数"""
    n_docs: int = 1_000_000          # 总文档数 N
    avg_doc_length: int = 500        # 平均文档长度（词条数）
    vocab_size: int = 100_000        # 词典大小（唯一词条数）
    avg_df: int = 50                 # 每个词条的平均倒排条目数（DF）
    k1: float = 1.2                  # TF 饱和参数
    b: float = 0.75                  # 长度归一化参数
    delta: float = 1.0               # BM25+ 的下界修正参数（0 = 退回普通 BM25）
    top_k: int = 10                  # 查询返回的文档数
    avg_terms_per_query: int = 3     # 平均查询词数
    # 存储
    doc_id_bytes: int = 4            # docID（uint32）
    tf_bytes: int = 2                # TF（uint16）
    term_bytes: int = 20             # 词条平均字节（UTF-8 中文）
    postings_ptr_bytes: int = 8      # 倒排列表指针
    norm_bytes_per_doc: int = 1      # 文档长度 norms 数组（Lucene 默认 1 字节 SmallFloat）
    compression_ratio: float = 4.0   # docID 差分 + varint 的压缩比
    # 延迟常数（纳秒，需在本机标定）
    dict_lookup_ns: float = 50.0     # 词典哈希查找
    tfidf_per_posting_ns: float = 10.0       # TF-IDF 每条目
    bm25_naive_per_posting_ns: float = 20.0  # BM25 朴素实现每条目（含一次除法）
    bm25_fast_per_posting_ns: float = 12.0   # BM25 预计算实现每条目（乘加 + norms 查表）
    norm_lookup_ns: float = 2.0      # norms 数组查表
    heap_push_ns: float = 60.0       # top-k 堆插入/替换
    # WAND / Block-Max WAND
    wand_block_size: int = 128       # 块大小（Lucene 默认 128）
    wand_block_max_ns: float = 8.0   # 读取并比较一个块最大值的代价
    wand_scored_ratio: float = 0.28  # 剪枝后实际打分的条目占比（实测标定，见 verify 脚本）


@dataclass
class BM25Metrics:
    n: int                       # 文档数
    # 打分形状
    tf_saturation_max: float     # TF 饱和上界（= k1+1）
    idf_at_df1: float            # DF=1 的 IDF
    idf_at_avg_df: float         # DF=平均 DF 的 IDF
    idf_zero_crossing_df: float  # IDF 由正转负的 DF
    length_penalty_ratio: float  # 长文档（DL=4×AVGDL）在高 TF 下的得分比（<1 表示被惩罚）
    # 存储（字节）
    postings_bytes: int          # 倒排列表（未压缩）
    compressed_postings_bytes: int
    norms_bytes: int             # 文档长度数组
    total_index_bytes: int       # 总索引（压缩后）
    # 延迟（微秒，单查询词）
    tfidf_us: float
    bm25_naive_us: float
    bm25_fast_us: float
    bm25_wand_us: float
    # 延迟（微秒，整条查询）
    full_query_tfidf_us: float
    full_query_bm25_fast_us: float
    full_query_bm25_wand_us: float


# ── 打分函数族 ──────────────────────────────────────────

def idf_classic(n: int, df: int) -> float:
    """经典 TF-IDF 的 IDF：log(N / DF)"""
    if df <= 0:
        return 0.0
    return math.log(n / df)


def idf_bm25(n: int, df: int) -> float:
    """BM25 的概率式 IDF：log((N - DF + 0.5) / (DF + 0.5) + 1)

    与经典 IDF 的差别：DF 接近 N 时不会冲到 0 以下（恒为正），
    低频端更平滑。这是 BM25 对常见词不过度惩罚的原因。
    """
    if df <= 0:
        return 0.0
    return math.log((n - df + 0.5) / (df + 0.5) + 1)


def idf_bm25_smoothed(n: int, df: int) -> float:
    """Lucene 的实际写法，与 idf_bm25 数学等价：log(1 + (N - DF + 0.5) / (DF + 0.5))"""
    if df <= 0:
        return 0.0
    return math.log(1.0 + (n - df + 0.5) / (df + 0.5))


def tf_weight(tf: int, k1: float) -> float:
    """BM25 的 TF 饱和权重，不含 IDF 与长度项。

    w(tf) = tf * (k1 + 1) / (tf + k1)
    上界 k1+1（tf→∞），起点斜率 k1+1（tf→0）。
    """
    if tf <= 0:
        return 0.0
    return tf * (k1 + 1.0) / (tf + k1)


def length_norm(dl: int, avgdl: int, b: float) -> float:
    """BM25 的文档长度归一化因子：1 - b + b * DL / AVGDL

    b=0 时恒为 1（不归一），b=1 时等于 DL/AVGDL。
    """
    if avgdl <= 0:
        return 1.0
    return 1.0 - b + b * dl / avgdl


def bm25_score(tf: int, df: int, n: int, dl: int, avgdl: int,
               k1: float = 1.2, b: float = 0.75, delta: float = 0.0) -> float:
    """Okapi BM25 打分。delta > 0 时退化为 BM25+（补一个下界修正项 δ）"""
    if tf <= 0 or df <= 0:
        return 0.0
    idf = idf_bm25(n, df)
    w = tf_weight(tf, k1) / length_norm(dl, avgdl, b)
    if delta > 0:
        w += delta
    return idf * w


def tfidf_score(tf: int, df: int, n: int) -> float:
    """作为对照的经典 TF-IDF 打分"""
    if tf <= 0 or df <= 0:
        return 0.0
    return tf * idf_classic(n, df)


# ── 存储 ────────────────────────────────────────────────

def compute_storage(spec: BM25Spec) -> dict:
    """倒排列表 + norms 数组的存储字节。

    BM25 相比 TF-IDF 唯一多出的东西是 norms 文档长度数组（每文档 1 字节），
    因为打分要 DL(d)。TF 本身本来就存在倒排列表里。
    """
    dict_bytes = spec.vocab_size * (spec.term_bytes + spec.postings_ptr_bytes)
    total_postings = spec.vocab_size * spec.avg_df
    posting_bytes = total_postings * (spec.doc_id_bytes + spec.tf_bytes)
    compressed_bytes = int(posting_bytes / spec.compression_ratio)
    norms_bytes = spec.n_docs * spec.norm_bytes_per_doc

    return {
        "dict_bytes": dict_bytes,
        "postings_bytes": posting_bytes,
        "compressed_postings_bytes": compressed_bytes,
        "norms_bytes": norms_bytes,
        "total_index_bytes": dict_bytes + compressed_bytes + norms_bytes,
    }


# ── 查询延迟 ────────────────────────────────────────────

def compute_query_cost(spec: BM25Spec) -> dict:
    """单查询词的打分延迟（微秒）。

    四种实现的每条目代价不同：
    - TF-IDF：乘法，IDF 词条级预算，最便宜
    - BM25 朴素：每条目一次除法，最贵
    - BM25 预计算：IDF*(k1+1) 提到词条级，长度归一化预计算进 norms，
      每条目只剩乘加 + 一次 norms 查表，接近 TF-IDF
    - BM25 + WAND：块最大值比较替代逐条目打分，只对剪枝剩下的条目打分

    堆插入代价按查询词数平摊到每个词上，保证四种实现可比。
    """
    df = spec.avg_df
    dict_ns = spec.dict_lookup_ns
    heap_amortized_ns = spec.top_k * spec.heap_push_ns / spec.avg_terms_per_query

    tfidf = (dict_ns + df * spec.tfidf_per_posting_ns + heap_amortized_ns) / 1000.0
    bm25_naive = (dict_ns + df * spec.bm25_naive_per_posting_ns + heap_amortized_ns) / 1000.0
    bm25_fast = (dict_ns + df * (spec.bm25_fast_per_posting_ns + spec.norm_lookup_ns)
                 + heap_amortized_ns) / 1000.0

    # WAND / Block-Max WAND：块最大值比较 + 剪枝后打分
    num_blocks = max(1, math.ceil(df / spec.wand_block_size))
    wand = (dict_ns + num_blocks * spec.wand_block_max_ns
            + df * spec.wand_scored_ratio * (spec.bm25_fast_per_posting_ns + spec.norm_lookup_ns)
            + heap_amortized_ns) / 1000.0

    # 整条查询：多词合并 + 一次 top-k 堆
    t = spec.avg_terms_per_query
    heap_ns = spec.top_k * spec.heap_push_ns
    full = {
        "tfidf": (tfidf * 1000 * t + heap_ns) / 1000.0,
        "bm25_fast": (bm25_fast * 1000 * t + heap_ns) / 1000.0,
        "bm25_wand": (wand * 1000 * t + heap_ns) / 1000.0,
    }

    return {
        "per_term_us": {
            "tfidf": tfidf,
            "bm25_naive": bm25_naive,
            "bm25_fast": bm25_fast,
            "bm25_wand": wand,
        },
        "full_query_us": full,
        "prune_speedup": bm25_fast / wand,
    }


# ── 打分形状 ────────────────────────────────────────────

def compute_score_shape(spec: BM25Spec) -> dict:
    """打分曲线的几个关键特征量。"""
    # TF 饱和上界
    sat_max = spec.k1 + 1.0

    # IDF 曲线特征
    idf_df1 = idf_bm25(spec.n_docs, 1)
    idf_avgdf = idf_bm25(spec.n_docs, spec.avg_df)
    # IDF 由正转负的 DF：(N - DF + 0.5)/(DF + 0.5) = 0 ⇒ DF ≈ N + 0.5
    # 即 DF 超过 N/2 的词条 IDF 才转负，真实语料几乎碰不到
    zero_df = spec.n_docs + 0.5

    # 长度惩罚比：DL = 4×AVGDL 的文档在高 TF 下的得分 / 平均长度文档的得分
    # 高 TF 下 TF 项饱和到 k1+1，比值只剩长度归一化因子的比
    r = 4.0
    penalty = (1.0 - spec.b + spec.b * r) ** -1

    return {
        "tf_saturation_max": sat_max,
        "idf_at_df1": idf_df1,
        "idf_at_avg_df": idf_avgdf,
        "idf_zero_crossing_df": zero_df,
        "length_penalty_ratio": penalty,
    }


# ── 总入口 ──────────────────────────────────────────────

def compute(spec: BM25Spec) -> BM25Metrics:
    """给定参数，递推全部指标。"""
    storage = compute_storage(spec)
    cost = compute_query_cost(spec)
    shape = compute_score_shape(spec)

    return BM25Metrics(
        n=spec.n_docs,
        tf_saturation_max=shape["tf_saturation_max"],
        idf_at_df1=shape["idf_at_df1"],
        idf_at_avg_df=shape["idf_at_avg_df"],
        idf_zero_crossing_df=shape["idf_zero_crossing_df"],
        length_penalty_ratio=shape["length_penalty_ratio"],
        postings_bytes=storage["postings_bytes"],
        compressed_postings_bytes=storage["compressed_postings_bytes"],
        norms_bytes=storage["norms_bytes"],
        total_index_bytes=storage["total_index_bytes"],
        tfidf_us=cost["per_term_us"]["tfidf"],
        bm25_naive_us=cost["per_term_us"]["bm25_naive"],
        bm25_fast_us=cost["per_term_us"]["bm25_fast"],
        bm25_wand_us=cost["per_term_us"]["bm25_wand"],
        full_query_tfidf_us=cost["full_query_us"]["tfidf"],
        full_query_bm25_fast_us=cost["full_query_us"]["bm25_fast"],
        full_query_bm25_wand_us=cost["full_query_us"]["bm25_wand"],
    )


def _selftest():
    """公式自洽性检查"""
    spec = BM25Spec()
    m = compute(spec)

    # 存储应为正，压缩后应更小
    assert m.postings_bytes > 0
    assert m.compressed_postings_bytes < m.postings_bytes
    assert m.norms_bytes == spec.n_docs  # 1 字节/文档
    assert m.total_index_bytes > 0

    # TF 饱和上界 = k1+1
    assert abs(m.tf_saturation_max - (spec.k1 + 1)) < 1e-9

    # BM25 应比 TF-IDF 慢（朴素实现），预计算实现应接近 TF-IDF
    assert m.bm25_naive_us > m.tfidf_us
    assert m.bm25_fast_us < m.bm25_naive_us

    # WAND 剪枝后应比全量打分快
    assert m.bm25_wand_us < m.bm25_fast_us

    # 两种 Lucene 写法应等价
    for n in (1000, 100_000, 1_000_000):
        for df in (1, 10, 100, 10_000):
            assert abs(idf_bm25(n, df) - idf_bm25_smoothed(n, df)) < 1e-12

    # BM25 应惩罚长文档
    score_short = bm25_score(tf=1, df=10, n=100000, dl=100, avgdl=500)
    score_long = bm25_score(tf=1, df=10, n=100000, dl=2000, avgdl=500)
    assert score_short > score_long, "BM25 应惩罚长文档"

    # k1 越大，TF 饱和越慢（高 TF 下得分越高）
    s_low = bm25_score(tf=50, df=10, n=100000, dl=500, avgdl=500, k1=0.5)
    s_high = bm25_score(tf=50, df=10, n=100000, dl=500, avgdl=500, k1=5.0)
    assert s_high > s_low, "k1 越大，高 TF 文档得分应越高"

    # b=0 时长度不影响得分
    s_a = bm25_score(tf=3, df=10, n=100000, dl=100, avgdl=500, b=0.0)
    s_b = bm25_score(tf=3, df=10, n=100000, dl=900, avgdl=500, b=0.0)
    assert abs(s_a - s_b) < 1e-12, "b=0 时不应有长度归一化"

    # BM25 的 IDF 在 DF ≤ N 时恒为非负，经典 IDF 在 DF = N 时已归零
    assert idf_bm25(100000, 100000) > 0
    assert abs(idf_classic(100000, 100000)) < 1e-12

    print("selftest 全部通过")
    # 演示
    print(f"\nN={m.n:,}  k1={spec.k1}  b={spec.b}  平均 DF={spec.avg_df}")
    print(f"TF 饱和上界={m.tf_saturation_max:.2f}  "
          f"IDF(DF=1)={m.idf_at_df1:.2f}  IDF(平均DF)={m.idf_at_avg_df:.3f}  "
          f"长文档惩罚比={m.length_penalty_ratio:.3f}")
    print(f"存储: 倒排列表(压缩)={m.compressed_postings_bytes/1024/1024:.1f} MiB  "
          f"norms={m.norms_bytes/1024/1024:.1f} MiB  总计={m.total_index_bytes/1024/1024:.1f} MiB")
    print(f"单查询词延迟: TF-IDF={m.tfidf_us:.2f} μs  BM25朴素={m.bm25_naive_us:.2f} μs  "
          f"BM25预计算={m.bm25_fast_us:.2f} μs  BM25+WAND={m.bm25_wand_us:.2f} μs")
    print(f"整条查询({spec.avg_terms_per_query} 词): TF-IDF={m.full_query_tfidf_us:.2f} μs  "
          f"BM25={m.full_query_bm25_fast_us:.2f} μs  BM25+WAND={m.full_query_bm25_wand_us:.2f} μs")


if __name__ == "__main__":
    _selftest()
