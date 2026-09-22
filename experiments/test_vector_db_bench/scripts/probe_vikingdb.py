# -*- coding: utf-8 -*-
"""VikingVectorIndex（OpenViking 内置向量库）混合检索探针。

背景：VikingVectorIndex 只支持 FLAT / FLAT_HYBRID 两种索引
（openviking/storage/vectordb/utils/validation.py:139），没有 HNSW/IVF 等 ANN 索引，
本质是暴力检索 + 可选稀疏向量加权。稀疏分量由
`SearchWithSparseLogitAlpha` 控制，打分公式见
src/index/detail/vector/common/bruteforce.h:433：
    dense_score * (1 - alpha) + sparse_score * alpha

关键限制（与 Milvus/LanceDB 的性质差异）：
1. 本地后端的 `search_by_keywords` 抛 NotImplementedError
   （local_collection.py:700），query 侧不做分词，BM25 稀疏向量要调用方自己算好传进去。
2. 不支持「纯 BM25」检索，alpha=1.0 才是纯稀疏，alpha=0 是纯 dense。
3. 没有独立部署形态，是内嵌模块（需编译 C++ abi3 扩展）。

本脚本测三件事：
  1. flat 与 flat_hybrid 能否建索引、能否查询
  2. alpha 扫描（0 / 0.3 / 0.5 / 0.7 / 1.0）对混合检索命中率的影响
  3. 中文场景下，自算 BM25 稀疏向量能否让它工作

用法（用隔离的 .venv-viking）：
  ./.venv-viking/Scripts/python.exe scripts/probe_vikingdb.py
"""
import shutil
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from index_bench import load_real  # noqa: E402

DOCS = [
    "向量数据库用于存储和检索高维向量，支持近似最近邻搜索。",
    "索引构建时间是指从原始向量建立可搜索索引结构所消耗的时间。",
    "今天天气很好，我打算去公园散步，顺便买一杯咖啡。",
    "全文检索通过倒排索引实现，可以对文本内容进行关键词匹配。",
]
QUERY = "向量数据库的索引构建"


def make_meta(name, dim):
    """collection meta：CollectionName 必填，向量索引配置不在这里。"""
    return {
        "CollectionName": name,
        "Fields": [
            {"FieldName": "pk", "FieldType": "int64", "IsPrimaryKey": True},
            {"FieldName": "text", "FieldType": "string"},
            {"FieldName": "vector", "FieldType": "vector", "Dim": dim},
        ],
    }


def make_index_meta(index_type="flat_hybrid", distance="cosine",
                   sparse_alpha=None, enable_sparse=True):
    """create_index 的索引参数。

    参照 examples/cuvs_smoke.py:39 的官方格式：IndexName + VectorIndex + ScalarIndex。
    flat_hybrid 配 SearchWithSparseLogitAlpha 控制稀疏分量权重。
    """
    vi = {"IndexType": index_type, "Distance": distance}
    if enable_sparse:
        vi["EnableSparse"] = True
    if sparse_alpha is not None:
        vi["SearchWithSparseLogitAlpha"] = sparse_alpha
        vi["IndexWithSparseLogitAlpha"] = sparse_alpha
    return {"IndexName": "default", "VectorIndex": vi, "ScalarIndex": []}


def simple_bm25_sparse(text, vocab=None):
    """自制 BM25 风格稀疏向量：词 -> 权重。

    这里只做词项计数当权重，目的是验证链路能否走通（库里没有分词器，
    调用方必须自己算）。真实 BM25 权重应该带 idf 和长度归一化。
    """
    import re
    toks = [w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 1]
    counts = {}
    for t in toks:
        counts[t] = counts.get(t, 0) + 1
    return counts


def probe_basic():
    """第一关：能不能装上、能不能建索引、能不能查。"""
    print("=== 1. 基础连通性 ===")
    from openviking.storage.vectordb.collection import get_or_create_local_collection
    rng = np.random.default_rng(0)
    dim = 8
    path = str(ROOT / "data" / "viking_probe")
    shutil.rmtree(path, ignore_errors=True)
    coll = get_or_create_local_collection(meta_data=make_meta("probe", dim), path=path)
    print("  collection 创建成功")
    coll.upsert_data([
        {"pk": i, "text": d, "vector": rng.normal(size=dim).astype("float32").tolist()}
        for i, d in enumerate(DOCS)
    ])
    print("  upsert 成功")
    coll.create_index("default", make_index_meta("flat_hybrid"))
    print("  create_index(flat_hybrid) 成功")
    qv = rng.normal(size=dim).astype("float32").tolist()
    r = coll.search_by_vector("default", dense_vector=qv, limit=3)
    # SearchResult.data: List[SearchItemResult]，字段是 id/fields/score
    print("  search_by_vector 返回 %d 条: %s" % (
        len(r.data), [(d.id, round(d.score, 3)) for d in r.data]))
    return coll


def probe_chinese():
    """第二关：中文 + 自算 sparse 向量，看 alpha 对混合检索的影响。"""
    print("\n=== 2. 中文混合检索（sparse 自算，alpha 扫描）===")
    from openviking.storage.vectordb.collection import get_or_create_local_collection
    dim = 384
    vecs, queries, _, _ = load_real(20, 0, return_idx=True)
    z = np.load(ROOT / "data" / "20news_texts.npz", allow_pickle=True)
    texts = [str(t) for t in z["train_texts"][:len(vecs)]]
    path = str(ROOT / "data" / "viking_cn")
    shutil.rmtree(path, ignore_errors=True)
    coll = get_or_create_local_collection(meta_data=make_meta("cn", dim), path=path)
    coll.upsert_data([
        {"pk": i, "text": t, "vector": vecs[i].tolist()} for i, t in enumerate(texts)
    ])
    qv = queries[0].tolist()
    qsparse = simple_bm25_sparse(texts[0])
    print("  库 %d 条，query sparse 词项数 %d" % (len(texts), len(qsparse)))
    for alpha in [None, 0.0, 0.3, 0.5, 0.7, 1.0]:
        try:
            coll.create_index("default", make_index_meta("flat_hybrid", sparse_alpha=alpha))
            r = coll.search_by_vector("default", dense_vector=qv,
                                      sparse_vector=qsparse, limit=5)
            rows = [(d.id, round(d.score, 3)) for d in r.data]
            print("  alpha=%-5s -> %s" % (alpha, rows))
        except Exception as e:
            print("  alpha=%-5s -> 失败 %s: %s" % (alpha, type(e).__name__, str(e)[:120]))


def main():
    try:
        probe_basic()
    except Exception as e:
        print("  基础连通失败: %s: %s" % (type(e).__name__, str(e)[:300]))
        import traceback
        traceback.print_exc()
        return
    try:
        probe_chinese()
    except Exception as e:
        print("  中文测试失败: %s: %s" % (type(e).__name__, str(e)[:300]))


if __name__ == "__main__":
    main()
