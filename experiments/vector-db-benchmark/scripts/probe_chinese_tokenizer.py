# -*- coding: utf-8 -*-
"""中文 BM25 分词实测：各库默认分词器在中文上的实际行为。

设计：一份中文小语料（几条语义相关的句子），用中文查询看能不能召回。
判据不是「有没有报错」，而是「中文查询能不能召回语义相关的文档」。
中文没有空格分隔，若分词器按空白切，整句会变成一个 token，永远匹配不上。

用法:
  ./.venv/Scripts/python.exe scripts/probe_chinese_tokenizer.py
"""
import shutil
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import re

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

# 中文语料：同一主题（数据库）的多条句子 + 一条完全无关的
DOCS = [
    "向量数据库用于存储和检索高维向量，支持近似最近邻搜索。",
    "索引构建时间是指从原始向量建立可搜索索引结构所消耗的时间。",
    "全文检索通过倒排索引实现，可以对文本内容进行关键词匹配。",
    "今天天气很好，我打算去公园散步，顺便买一杯咖啡。",
]
# 查询与 doc0 语义相关，且与 doc0 共享多个中文词
QUERY = "向量数据库的索引构建"

results = {}


def probe_lance(base_tokenizer):
    """LanceDB 指定 base_tokenizer，返回命中的文档下标。

    只传 base_tokenizer，不带 language/stem 等参数：实测带这些参数会触发
    LanceDB 内部一个 `WindowsPath % str` 的 TypeError（其自身 bug），与本测试无关。
    """
    import lancedb
    uri = str(ROOT / "data" / ("cn_lance_%s" % base_tokenizer.replace("/", "_")))
    shutil.rmtree(uri, ignore_errors=True)
    db = lancedb.connect(uri)
    data = [{"pk": i, "text": d, "vector": np.zeros(4, dtype="float32").tolist()}
            for i, d in enumerate(DOCS)]
    tbl = db.create_table("cn", data=data)
    tbl.create_fts_index("text", base_tokenizer=base_tokenizer)
    rows = tbl.search(QUERY, query_type="fts").limit(5).to_list()
    return [r["pk"] for r in rows]


def probe_milvus(analyzer_params):
    """Milvus: 指定 analyzer（tokenizer），返回命中的文档下标。

    analyzer_params 是 JSON 字符串，如 '{"tokenizer":{"type":"jieba","mode":"search"}}'。
    不带 analyzer 参数时用默认标准分词器（英文按空白切，中文整句一个 token）。
    """
    from pymilvus import MilvusClient, DataType, Function, FunctionType
    from pymilvus.milvus_client.index import IndexParams
    c = MilvusClient(uri="http://127.0.0.1:19530")
    tag = re.sub(r"[^0-9a-zA-Z]+", "_", analyzer_params)[:40] if analyzer_params else "default"
    coll = "cn_%s" % tag
    if c.has_collection(coll):
        c.drop_collection(coll)
    schema = c.create_schema(auto_id=False)
    schema.add_field("pk", DataType.INT64, is_primary=True)
    kwargs = {"max_length": 1000, "enable_analyzer": True}
    if analyzer_params:
        kwargs["analyzer_params"] = analyzer_params
    schema.add_field("text", DataType.VARCHAR, **kwargs)
    schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_function(Function(
        name="text_bm25_emb", input_field_names=["text"],
        output_field_names=["sparse"], function_type=FunctionType.BM25))
    c.create_collection(collection_name=coll, schema=schema)
    c.insert(collection_name=coll,
             data=[{"pk": i, "text": d} for i, d in enumerate(DOCS)])
    ip = IndexParams()
    ip.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX",
                 metric_type="BM25")
    c.create_index(collection_name=coll, index_params=ip)
    c.load_collection(collection_name=coll)
    time.sleep(3)
    r = c.search(collection_name=coll, data=[QUERY], anns_field="sparse",
                 limit=5, output_fields=[])
    return [(h["pk"], round(h["distance"], 2)) for h in r[0]]


def probe_surreal():
    """SurrealDB: DEFINE ANALYZER，四种 tokenizer 逐个试。"""
    from surrealdb import Surreal
    out = {}
    for tok in ["punct", "class", "blank"]:
        db = Surreal("ws://127.0.0.1:8000/rpc")
        db.signin({"username": "root", "password": "root"})
        db.use("cnprobe", "main")
        db.query("REMOVE TABLE IF EXISTS docs;")
        db.query("REMOVE ANALYZER IF EXISTS an;")
        db.query("DEFINE TABLE docs SCHEMALESS;")
        db.query("DEFINE FIELD body ON docs TYPE string;")
        db.query("DEFINE ANALYZER an TOKENIZERS %s FILTERS lowercase;" % tok)
        db.query("DEFINE INDEX idx ON docs FIELDS body "
                 "FULLTEXT ANALYZER an BM25(1.2, 0.75) HIGHLIGHTS;")
        db.query("INSERT INTO docs $rows",
                 {"rows": [{"pk": i, "body": d} for i, d in enumerate(DOCS)]})
        time.sleep(1)
        r = db.query("SELECT pk, search::score(1) AS s FROM docs "
                     "WHERE body @1@ $q ORDER BY s DESC LIMIT 5;", {"q": QUERY})
        out[tok] = [row.get("pk") for row in (r or [])]
    return out


def probe_chroma():
    """ChromaDB: ChromaBm25EmbeddingFunction。"""
    try:
        from chromadb import PersistentClient
        from chromadb.utils.embedding_functions import ChromaBm25EmbeddingFunction
        import tempfile
        d = tempfile.mkdtemp(prefix="cn_chroma_")
        cl = PersistentClient(path=d)
        ef = ChromaBm25EmbeddingFunction()
        coll = cl.create_collection("cn_bm25", embedding_function=None)
        coll.add(ids=[str(i) for i in range(len(DOCS))], documents=DOCS)
        r = coll.query(query_texts=[QUERY], n_results=5)
        return [int(x) for x in r["ids"][0]]
    except Exception as e:
        return "失败: %s: %s" % (type(e).__name__, str(e)[:120])


def main():
    print("语料 %d 条，查询: %s" % (len(DOCS), QUERY))
    print("doc0 与查询共享词: 向量/数据库/索引\n")

    print("=== LanceDB ===")
    for tok in ["simple", "icu", "icu/split", "raw", "ngram"]:
        try:
            print("  base_tokenizer=%-11s -> 命中 %s" % (tok, probe_lance(tok)))
        except Exception as e:
            print("  base_tokenizer=%-11s -> 失败 %s: %s" % (tok, type(e).__name__, str(e)[:110]))

    print("\n=== Milvus（analyzer tokenizer）===")
    for tag, params in [
        ("默认（无 analyzer）", None),
        ("jieba search 模式", '{"tokenizer":{"type":"jieba","mode":"search"}}'),
        ("jieba exact 模式", '{"tokenizer":{"type":"jieba","mode":"exact"}}'),
        ("icu", '{"tokenizer":{"type":"icu"}}'),
    ]:
        try:
            print("  %-20s -> %s" % (tag, probe_milvus(params)))
        except Exception as e:
            print("  %-20s -> 失败 %s: %s" % (tag, type(e).__name__, str(e)[:110]))

    print("\n=== SurrealDB（自定义 analyzer）===")
    try:
        for tok, hits in probe_surreal().items():
            print("  tokenizer=%-8s -> 命中 %s" % (tok, hits))
    except Exception as e:
        print("  失败: %s: %s" % (type(e).__name__, str(e)[:150]))

    print("\n=== ChromaDB ===")
    print("  ->", probe_chroma())


if __name__ == "__main__":
    main()
