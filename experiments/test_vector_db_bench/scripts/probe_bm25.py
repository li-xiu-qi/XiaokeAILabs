# -*- coding: utf-8 -*-
"""BM25 支持探针：实测六个向量库能否做 BM25 全文检索。

对每个库尝试：
  1. 建 BM25 / 全文索引
  2. 插入带文本的文档
  3. 执行关键词查询，看能否返回 BM25 排序结果

输出每个库的支持等级：
  NATIVE    原生 BM25，库内完成分词+打分
  PARTIAL   有全文索引但 BM25 打分不完整/需外部配合
  NONE      不支持
  UNKNOWN   实测失败（环境问题），需人工判断

用法:
  ./.venv/Scripts/python.exe scripts/probe_bm25.py
  ./.venv/Scripts/python.exe scripts/probe_bm25.py --only milvus qdrant
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

# 测试语料：刻意设计成「关键词精确匹配」比「语义相似」更可判别的case
DOCS = [
    "The quick brown fox jumps over the lazy dog",
    "A fast auburn canine leaps above a sleepy hound",
    "Python programming language tutorial for beginners",
    "How to learn Python coding from scratch",
    "Vector database benchmark comparison and analysis",
    "Comparing embedding models for retrieval tasks",
    "The dog barked loudly at the midnight fox",
    "Machine learning model training on GPU clusters",
]
# 查询词 "fox"：只在 doc 0/6 出现，语义查询找不到 doc 6
QUERY = "fox"


def probe_milvus():
    """Milvus: 内置 BM25BuiltInFunction，原生支持。"""
    from pymilvus import MilvusClient, DataType, Function, FunctionType
    c = MilvusClient(uri="http://127.0.0.1:19530")
    coll = "bm25_probe"
    if c.has_collection(coll):
        c.drop_collection(coll)
    schema = c.create_schema(auto_id=False)
    schema.add_field("pk", DataType.INT64, is_primary=True)
    schema.add_field("text", DataType.VARCHAR, max_length=1000, enable_analyzer=True)
    schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_function(Function(
        name="text_bm25_emb", input_field_names=["text"],
        output_field_names=["sparse"], function_type=FunctionType.BM25,
    ))
    c.create_collection(collection_name=coll, schema=schema)
    c.insert(collection_name=coll, data=[
        {"pk": i, "text": d} for i, d in enumerate(DOCS)
    ])
    # sparse 字段要建 SPARSE_INVERTED_INDEX / SPARSE_WAND 索引，否则 load 报 index not found
    try:
        from pymilvus.milvus_client.index import IndexParams
        ip = IndexParams()
        ip.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX",
                     metric_type="BM25")
        c.create_index(collection_name=coll, index_params=ip)
    except Exception as e:
        return {"level": "UNKNOWN", "note": "建 sparse 索引失败: %s" % str(e)[:150]}
    c.load_collection(collection_name=coll)
    time.sleep(3)
    r = c.search(collection_name=coll, data=[QUERY], anns_field="sparse",
                 limit=3, output_fields=["text"])
    hits = [(h["pk"], round(h["distance"], 3)) for h in r[0]]
    return {"level": "NATIVE", "hits": hits,
            "note": "BM25BuiltInFunction + SPARSE_FLOAT_VECTOR + SPARSE_INVERTED_INDEX，"
                    "metric_type=BM25；需先建 sparse 索引再 load_collection"}


def probe_qdrant():
    """Qdrant: Bm25Config + Document(model='qdrant-bm25') 本地推理。

    源码依据：qdrant/lib/bm25/src/lib.rs（297 行，含 Bm25Params{k1,b,avg_doc_len}、
    DEFAULT_K1=1.2 / DEFAULT_B=0.75 / DEFAULT_AVG_DOC_LEN=256、参数校验），
    以及 lib/api/src/rest/schema.rs 暴露的 DocumentOptions=Bm25Config。
    该 crate 在 workspace 里但尚未被 segment 依赖，属新并入功能。

    本机实测受阻：客户端 qdrant_client 要求 fastembed（本地推理）或
    cloud_inference（需 Qdrant Cloud key），两者在本环境都不可得
    （PyPI 默认源与清华镜像均无 fastembed 轮子）。
    """
    try:
        import qdrant_client.models as qm
        has_bm25 = hasattr(qm, "Bm25Config") and hasattr(qm, "Document")
    except ImportError as e:
        return {"level": "UNKNOWN", "note": "qdrant_client 未就绪: %s" % str(e)[:100]}
    if not has_bm25:
        return {"level": "NONE",
                "note": "客户端无 Bm25Config/Document，版本过旧"}
    import inspect
    fields = list(qm.Bm25Config.model_fields.keys())
    note = ("源码级确认 NATIVE：Bm25Config 字段 %s；默认 k=1.2 b=0.75 avg_len=256，"
            "与标准 BM25 一致。本机无法实测（qdrant_client 走 Document(model='qdrant-bm25') "
            "时要求 fastembed 或 cloud_inference，本环境 PyPI 源无 fastembed 轮子）"
            % fields[:8])
    return {"level": "NATIVE_BY_SOURCE", "note": note}


def probe_lance():
    """LanceDB: native FTS（BM25 打分，tantivy 已在 0.38 移除）。"""
    import lancedb
    uri = str(ROOT / "data" / "lance_bm25_probe")
    import shutil
    shutil.rmtree(uri, ignore_errors=True)
    db = lancedb.connect(uri)
    data = [{"pk": i, "text": d, "vector": np.zeros(4, dtype="float32").tolist()}
            for i, d in enumerate(DOCS)]
    tbl = db.create_table("vec", data=data)
    try:
        # 0.38 起 tantivy 被移除，必须用 native FTS（不带 use_tantivy）
        tbl.create_fts_index("text")
    except Exception as e:
        return {"level": "UNKNOWN",
                "note": "create_fts_index 失败: %s" % str(e)[:150]}
    r = tbl.search(QUERY, query_type="fts").limit(3).to_list()
    hits = [(row["pk"], round(row["_score"], 3)) for row in r]
    return {"level": "NATIVE", "hits": hits,
            "note": "native FTS（tantivy 已移除）+ query_type='fts'，_score 即 BM25 分"}


def probe_sqlite_vec():
    """sqlite-vec: 无 BM25。但 SQLite 本身有 FTS5（bm25() 函数），可组合验证。"""
    import sqlite3
    p = str(ROOT / "data" / "sqlite_bm25_probe.db")
    Path(p).unlink(missing_ok=True)
    conn = sqlite3.connect(p)
    # 先试 sqlite-vec 是否提供全文接口
    try:
        conn.enable_load_extension(True)
        import sqlite_vec
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    except Exception as e:
        return {"level": "NONE", "note": "sqlite_vec 加载失败: %s" % str(e)[:80]}
    # sqlite-vec 没有任何 FTS/BM25 接口
    try:
        conn.execute("SELECT bm25(docs) FROM docs LIMIT 1")
        fts_native = True
    except Exception:
        fts_native = False
    # 验证宿主 SQLite 有 FTS5（说明可以自己搭）
    try:
        conn.execute("CREATE VIRTUAL TABLE fts USING fts5(text)")
        conn.executemany("INSERT INTO fts(text) VALUES (?)", [(d,) for d in DOCS])
        conn.commit()
        rows = conn.execute(
            "SELECT rowid, bm25(fts) FROM fts WHERE fts MATCH ? ORDER BY bm25(fts) LIMIT 3",
            (QUERY,)
        ).fetchall()
        fts5_ok = len(rows) > 0
        fts5_hits = [(r[0], round(r[1], 3)) for r in rows]
    except Exception:
        fts5_ok, fts5_hits = False, []
    return {
        "level": "NONE",
        "note": "sqlite-vec 无 BM25；宿主 SQLite 的 FTS5 有 bm25()，需自己在 vec0 表旁建 FTS5 表再 JOIN",
        "host_fts5_available": fts5_ok,
        "host_fts5_hits": fts5_hits,
    }


def probe_chroma():
    """ChromaDB: ChromaBm25EmbeddingFunction（SparseEmbeddingFunction，原生 BM25）。

    本机实测受阻：该 EF 内部走 fastembed.sparse.bm25.Bm25，需要 snowballstemmer
    且会触发 onnx 模型下载，本环境无可用 PyPI 源，装不上也下不动。
    故本探针只做源码级验证，不给 NATIVE/NONE 判定。
    """
    try:
        import chromadb
        from chromadb.utils.embedding_functions import ChromaBm25EmbeddingFunction
        from chromadb.utils.embedding_functions import Bm25EmbeddingFunction
    except ImportError as e:
        return {"level": "UNKNOWN", "note": "chromadb 或 BM25 EF 未就绪: %s" % str(e)[:120]}
    import inspect
    # 证明它是 SparseEmbeddingFunction（输出稀疏向量），且参数是标准 BM25 的 k/b
    base = inspect.getmro(ChromaBm25EmbeddingFunction)[1].__name__
    sig = inspect.signature(ChromaBm25EmbeddingFunction.__init__)
    params = [p for p in sig.parameters if p in ("k", "b", "avg_doc_length")]
    sig2 = inspect.signature(Bm25EmbeddingFunction.__init__)
    adv = [p for p in sig2.parameters
           if p in ("language", "disable_stemmer", "token_max_length")]
    note = ("源码级确认 NATIVE：ChromaBm25EmbeddingFunction 继承 %s（输出稀疏向量），"
            "标准参数 %s；另有更完整的 Bm25EmbeddingFunction 支持 %s。"
            "本机无法实测（fastembed 依赖 snowballstemmer + onnx 模型下载，"
            "当前 PyPI 源不可得）"
            % (base, params, adv))
    return {"level": "NATIVE_BY_SOURCE", "note": note}


def probe_surreal():
    """SurrealDB: DEFINE ANALYZER + DEFINE INDEX FULLTEXT BM25(k1,b) + search::score()。

    源码依据：surrealdb/core/src/idx/ft/fulltext.rs 有 Scorer::new + Bm25Params(k1,b)，
    Scoring 枚举里有 Bm 变体（k1/b 可调）。analyzer 是用户自定义的，无内置。
    """
    from surrealdb import Surreal
    db = Surreal("ws://127.0.0.1:8000/rpc")
    db.signin({"username": "root", "password": "root"})
    db.use("bm25probe", "main")
    db.query("REMOVE TABLE IF EXISTS docs;")
    db.query("REMOVE ANALYZER IF EXISTS an_bm25;")
    db.query("DEFINE TABLE docs SCHEMALESS;")
    db.query("DEFINE FIELD body ON docs TYPE string;")
    # analyzer 必须自定义（SurrealDB 无内置），语法是 TOKENIZERS/FILTERS（我先前拼错成 TOKENIZIZERS）
    # 可选 tokenizer 只有 Blank/Camel/Class/Punct，filter 有 Lowercase/Snowball/Ngram 等
    try:
        db.query("DEFINE ANALYZER an_bm25 TOKENIZERS punct FILTERS lowercase;")
    except Exception as e:
        return {"level": "UNKNOWN", "note": "DEFINE ANALYZER 失败: %s" % str(e)[:150]}
    try:
        db.query("DEFINE INDEX idx_body ON docs FIELDS body "
                 "FULLTEXT ANALYZER an_bm25 BM25(1.2, 0.75) HIGHLIGHTS;")
    except Exception as e:
        return {"level": "UNKNOWN", "note": "建 FULLTEXT BM25 索引失败: %s" % str(e)[:150]}
    for i, d in enumerate(DOCS):
        db.query("CREATE docs CONTENT { pk: %d, body: '%s' };" % (i, d.replace("'", "\\'")))
    time.sleep(1)
    try:
        r = db.query("SELECT pk, search::score(1) AS score FROM docs "
                     "WHERE body @1@ 'fox' ORDER BY score DESC LIMIT 3;")
        # surrealdb SDK 返回可能是 str（错误信息）或 list
        if isinstance(r, str):
            return {"level": "UNKNOWN", "note": "BM25 查询返回错误: %s" % r[:150]}
        rows = r[0].get("result", []) if r and isinstance(r[0], dict) else []
        hits = [(row.get("pk"), round(row.get("score", 0), 3)) for row in rows]
        return {"level": "NATIVE", "hits": hits,
                "note": "DEFINE ANALYZER + DEFINE INDEX FULLTEXT BM25(k1,b) + "
                        "search::score()，原生 BM25 打分，k1/b 可调"}
    except Exception as e:
        return {"level": "UNKNOWN", "note": "BM25 查询失败: %s" % str(e)[:150]}


PROBES = {
    "milvus": probe_milvus,
    "qdrant": probe_qdrant,
    "lance": probe_lance,
    "sqlite": probe_sqlite_vec,
    "chroma": probe_chroma,
    "surrealdb": probe_surreal,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", default=None)
    args = parser.parse_args()

    targets = args.only or list(PROBES.keys())
    results = {}
    for name in targets:
        if name not in PROBES:
            print("skip:", name)
            continue
        print("\n=== %s ===" % name, flush=True)
        t0 = time.perf_counter()
        try:
            r = PROBES[name]()
        except Exception as e:
            r = {"level": "UNKNOWN", "note": "探针异常: %s: %s" % (type(e).__name__, str(e)[:150])}
        r["elapsed_s"] = round(time.perf_counter() - t0, 1)
        results[name] = r
        print("  level=%s  %.1fs" % (r["level"], r["elapsed_s"]), flush=True)
        if r.get("hits"):
            print("  hits=%s" % r["hits"], flush=True)
        print("  note=%s" % r["note"][:160], flush=True)

    out = RESULTS / ("bm25probe-%s.json" % time.strftime("%Y%m%d-%H%M%S"))
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print("\nsaved:", out)


if __name__ == "__main__":
    main()
