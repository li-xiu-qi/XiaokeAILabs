# -*- coding: utf-8 -*-
"""混合检索实验：纯向量 vs 纯 BM25 vs 混合（RRF）。

设计要点：
  库     = train 全部文档（文本 + 向量，11293 条）
  查询   = test 分层采样 40 条，与库内文档不重叠
  向量 GT = 暴力 L2 top10
  BM25 GT = 词项重叠集合（去停用词 + 实词长度过滤）

注意：查询文本必须与查询向量同源对齐。
早期版本用 test_texts[:40] 配分层采样向量，40 条里只有 4 条真正对应，
BM25 那一路搜的是无关文本，结论无效。现版本由 load_real(return_idx=True)
拿到采样行号，再用同一行号取文本。

评估指标：
  vector_hit      = 向量检索命中向量 GT 的比例
  bm25_hit        = BM25 检索命中 BM25 GT 的比例
  hybrid_vec_hit  = 混合检索命中向量 GT 的比例
  hybrid_bm25_hit = 混合检索命中 BM25 GT 的比例

用法:
  ./.venv/Scripts/python.exe scripts/hybrid_search_bench.py
"""
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

from index_bench import load_real  # noqa: E402

# 20news 是英文技术文本，通用词密度高，不滤停用词时词项重叠 GT 会覆盖
# 近半语料，指标失去区分度。这里用一份标准英文停用词表。
STOPWORDS = set("""
a about above after again against all am an and any are aren't as at be because been
before being below between both but by can't cannot could couldn't did didn't do does
doesn't doing don't down during each few for from further had hadn't has hasn't have
haven't having he he'd he'll he's her here here's hers herself him himself his how
how's i i'd i'll i'm i've if in into is isn't it it's its itself let's me more most
mustn't my myself no nor not of off on once only or other ought our ours ourselves out
over own same shan't she she'd she'll she's should shouldn't so some such than that
that's the their theirs them themselves then there there's these they they'd they'll
they're they've this those through to too under until up very was wasn't we we'd we'll
we're we've were weren't what what's when when's where where's which while who who's whom
why why's will with won't would wouldn't you you'd you'll you're you've your yours
yourself yourselves also however many may might must shall will can need said say says
get got go going one two three even like just know think dont doesnt didnt wont cant
isnt arent wasnt werent im ive id youre theyre thats whats
article writes write write wrote org com edu net gov mil subject lines from date
reply message posting host nntp path distribution organization keywords summary
""".split())

# 语料相关的高频噪声词：20news 邮件头残留与新闻组套话
CORPUS_STOP = {
    "writes", "write", "article", "apr", "aug", "dec", "feb", "jan", "jul", "jun",
    "mar", "may", "nov", "oct", "sep", "gmt", "utc", "nntp", "posting", "host",
    "distribution", "organization", "keywords", "summary", "lines", "subject",
    "message", "reply", "references", "sender", "followup", "xref", "path",
    "buphy", "bu", "edu", "wrote", "says", "said", "one", "two", "would", "could",
    "know", "think", "people", "thing", "things", "get", "got", "make", "made",
    "well", "even", "also", "much", "many", "way", "use", "used", "using",
    "dont", "doesnt", "didnt", "isnt", "arent", "wasnt", "werent", "cant", "wont",
    "im", "ive", "id", "youre", "theyre", "thats", "whats", "hes", "shes", "theyre",
}


def rrf_fuse(rank_lists, k=60):
    """Reciprocal Rank Fusion：多路召回结果融合。"""
    scores = {}
    for lst in rank_lists:
        for rank, doc_id in enumerate(lst):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


def hit_at_k(retrieved, truth, k=10):
    if not truth:
        return 0.0
    return len(set(retrieved[:k]) & truth) / min(len(truth), k)


def l2_truth(docs_vec, query_vec, k=10):
    """向量 ground truth：暴力 L2 top-k。"""
    d = np.sum((docs_vec - query_vec) ** 2, axis=1)
    return set(np.argsort(d)[:k].tolist())


def token_set(text):
    """实词集合：小写化，滤掉停用词、语料噪声词与长度<=2 的词。"""
    import re
    return {w for w in re.findall(r"[a-z0-9]+", text.lower())
            if len(w) > 2 and w not in STOPWORDS and w not in CORPUS_STOP}


def scan_threshold(docs_text, q_texts):
    """扫 min_overlap，打印各阈值下 GT 覆盖语料的比例，供报告引用。

    阈值太小则 GT 覆盖近半语料，hit_at_k 退化为「抽到就给分」，没有区分度；
    阈值太大则 GT 趋空，分母塌缩。这里只做诊断，不自动选阈值。
    """
    dt = [token_set(t) for t in docs_text]
    print("\n=== BM25 GT 阈值扫描（库 %d 条）===" % len(docs_text))
    rows = []
    for mo in [2, 3, 4, 5, 6, 8]:
        sizes = [sum(1 for t in dt if len(token_set(q) & t) >= mo) if token_set(q) else 0
                 for q in q_texts]
        avg = float(np.mean(sizes))
        pct = 100.0 * avg / len(docs_text)
        empty = sum(1 for s in sizes if s == 0)
        rows.append((mo, avg, pct))
        print("  min_overlap=%-3d 平均 GT=%-8.1f 占库 %5.1f%%  空 GT 查询数=%d"
              % (mo, avg, pct, empty))
    return rows


def bm25_truth(docs_text, query_text, min_overlap):
    """BM25 ground truth：与查询共享至少 min_overlap 个实词的文档。"""
    qt = token_set(query_text)
    if not qt:
        return set()
    return {i for i, t in enumerate(docs_text) if len(qt & token_set(t)) >= min_overlap}


def bench_milvus(docs_vec, docs_text, q_vecs, q_texts, min_overlap):
    """Milvus: 纯向量 / 纯 BM25 / 混合 RRF。"""
    from pymilvus import MilvusClient, DataType, Function, FunctionType
    from pymilvus.milvus_client.index import IndexParams

    c = MilvusClient(uri="http://127.0.0.1:19530")
    coll = "hybridbench"
    if c.has_collection(coll):
        c.drop_collection(coll)
    schema = c.create_schema(auto_id=False)
    schema.add_field("pk", DataType.INT64, is_primary=True)
    schema.add_field("text", DataType.VARCHAR, max_length=4000, enable_analyzer=True)
    schema.add_field("dense", DataType.FLOAT_VECTOR, dim=docs_vec.shape[1])
    schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_function(Function(
        name="text_bm25_emb", input_field_names=["text"],
        output_field_names=["sparse"], function_type=FunctionType.BM25))
    c.create_collection(collection_name=coll, schema=schema)
    B = 2000
    for s in range(0, len(docs_vec), B):
        e = min(s + B, len(docs_vec))
        c.insert(collection_name=coll, data=[
            {"pk": i, "text": docs_text[i], "dense": docs_vec[i].tolist()}
            for i in range(s, e)
        ])
    ip = IndexParams()
    ip.add_index(field_name="dense", index_type="HNSW", metric_type="L2",
                 params={"M": 16, "efConstruction": 200})
    ip.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX",
                 metric_type="BM25")
    c.create_index(collection_name=coll, index_params=ip)
    c.load_collection(collection_name=coll)
    time.sleep(8)

    out = []
    for qi in range(len(q_vecs)):
        rv = c.search(collection_name=coll, data=[q_vecs[qi].tolist()],
                      anns_field="dense", limit=10, output_fields=[])
        vec_ids = [h["pk"] for h in rv[0]]
        rb = c.search(collection_name=coll, data=[q_texts[qi]],
                      anns_field="sparse", limit=10, output_fields=[])
        bm25_ids = [h["pk"] for h in rb[0]]
        hybrid_ids = rrf_fuse([vec_ids, bm25_ids])[:10]
        v_gt = l2_truth(docs_vec, q_vecs[qi])
        b_gt = bm25_truth(docs_text, q_texts[qi], min_overlap)
        out.append({
            "query": q_texts[qi][:60],
            "vec_gt_size": len(v_gt), "bm25_gt_size": len(b_gt),
            "vector_hit": round(hit_at_k(vec_ids, v_gt), 3),
            "bm25_hit": round(hit_at_k(bm25_ids, b_gt), 3),
            "hybrid_vec_hit": round(hit_at_k(hybrid_ids, v_gt), 3),
            "hybrid_bm25_hit": round(hit_at_k(hybrid_ids, b_gt), 3),
        })
    return out


def bench_lance(docs_vec, docs_text, q_vecs, q_texts, min_overlap):
    """LanceDB: 纯向量 / 纯 FTS(BM25) / 混合。"""
    import lancedb
    import shutil
    uri = str(ROOT / "data" / "lance_hybrid")
    shutil.rmtree(uri, ignore_errors=True)
    db = lancedb.connect(uri)
    db.create_table("vec", data=[
        {"pk": i, "text": docs_text[i], "vector": docs_vec[i].tolist()}
        for i in range(len(docs_vec))
    ])
    tbl = db.open_table("vec")
    tbl.create_index("vector")
    tbl.create_fts_index("text")

    out = []
    for qi in range(len(q_vecs)):
        vec_rows = tbl.search(q_vecs[qi].tolist()).limit(10).to_list()
        vec_ids = [r["pk"] for r in vec_rows]
        fts_rows = tbl.search(q_texts[qi], query_type="fts").limit(10).to_list()
        bm25_ids = [r["pk"] for r in fts_rows]
        hybrid_ids = rrf_fuse([vec_ids, bm25_ids])[:10]
        v_gt = l2_truth(docs_vec, q_vecs[qi])
        b_gt = bm25_truth(docs_text, q_texts[qi], min_overlap)
        out.append({
            "query": q_texts[qi][:60],
            "vec_gt_size": len(v_gt), "bm25_gt_size": len(b_gt),
            "vector_hit": round(hit_at_k(vec_ids, v_gt), 3),
            "bm25_hit": round(hit_at_k(bm25_ids, b_gt), 3),
            "hybrid_vec_hit": round(hit_at_k(hybrid_ids, v_gt), 3),
            "hybrid_bm25_hit": round(hit_at_k(hybrid_ids, b_gt), 3),
        })
    return out


def bench_surreal(docs_vec, docs_text, q_vecs, q_texts, min_overlap):
    """SurrealDB: 纯向量 / 纯 BM25 / 混合。

    注意：此函数本轮未启用。SurrealDB 3.2.4 的 BM25 打分在探针中确认为真
    （search::score() 返回合理分值，k1/b 可调），但在此语料上全文检索检索不到
    结果：单词查询部分命中（'karner'→1 条，'the'→3 条），'article'(df=173)、
    '1993'(df=23) 返回 0 条，所有多词查询（@1@..@5@ 全试过）返回 0 条。
    该行为未定位，暂不作为混合检索对比的一方。

    向量走 HNSW 索引，3.x 起必须用 <|k,EF|> 语法（<|k|> 的 KTree/M-Tree 已移除）。
    db.query() 直接返回行列表，不是 [{'result': ...}] 结构。
    """
    from surrealdb import Surreal

    db = Surreal("ws://127.0.0.1:8000/rpc")
    db.signin({"username": "root", "password": "root"})
    db.use("hybridbench", "main")
    db.query("REMOVE TABLE IF EXISTS docs;")
    db.query("REMOVE ANALYZER IF EXISTS an_bm25;")
    db.query("DEFINE TABLE docs SCHEMALESS;")
    db.query("DEFINE FIELD pk ON docs TYPE int;")
    db.query("DEFINE FIELD body ON docs TYPE string;")
    db.query("DEFINE FIELD vec ON docs TYPE array<float>;")
    db.query("DEFINE ANALYZER an_bm25 TOKENIZERS punct FILTERS lowercase;")
    db.query("DEFINE INDEX idx_body ON docs FIELDS body "
             "FULLTEXT ANALYZER an_bm25 BM25(1.2, 0.75) HIGHLIGHTS;")
    db.query("DEFINE INDEX idx_vec ON docs FIELDS vec HNSW DIMENSION %d;" % docs_vec.shape[1])

    B = 500
    for s in range(0, len(docs_vec), B):
        e = min(s + B, len(docs_vec))
        db.query("INSERT INTO docs $rows", {"rows": [
            {"pk": i, "body": docs_text[i], "vec": docs_vec[i].tolist()}
            for i in range(s, e)
        ]})
    time.sleep(3)

    out = []
    for qi in range(len(q_vecs)):
        rv = db.query("SELECT pk FROM docs WHERE vec <|10,20|> $q;",
                      {"q": q_vecs[qi].tolist()})
        vec_ids = [row["pk"] for row in (rv or [])]
        rb = db.query(
            "SELECT pk, search::score(1) AS score FROM docs "
            "WHERE body @1@ $q ORDER BY score DESC LIMIT 10;",
            {"q": q_texts[qi]})
        bm25_ids = [row["pk"] for row in (rb or [])]
        hybrid_ids = rrf_fuse([vec_ids, bm25_ids])[:10]
        v_gt = l2_truth(docs_vec, q_vecs[qi])
        b_gt = bm25_truth(docs_text, q_texts[qi], min_overlap)
        out.append({
            "query": q_texts[qi][:60],
            "vec_gt_size": len(v_gt), "bm25_gt_size": len(b_gt),
            "vector_hit": round(hit_at_k(vec_ids, v_gt), 3),
            "bm25_hit": round(hit_at_k(bm25_ids, b_gt), 3),
            "hybrid_vec_hit": round(hit_at_k(hybrid_ids, v_gt), 3),
            "hybrid_bm25_hit": round(hit_at_k(hybrid_ids, b_gt), 3),
        })
    return out


def main():
    # min_overlap 校准：去停用词后词项重叠大幅下降，扫描区间要下移。
    # min_overlap=3 → GT 平均 1369 条（占库 12.1%），无空 GT，top10 指标有区分度。
    # 阈值过小（2 → 28.2%）则 GT 覆盖近三成语料，hit_at_k 失去区分度。
    min_overlap = 3
    vecs, queries, dim, idx = load_real(40, 0, return_idx=True)
    texts_p = ROOT / "data" / "20news_texts.npz"
    if not texts_p.exists():
        print("缺 data/20news_texts.npz")
        return
    z = np.load(texts_p, allow_pickle=True)
    docs_text = [str(t) for t in z["train_texts"][:len(vecs)]]
    # 关键：查询文本必须用同一批采样行号取，不能取 test_texts[:40]
    q_texts = [str(z["test_texts"][i]) for i in idx]

    print("库: %d 条文档, 查询: %d 条, min_overlap=%d" % (len(docs_text), len(queries), min_overlap))
    scan_threshold(docs_text, q_texts)

    results = {"meta": {"n_docs": len(docs_text), "n_queries": len(queries),
                        "min_overlap": min_overlap}}
    # surreal 本轮不跑：其全文检索在此语料上检索不到结果，见 bench_surreal 文档字符串
    for name, fn in [("milvus", bench_milvus), ("lance", bench_lance)]:
        print("\n=== %s 混合检索 ===" % name, flush=True)
        try:
            results[name] = fn(vecs, docs_text, queries, q_texts, min_overlap)
        except Exception as e:
            print("  %s 失败: %s: %s" % (name, type(e).__name__, str(e)[:300]))
            results[name] = {"error": "%s: %s" % (type(e).__name__, str(e)[:300])}

    out = RESULTS / ("hybridbench-%s.json" % time.strftime("%Y%m%d-%H%M%S"))
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print("\nsaved:", out)

    keys = ["vector_hit", "bm25_hit", "hybrid_vec_hit", "hybrid_bm25_hit"]
    for name in ["milvus", "lance"]:
        rows = results[name]
        if isinstance(rows, dict):
            print("\n%s: %s" % (name, rows["error"]))
            continue
        print("\n--- %s（%d 条查询均值）---" % (name, len(rows)))
        print("%-18s %s" % ("query", "  ".join("%8s" % k.replace("_hit", "") for k in keys)))
        for r in rows[:6]:
            print("%-18s %s" % (r["query"][:16],
                                "  ".join("%8.3f" % r[k] for k in keys)))
        avg = {k: float(np.mean([r[k] for r in rows])) for k in keys}
        print("%-18s %s" % ("AVG", "  ".join("%8.3f" % avg[k] for k in avg)))


if __name__ == "__main__":
    main()
