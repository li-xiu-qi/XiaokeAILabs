# -*- coding: utf-8 -*-
"""向量库索引矩阵基准：同一份数据、同一查询集，按 (库 x 索引类型) 两重循环跑。

覆盖 multi_bench.py 缺的三块：
  1. 索引类型维度   每个库跑它实际支持的索引，不只一种
  2. 索引构建时间   build_index() 单独计时，含 Milvus 的异步就绪等待
  3. 存储分段记账   空库 / 写入后 / 建索引后 三段字节数，可减出索引净开销

adapter 接口：
    setup(spec)            建表，索引规格由 spec 指定
    write(vecs)            批量写入，返回耗时秒
    build_index()          建索引，返回耗时秒（含等待就绪）
    query(q, ef)           单条 KNN，返回 (命中 pk 集合, 耗时 ms)
    storage()              返回当前落盘字节数
    teardown()             清理

用法:
  ./.venv/Scripts/python.exe scripts/index_bench.py
  ./.venv/Scripts/python.exe scripts/index_bench.py --only milvus qdrant
  ./.venv/Scripts/python.exe scripts/index_bench.py --index HNSW IVF_PQ FLAT
"""
import argparse
import importlib
import json
import os
import platform
import shutil
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

MILVUS_URI = "http://127.0.0.1:19530"
QDRANT_URI = "http://127.0.0.1:6333"
SURREAL_URI = "ws://127.0.0.1:8000/rpc"

# Milvus 3.0 standalone 的 minio 数据根（容器 bind mount 到 config/volumes/minio）
MILVUS_MINIO = ROOT / "config" / "volumes" / "minio" / "a-bucket" / "files"
# SurrealDB rocksdb 每轮一个独立目录，避免 drop table 后空间不回收污染增量
# SurrealDB 每轮起独立进程（独立端口 + 独立数据目录）。同实例 drop table 不回收
# rocksdb 空间，共享目录会让增量测量逐轮偏高。
SURREAL_BIN = ROOT / "bin" / "surreal.exe"
SURREAL_DATA = ROOT / "data" / "surreal"
SURREAL_PORT_BASE = 8100
QDRANT_DATA = ROOT / "data" / "qdrant" / "collections"
LANCE_DATA = ROOT / "data" / "lance"
SQLITE_DATA = ROOT / "data"
CHROMA_DATA = ROOT / "data" / "chroma"


def dir_bytes(p):
    """目录落盘字节数。路径不存在返回 0。"""
    p = Path(p)
    if not p.exists():
        return 0
    if p.is_file():
        try:
            return p.stat().st_size
        except OSError:
            return 0
    t = 0
    for r, _d, f in os.walk(p):
        for x in f:
            try:
                t += os.path.getsize(os.path.join(r, x))
            except OSError:
                pass
    return t


# ---------------------------------------------------------------- 数据

def load_real(n_queries=40, seed=0, return_idx=False):
    """20 Newsgroups 真实 embedding：train 入库，test 分层采样做查询。

    return_idx=True 时额外返回采样到的 test 行号，供调用方对齐文本。
    """
    p = ROOT / "data" / "20news_emb.npz"
    if not p.exists():
        raise SystemExit("缺 data/20news_emb.npz，先跑 scripts/gen_emb.py")
    z = np.load(p, allow_pickle=True)
    vecs = z["train_emb"].astype("float32")
    test_emb = z["test_emb"].astype("float32")
    test_labels = z["test_labels"]
    rng = np.random.default_rng(seed)
    cats = list(dict.fromkeys(test_labels.tolist()))
    per = max(1, n_queries // len(cats))
    idx = []
    for c in cats:
        cand = np.where(test_labels == c)[0]
        take = min(per, len(cand))
        idx.extend(rng.choice(cand, take, replace=False).tolist())
    idx = np.array(sorted(idx[:n_queries]))
    if return_idx:
        return vecs, test_emb[idx], int(vecs.shape[1]), idx
    return vecs, test_emb[idx], int(vecs.shape[1])


def ground_truth(vecs, queries, k):
    """暴力精确 top-k。用 |a-b|^2 = |a|^2+|b|^2-2ab 展开式避免大中间张量。"""
    vv = np.sum(vecs ** 2, axis=1)
    qq = np.sum(queries ** 2, axis=1)
    out = []
    for i in range(0, len(queries), 25):
        q = queries[i:i + 25]
        d = qq[i:i + 25, None] + vv[None, :] - 2.0 * (q @ vecs.T)
        out.append(np.argsort(d, axis=1)[:, :k])
    return np.concatenate(out, axis=0)


# ---------------------------------------------------------------- Milvus

class MilvusAdapter:
    """索引规格 -> (index_type, params, search_params_fn)。

    IVF_* 的 nlist 必须 >= 数据量下限，11293 条用 nlist=128 偏大但可跑；
    nlist 太小召回掉得快，取 128 是召回与构建耗时的折中。
    """

    name = "Milvus"
    version = "v3.0.0"

    SPECS = {
        "FLAT":      dict(index_type="FLAT", params={},
                          search=lambda ef: {"metric_type": "L2", "params": {}}),
        "IVF_FLAT":  dict(index_type="IVF_FLAT", params={"nlist": 128},
                          search=lambda ef: {"metric_type": "L2", "params": {"nprobe": ef}}),
        "IVF_SQ8":   dict(index_type="IVF_SQ8", params={"nlist": 128},
                          search=lambda ef: {"metric_type": "L2", "params": {"nprobe": ef}}),
        "IVF_PQ":    dict(index_type="IVF_PQ", params={"nlist": 128, "m": 8, "nbits": 8},
                          search=lambda ef: {"metric_type": "L2", "params": {"nprobe": ef}}),
        "HNSW":      dict(index_type="HNSW", params={"M": 16, "efConstruction": 200},
                          search=lambda ef: {"metric_type": "L2", "params": {"ef": ef}}),
        "DISKANN":   dict(index_type="DISKANN", params={},
                          search=lambda ef: {"metric_type": "L2", "params": {"search_list": ef}}),
        "SCANN":     dict(index_type="SCANN", params={"nlist": 128, "with_raw_data": True},
                          search=lambda ef: {"metric_type": "L2",
                                             "params": {"nprobe": ef, "reorder_k": 100}}),
    }

    def __init__(self, dim, ns="idxbench", with_text=False):
        self.dim = dim
        self.ns = ns
        self.with_text = with_text

    def _coll(self, spec):
        return "%s_%s" % (self.ns, spec.lower())

    def setup(self, spec):
        from pymilvus import MilvusClient, CollectionSchema, FieldSchema, DataType
        self.spec = spec
        self.coll = self._coll(spec)
        self.c = MilvusClient(uri=MILVUS_URI)
        if self.c.has_collection(self.coll):
            self.c.drop_collection(self.coll)
        while self.c.has_collection(self.coll):
            time.sleep(0.2)
        # 走 schema 路径：MilvusClient 的快捷建表会默认挂 AUTOINDEX，
        # 之后再 create_index 会撞 "at most one distinct index is allowed per field"
        schema_fields = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
        ]
        if self.with_text:
            # 20news 正文截断到 1200 字符，max_length 取 2048 留余量
            schema_fields.append(FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2048))
        schema = CollectionSchema(fields=schema_fields)
        self.c.create_collection(collection_name=self.coll, schema=schema)
        self.cid = self.c.describe_collection(self.coll).get("collection_id")

    def write(self, vecs, texts=None):
        t0 = time.perf_counter()
        for s in range(0, len(vecs), 5000):
            e = min(s + 5000, len(vecs))
            data = [{"pk": int(i), "vector": vecs[i].tolist()} for i in range(s, e)]
            if texts is not None:
                for j, i in enumerate(range(s, e)):
                    data[j]["text"] = texts[i]
            self.c.insert(collection_name=self.coll, data=data)
        self.c.flush(collection_name=self.coll)
        return time.perf_counter() - t0

    def build_index(self):
        """返回索引构建耗时。Milvus 3.0 索引异步，必须轮询到 Finished
        且 pending_index_rows 归零，否则查询打在半成品索引上。"""
        from pymilvus.milvus_client.index import IndexParams
        sp = self.SPECS[self.spec]
        t0 = time.perf_counter()
        ip = IndexParams()
        ip.add_index(field_name="vector", index_type=sp["index_type"],
                     metric_type="L2", params=sp["params"] if sp["params"] else None)
        self.c.create_index(collection_name=self.coll, index_params=ip)
        self.c.load_collection(collection_name=self.coll)
        deadline = time.time() + 300
        while time.time() < deadline:
            d = self.c.describe_index(collection_name=self.coll, index_name="vector")
            if d.get("state") == "Finished" and d.get("pending_index_rows", 0) == 0:
                break
            time.sleep(0.5)
        return time.perf_counter() - t0

    def query(self, q, ef):
        t0 = time.perf_counter()
        r = self.c.search(collection_name=self.coll, data=[q.tolist()], limit=10,
                          search_params=self.SPECS[self.spec]["search"](ef),
                          output_fields=["pk"])
        return {h["entity"]["pk"] for h in r[0]}, (time.perf_counter() - t0) * 1000

    def _snapshot(self):
        """扫描 minio 对象，返回 {(kind, top_id): bytes}。

        Milvus 3.0 对象路径是 files/<kind>/<id>/1/<seg>/<field>/<idx>/<uuid>/part.N，
        第一段 id 与 describe_collection 的 collection_id 不是同一个值（实测不匹配），
        无法用它过滤。改为按 (kind, id) 聚合，跑分前后各拍一次快照取差分，
        只统计本轮新增的对象，避免上一轮 drop_collection 未回收的空间污染基线。
        """
        agg = {}
        if not MILVUS_MINIO.exists():
            return agg
        for kind in ("insert_log", "stats_log", "index_files"):
            kd = MILVUS_MINIO / kind
            if not kd.exists():
                continue
            for sub in kd.iterdir():
                if not sub.is_dir():
                    continue
                t = dir_bytes(sub)
                if t:
                    agg[(kind, sub.name)] = t
        return agg

    def storage(self):
        """本轮新增对象的字节数。基线在 snapshot_base() 时定格，之后不再更新，
        所以 base / after_write / after_index 三次调用都减去同一个基线，
        差值就是各阶段实际落盘的增量。"""
        agg = self._snapshot()
        base = getattr(self, "_base_agg", {})
        return sum(v for k, v in agg.items() if k not in base)

    def snapshot_base(self):
        self._base_agg = self._snapshot()

    def teardown(self):
        try:
            if self.c.has_collection(self.coll):
                self.c.drop_collection(self.coll)
        except Exception:
            pass


# ---------------------------------------------------------------- Qdrant

class QdrantAdapter:
    name = "Qdrant"
    version = "1.19.0"

    SPECS = {
        "HNSW_m16":     dict(hnsw=dict(m=16, ef_construct=200), quant=None,
                             search=lambda ef: {"hnsw_ef": ef, "exact": False}),
        "HNSW_m32":     dict(hnsw=dict(m=32, ef_construct=256), quant=None,
                             search=lambda ef: {"hnsw_ef": ef, "exact": False}),
        "FLAT":         dict(hnsw=dict(m=0), quant=None,
                             search=lambda ef: {"exact": True}),
        "HNSW_ScalarI8": dict(hnsw=dict(m=16, ef_construct=200), quant="scalar_i8",
                             search=lambda ef: {"hnsw_ef": ef, "exact": False}),
        "HNSW_PQ":      dict(hnsw=dict(m=16, ef_construct=200), quant="pq",
                             search=lambda ef: {"hnsw_ef": ef, "exact": False}),
        "HNSW_Binary":  dict(hnsw=dict(m=16, ef_construct=200), quant="binary",
                             search=lambda ef: {"hnsw_ef": ef, "exact": False}),
    }

    def __init__(self, dim, ns="idxbench", with_text=False):
        self.dim = dim
        self.ns = ns
        self.with_text = with_text

    def setup(self, spec):
        from qdrant_client import QdrantClient
        from qdrant_client.models import (Distance, VectorParams, HnswConfigDiff,
                                          ScalarQuantization, ScalarQuantizationConfig,
                                          ScalarType, ProductQuantization,
                                          ProductQuantizationConfig, BinaryQuantization,
                                          BinaryQuantizationConfig, CompressionRatio)
        self.spec = spec
        self.coll = "%s_%s" % (self.ns, spec.lower())
        sp = self.SPECS[spec]
        self.c = QdrantClient(url=QDRANT_URI, timeout=120)
        if self.c.collection_exists(self.coll):
            self.c.delete_collection(self.coll)
        vp = VectorParams(size=self.dim, distance=Distance.EUCLID,
                          hnsw_config=HnswConfigDiff(**sp["hnsw"]))
        kw = dict(collection_name=self.coll, vectors_config=vp)
        q = sp["quant"]
        if q == "scalar_i8":
            kw["quantization_config"] = ScalarQuantization(scalar=ScalarQuantizationConfig(
                type=ScalarType.INT8, quantile=0.99, always_ram=True))
        elif q == "pq":
            kw["quantization_config"] = ProductQuantization(product=ProductQuantizationConfig(
                compression=CompressionRatio.X16, always_ram=True))
        elif q == "binary":
            kw["quantization_config"] = BinaryQuantization(binary=BinaryQuantizationConfig(
                always_ram=True))
        self.c.create_collection(**kw)

    def write(self, vecs, texts=None):
        from qdrant_client.models import PointStruct
        t0 = time.perf_counter()
        for s in range(0, len(vecs), 1000):
            e = min(s + 1000, len(vecs))
            self.c.upsert(collection_name=self.coll, points=[
                PointStruct(id=i, vector=vecs[i].tolist(),
                            payload={"pk": i, "text": texts[i]} if texts is not None else {"pk": i})
                for i in range(s, e)])
        return time.perf_counter() - t0

    def build_index(self):
        """Qdrant 的 HNSW 是随写入增量构建的，没有独立的建索引阶段。
        create_collection 时图参数即定，返回时间近似为 0。
        这里等索引追上写入（indexed_vectors_count == points_count）后计时结束。"""
        from qdrant_client.models import OptimizersConfigDiff
        t0 = time.perf_counter()
        self.c.update_collection(collection_name=self.coll,
                                 optimizer_config=OptimizersConfigDiff(indexing_threshold=1))
        deadline = time.time() + 120
        while time.time() < deadline:
            info = self.c.get_collection(self.coll)
            if info.indexed_vectors_count and info.indexed_vectors_count >= info.points_count:
                break
            time.sleep(0.5)
        return time.perf_counter() - t0

    def query(self, q, ef):
        t0 = time.perf_counter()
        r = self.c.query_points(collection_name=self.coll, query=q.tolist(), limit=10,
                                search_params=self.SPECS[self.spec]["search"](ef))
        return {p.payload["pk"] for p in r.points}, (time.perf_counter() - t0) * 1000

    def storage(self):
        return dir_bytes(QDRANT_DATA / self.coll)

    def teardown(self):
        """delete_collection 的空间释放是异步的，不等就进下一轮会让下一轮的
        empty_bytes 带上上一轮残留（实测残留可达数百 MB）。"""
        try:
            if self.c.collection_exists(self.coll):
                self.c.delete_collection(self.coll)
            d = QDRANT_DATA / self.coll
            for _ in range(60):
                if not d.exists():
                    return
                time.sleep(0.5)
        except Exception:
            pass


# ---------------------------------------------------------------- SurrealDB

class SurrealAdapter:
    name = "SurrealDB"
    version = "3.2.4"

    SPECS = {
        "HNSW_m16": dict(efc=200, m=16),
        "HNSW_m32": dict(efc=256, m=32),
    }

    def __init__(self, dim, ns="idxbench", with_text=False):
        self.dim = dim
        self.ns = ns
        self.with_text = with_text

    # 端口按 spec 顺序分配，保证两种规格不撞端口
    _PORT_SEQ = []

    def setup(self, spec):
        """起一个专属 SurrealDB 进程。rocksdb 路径是启动参数决定的，
        连接时改不了；同实例 drop table 也不回收空间，所以每轮必须独立实例。"""
        import subprocess
        import socket
        from surrealdb import Surreal
        self.spec = spec
        if not SurrealAdapter._PORT_SEQ:
            SurrealAdapter._PORT_SEQ.append(SURREAL_PORT_BASE)
        self.port = SurrealAdapter._PORT_SEQ.pop(0)
        SurrealAdapter._PORT_SEQ.append(self.port + len(self.SPECS))
        self.uri = "ws://127.0.0.1:%d/rpc" % self.port
        self.dbpath = SURREAL_DATA / ("%s_%s" % (self.ns, spec.lower()))
        if self.dbpath.exists():
            shutil.rmtree(self.dbpath, ignore_errors=True)
        self.dbpath.mkdir(parents=True, exist_ok=True)
        self.proc = subprocess.Popen(
            [str(SURREAL_BIN), "start", "--user", "root", "--pass", "root",
             "--bind", "127.0.0.1:%d" % self.port,
             "rocksdb:%s" % self.dbpath],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # 等端口起来
        for _ in range(100):
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.5):
                    break
            except OSError:
                time.sleep(0.2)
        self.db = Surreal(self.uri)
        self.db.signin({"username": "root", "password": "root"})
        self.db.use(self.ns, "main")

    def _ddl(self):
        sp = self.SPECS[self.spec]
        txt_field = "DEFINE FIELD txt ON vec TYPE string; " if self.with_text else ""
        return ("DEFINE TABLE vec SCHEMALESS; "
                "DEFINE FIELD pk ON vec TYPE int; "
                "DEFINE FIELD emb ON vec TYPE array<float>; "
                + txt_field +
                "DEFINE INDEX emb_idx ON vec FIELDS emb HNSW DIMENSION %d "
                "DIST EUCLIDEAN EFC %d M %d;" % (self.dim, sp["efc"], sp["m"]))

    def write(self, vecs, texts=None):
        import json
        def arr(v):
            return "[" + ",".join("%.6f" % x for x in v) + "]"
        self.db.query("REMOVE TABLE IF EXISTS vec;")
        self.db.query(self._ddl())
        t0 = time.perf_counter()
        # 400 一批：更大批次触发 HTTP 413 / WS 断连（实测阈值）
        for s in range(0, len(vecs), 400):
            e = min(s + 400, len(vecs))
            parts = []
            for i in range(s, e):
                if texts is not None:
                    # json.dumps 产出合法 SurrealQL 字符串字面量（转义引号/反斜杠/换行）
                    parts.append("{pk: %d, emb: %s, txt: %s}" % (i, arr(vecs[i]), json.dumps(texts[i])))
                else:
                    parts.append("{pk: %d, emb: %s}" % (i, arr(vecs[i])))
            self.db.query("INSERT INTO vec [%s];" % ",".join(parts))
        return time.perf_counter() - t0

    def build_index(self):
        """索引在 write() 里已随 DEFINE INDEX 建好（SurrealDB 的 HNSW 索引
        定义即开始构建）。此处补一次空转计时，使接口与其他库一致。"""
        t0 = time.perf_counter()
        self.db.query("RETURN 1;")
        return time.perf_counter() - t0

    def query(self, q, ef):
        arr = "[" + ",".join("%.6f" % x for x in q) + "]"
        t0 = time.perf_counter()
        res = self.db.query("SELECT pk FROM vec WHERE emb <|10,%d|> %s;" % (ef, arr))
        dt = (time.perf_counter() - t0) * 1000
        rows = res[0] if isinstance(res, list) and res and isinstance(res[0], list) else res
        return {r["pk"] for r in rows}, dt

    def storage(self):
        return dir_bytes(self.dbpath)

    def teardown(self):
        try:
            self.db.query("REMOVE TABLE IF EXISTS vec;")
            self.db.close()
        except Exception:
            pass
        try:
            self.proc.terminate()
            self.proc.wait(timeout=10)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass


# ---------------------------------------------------------------- LanceDB

class LanceAdapter:
    name = "LanceDB"
    version = "0.38.0"

    SPECS = {
        # IVF_PQ 参数由 probe_lance.py 在真实数据上扫出：sub=64/part=128 配 refine
        # 召回 0.98~1.00。refine 让 LanceDB 变成粗排+精排，与纯 HNSW 不同构，
        # 报告里必须单列标注。
        "IVF_PQ":    dict(index_type="IVF_PQ", refine=10),
        "IVF_FLAT":  dict(index_type="IVF_FLAT", refine=0),
        "IVF_SQ":    dict(index_type="IVF_SQ", refine=10),
        "HNSW_PQ":   dict(index_type="HNSW_PQ", refine=0),
    }

    def __init__(self, dim, ns="idxbench", with_text=False):
        self.dim = dim
        self.ns = ns
        self.with_text = with_text

    def setup(self, spec):
        import lancedb
        self.spec = spec
        self.uri = str(LANCE_DATA / ("%s_%s" % (self.ns, spec.lower())))
        self.db = lancedb.connect(self.uri)
        self.tbl_name = "vec"
        if self.tbl_name in self.db.table_names():
            self.db.drop_table(self.tbl_name)

    def write(self, vecs, texts=None):
        import pandas as pd
        t0 = time.perf_counter()
        if texts is not None:
            data = [{"pk": int(i), "vector": vecs[i].tolist(), "text": texts[i]}
                    for i in range(len(vecs))]
        else:
            data = [{"pk": int(i), "vector": vecs[i].tolist()} for i in range(len(vecs))]
        self.tbl = self.db.create_table(self.tbl_name, data=data)
        return time.perf_counter() - t0

    def build_index(self):
        from lancedb.index import (IvfPq, IvfFlat, IvfSq, HnswPq)
        cfg = {"IVF_PQ": IvfPq, "IVF_FLAT": IvfFlat,
               "IVF_SQ": IvfSq, "HNSW_PQ": HnswPq}[self.SPECS[self.spec]["index_type"]]
        t0 = time.perf_counter()
        self.tbl.create_index("vector", config=cfg())
        return time.perf_counter() - t0

    def query(self, q, ef):
        t0 = time.perf_counter()
        s = self.tbl.search(q.tolist()).limit(10).nprobes(ef)
        rf = self.SPECS[self.spec]["refine"]
        if rf:
            s = s.refine_factor(rf)
        r = s.to_list()
        return {row["pk"] for row in r}, (time.perf_counter() - t0) * 1000

    def storage(self):
        return dir_bytes(Path(self.uri) / (self.tbl_name + ".lance"))

    def teardown(self):
        try:
            if self.tbl_name in self.db.table_names():
                self.db.drop_table(self.tbl_name)
        except Exception:
            pass


# ---------------------------------------------------------------- sqlite-vec

class SqliteVecAdapter:
    """无 ANN 索引，只有暴力扫描。ef 参数无效，作为对照组。"""

    name = "sqlite-vec"
    version = "0.1.9"

    SPECS = {"FLAT": dict()}

    def __init__(self, dim, ns="idxbench", with_text=False):
        self.dim = dim
        self.ns = ns
        self.with_text = with_text
        self.path = SQLITE_DATA / ("sqlite_vec_%s.db" % ns)

    def setup(self, spec):
        import sqlite3
        import sqlite_vec
        self.spec = spec
        Path(self.path).unlink(missing_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)
        if self.with_text:
            self.conn.execute(
                "CREATE VIRTUAL TABLE vec USING vec0("
                "pk integer primary key, embedding float[%d], text text)" % self.dim)
        else:
            self.conn.execute(
                "CREATE VIRTUAL TABLE vec USING vec0(pk integer primary key, embedding float[%d])"
                % self.dim)
        self.conn.commit()

    def write(self, vecs, texts=None):
        import sqlite3
        t0 = time.perf_counter()
        if texts is not None:
            rows = [(int(i), sqlite3.Binary(vecs[i].tobytes()), texts[i]) for i in range(len(vecs))]
            self.conn.executemany("INSERT INTO vec(pk, embedding, text) VALUES (?, ?, ?)", rows)
        else:
            rows = [(int(i), sqlite3.Binary(vecs[i].tobytes())) for i in range(len(vecs))]
            self.conn.executemany("INSERT INTO vec(pk, embedding) VALUES (?, ?)", rows)
        self.conn.commit()
        return time.perf_counter() - t0

    def build_index(self):
        return 0.0

    def query(self, q, ef):
        import sqlite3
        t0 = time.perf_counter()
        cur = self.conn.execute(
            "SELECT pk, distance FROM vec WHERE embedding MATCH ? AND k = 10",
            [sqlite3.Binary(q.tobytes())])
        rows = cur.fetchall()
        return {r[0] for r in rows}, (time.perf_counter() - t0) * 1000

    def storage(self):
        return dir_bytes(self.path)

    def teardown(self):
        try:
            self.conn.close()
            Path(self.path).unlink(missing_ok=True)
        except Exception:
            pass


class ChromaAdapter:
    """ChromaDB 嵌入式向量库。HNSW 索引，不支持运行时调 ef。"""

    name = "chroma"
    version = "latest"

    SPECS = {"HNSW": dict()}

    def __init__(self, dim, ns="idxbench", with_text=False):
        self.dim = dim
        self.ns = ns
        self.with_text = with_text
        self.path = CHROMA_DATA / ("chroma_%s" % ns)

    def setup(self, spec):
        import chromadb
        self.spec = spec
        shutil.rmtree(self.path, ignore_errors=True)
        self.client = chromadb.PersistentClient(path=str(self.path))
        self.collection = self.client.create_collection(
            name="vec",
            metadata={"hnsw:space": "cosine", "hnsw:M": 16, "hnsw:construction_ef": 100}
        )

    def write(self, vecs, texts=None):
        t0 = time.perf_counter()
        batch = 5000
        for i in range(0, len(vecs), batch):
            chunk = vecs[i:i + batch]
            self.collection.add(
                embeddings=chunk.tolist(),
                documents=list(texts[i:i + batch]) if texts is not None else None,
                ids=[str(j) for j in range(i, i + len(chunk))]
            )
        return time.perf_counter() - t0

    def build_index(self):
        return 0.0

    def query(self, q, ef):
        t0 = time.perf_counter()
        results = self.collection.query(query_embeddings=[q.tolist()], n_results=10)
        return {int(r) for r in results["ids"][0]}, (time.perf_counter() - t0) * 1000

    def storage(self):
        return dir_bytes(self.path)

    def teardown(self):
        shutil.rmtree(self.path, ignore_errors=True)


REGISTRY = {
    "milvus": MilvusAdapter,
    "qdrant": QdrantAdapter,
    "surreal": SurrealAdapter,
    "lance": LanceAdapter,
    "sqlite": SqliteVecAdapter,
    "chroma": ChromaAdapter,
}

# 默认只跑能跨库对比的索引机制，外加每个库的独有索引（单列标注）
DEFAULT_INDEXES = None  # None = 每个库跑它全部 SPECS


def measure_text_storage(cls, vecs, texts, dim, spec="FLAT"):
    """A/B 测文本存储成本：同一索引（默认 FLAT）下，向量+pk 与 向量+pk+文本
    两轮各自独立建表，用落盘字节差得出文本增量。

    各库的 setup/teardown 已处理各自的落盘怪癖（Milvus 不回收 minio 用快照差分、
    SurrealDB 每轮独立目录、Qdrant 异步删表等），这里复用 run_one 的同一套生命周期。
    spec 在该库不存在时回退到它的第一个 SPECS。
    """
    if spec not in cls.SPECS:
        spec = list(cls.SPECS.keys())[0]
    print("\n%s [%s 文本存储 A/B]" % ("-" * 18, cls.name), flush=True)
    result = {"db": cls.name, "index": spec, "text_raw_bytes": sum(len(t.encode()) for t in texts)}

    def _run(with_text):
        a = cls(dim=dim, ns="textbench", with_text=with_text)
        try:
            a.setup(spec)
            if hasattr(a, "snapshot_base"):
                a.snapshot_base()
            base = a.storage()
            a.write(vecs, texts=texts if with_text else None)
            after = a.storage()
            return {"empty": base, "after": after, "data_net": after - base}
        finally:
            try:
                a.teardown()
            except Exception:
                pass

    v = _run(False)
    t = _run(True)
    result["vector_only"] = v
    result["vector_plus_text"] = t
    result["text_net_bytes"] = t["data_net"] - v["data_net"]
    result["text_amplification"] = (result["text_net_bytes"] /
                                   max(1, result["text_raw_bytes"]))
    print("  vector+pk: empty=%.1fMB after=%.1fMB data=%.1fMB"
          % (v["empty"] / 1e6, v["after"] / 1e6, v["data_net"] / 1e6), flush=True)
    print("  +text    : after=%.1fMB data=%.1fMB  文本增量=%.1fMB (原始 %.1fMB, 比值 %.2fx)"
          % (t["after"] / 1e6, t["data_net"] / 1e6,
             result["text_net_bytes"] / 1e6, result["text_raw_bytes"] / 1e6,
             result["text_amplification"]), flush=True)
    return result


# ---------------------------------------------------------------- 跑分

def run_one(cls, spec, vecs, queries, gt, dim, efs, rounds):
    print("\n%s  [%s]" % ("-" * 24, spec), flush=True)
    a = cls(dim=dim)
    try:
        a.setup(spec)
        # Milvus 的 minio 不回收已删除 collection 的空间，需要在 setup 后先拍一张
        # 基线快照，后续三次 storage() 都只统计相对这张基线新增的对象
        if hasattr(a, "snapshot_base"):
            a.snapshot_base()
        base = a.storage()
        tw = a.write(vecs)
        after_write = a.storage()
        ti = a.build_index()
        after_index = a.storage()
        print("  insert %d: %.1fs (%.0f vec/s)   index build: %.2fs"
              % (len(vecs), tw, len(vecs) / tw, ti), flush=True)

        res = {
            "db": a.name,
            "version": a.version,
            "index": spec,
            "n": len(vecs),
            "dim": dim,
            "write_s": tw,
            "write_vec_per_s": len(vecs) / tw,
            "index_build_s": ti,
            "storage": {
                "empty_bytes": base,
                "after_write_bytes": after_write,
                "after_index_bytes": after_index,
                # 原始向量字段的理论字节数，用于从总量里减出索引净开销
                "raw_vector_bytes": len(vecs) * dim * 4,
                "index_net_bytes": after_index - after_write,
                "data_net_bytes": after_write - base,
                "total_net_bytes": after_index - base,
                # 落盘 / 原始向量 的比值，>1 说明索引膨胀，<1 说明有压缩。
                # 注意 Qdrant 的空 collection 就预分配了 HNSW 图结构（实测 11293 条
                # 384 维时 empty=352MB），total_net 会把预分配算进去而失真，
                # 跨库对比应以 index_net 为准。
                "amplification": (after_index - base) / max(1, len(vecs) * dim * 4),
            },
        }
        for ef in efs:
            lat, rc, first = [], [], None
            for r in range(rounds):
                for i, q in enumerate(queries):
                    pks, dt = a.query(q, ef)
                    if r == 0 and i == 0:
                        first = dt
                    elif r > 0:
                        lat.append(dt)
                    rc.append(len(pks & set(gt[i].tolist())) / len(gt[i]))
            res["ef%d" % ef] = {
                "first_call_ms": first,
                "p50_ms": float(np.median(lat)),
                "p95_ms": float(np.percentile(lat, 95)),
                "recall": float(np.mean(rc)),
            }
            print("    ef=%-5d p50=%8.1fms p95=%8.1fms recall=%.4f"
                  % (ef, np.median(lat), np.percentile(lat, 95), np.mean(rc)), flush=True)
        st = res["storage"]
        print("    storage: empty=%.1fMB write=%.1fMB index=%.1fMB  "
              "(data=%+.1fMB index=%+.1fMB, ratio=%.2fx)"
              % (st["empty_bytes"] / 1e6, st["after_write_bytes"] / 1e6,
                 st["after_index_bytes"] / 1e6, st["data_net_bytes"] / 1e6,
                 st["index_net_bytes"] / 1e6, st["amplification"]), flush=True)
        return res
    except Exception as e:
        print("    FAILED: %s: %s" % (type(e).__name__, str(e)[:160]), flush=True)
        return {"db": cls.__name__, "index": spec, "error": "%s: %s" % (type(e).__name__, e)}
    finally:
        try:
            a.teardown()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None,
                    help="只跑指定库: milvus qdrant surreal lance sqlite")
    ap.add_argument("--index", nargs="*", default=None,
                    help="只跑指定索引类型（跨库同名索引才会都跑）")
    ap.add_argument("--efs", nargs="*", type=int, default=[64, 128, 256])
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--synthetic", action="store_true",
                    help="用合成随机向量（默认用 20 Newsgroups 真实 embedding）")
    ap.add_argument("--text-storage", action="store_true",
                    help="测文本存储成本：每库跑 向量+pk 与 向量+pk+文本 的 A/B，只报存储")
    args = ap.parse_args()

    keys = args.only or list(REGISTRY.keys())
    for k in keys:
        if k not in REGISTRY:
            raise SystemExit("未知库 '%s'，可选: %s" % (k, ", ".join(REGISTRY)))

    if args.text_storage:
        vecs, queries, dim = load_real()
        tp = ROOT / "data" / "20news_texts.npz"
        if not tp.exists():
            raise SystemExit("缺 data/20news_texts.npz，先从 20news-bydate 提取正文后保存")
        zt = np.load(tp, allow_pickle=True)
        texts = zt["train_texts"]
        print("text-storage A/B: %d docs, text raw %.1f MB"
              % (len(texts), sum(len(t.encode()) for t in texts) / 1e6), flush=True)
        tresults = []
        for k in keys:
            tresults.append(measure_text_storage(REGISTRY[k], vecs, texts, dim))
        tresults[0]["env"] = {"cpu": platform.processor(),
                               "python": platform.python_version(),
                               "os": platform.platform()}
        RESULTS.mkdir(exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        out = RESULTS / ("textbench-%s-%s.json" % ("real-20news", stamp))
        out.write_text(json.dumps(tresults, indent=2, ensure_ascii=False), encoding="utf-8")
        print("\nsaved: %s" % out)
        return

    if args.synthetic:
        vecs = np.random.default_rng(0).random((30000, 768), dtype="float32")
        queries = np.random.default_rng(99).random((50, 768), dtype="float32")
        dim = 768
        tag = "synthetic-30k-d768"
    else:
        vecs, queries, dim = load_real()
        tag = "real-20news"
    gt = ground_truth(vecs, queries, 10)
    print("dataset: %d x %d, queries: %d, k=10, mode=%s"
          % (len(vecs), dim, len(queries), tag), flush=True)
    print("raw vector bytes: %.1f MB" % (len(vecs) * dim * 4 / 1e6), flush=True)

    results = []
    total = sum(len(args.index or REGISTRY[k].SPECS.keys()) for k in keys)
    done = 0
    for k in keys:
        cls = REGISTRY[k]
        specs = args.index or list(cls.SPECS.keys())
        # 指定 --index 时跳过该库不支持的规格
        specs = [s for s in specs if s in cls.SPECS]
        if not specs:
            print("\n%s: 无匹配索引规格，跳过" % cls.name, flush=True)
            continue
        for spec in specs:
            done += 1
            print("\n[%d/%d] %s" % (done, total, cls.name), flush=True)
            r = run_one(cls, spec, vecs, queries, gt, dim, args.efs, args.rounds)
            r["mode"] = tag
            results.append(r)

    env = {"cpu": platform.processor(), "python": platform.python_version(),
           "os": platform.platform()}
    for r in results:
        if "error" not in r:
            r["env"] = env

    RESULTS.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = RESULTS / ("indexbench-%s-%s.json" % (tag, stamp))
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nsaved: %s" % out)


if __name__ == "__main__":
    main()
