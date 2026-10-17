"""索引能力探测：逐个库试建各种索引，确认 3.0 版本实际支持哪些。

用小数据集（2000 x 128）快速试，只验证「建得起来 + 查得通」，不测性能。

用法: ./.venv/Scripts/python.exe scripts/probe_index.py
"""
import json
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np

ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent

MILVUS_URI = "http://127.0.0.1:19530"
QDRANT_URI = "http://127.0.0.1:6333"
SURREAL_URI = "ws://127.0.0.1:8000/rpc"

DIM = 128
N = 2000


def data():
    return np.random.default_rng(0).random((N, DIM), dtype="float32")


def gt(vecs, q, k=10):
    vv = np.sum(vecs ** 2, 1)
    qq = np.sum(q ** 2, 1)
    d = qq[:, None] + vv[None, :] - 2 * (q @ vecs.T)
    return np.argsort(d, 1)[:, :k]


VECS = data()
QUERIES = VECS[:20].copy()
G = gt(VECS, QUERIES)


def recall(pks, i):
    return len(set(pks) & set(G[i].tolist())) / 10.0


# ---------------------------------------------------------------- Milvus

def probe_milvus():
    from pymilvus import MilvusClient, CollectionSchema, FieldSchema, DataType
    from pymilvus.milvus_client.index import IndexParams
    print("\n=== Milvus v3.0.0 ===", flush=True)
    c = MilvusClient(uri=MILVUS_URI)

    def fresh(coll):
        """走 schema 路径建表。MilvusClient 的快捷建表会默认挂 AUTOINDEX，
        之后再 create_index 会撞 "at most one distinct index is allowed per field"。"""
        if c.has_collection(coll):
            c.drop_collection(coll)
        while c.has_collection(coll):
            time.sleep(0.2)
        schema = CollectionSchema(fields=[
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=DIM),
        ])
        c.create_collection(collection_name=coll, schema=schema)
    specs = [
        ("FLAT", {}),
        ("IVF_FLAT", {"nlist": 128}),
        ("IVF_SQ8", {"nlist": 128}),
        ("IVF_PQ", {"nlist": 128, "m": 8, "nbits": 8}),
        ("HNSW", {"M": 16, "efConstruction": 200}),
        ("DISKANN", {}),
        ("SCANN", {"nlist": 128, "with_raw_data": True}),
        ("GPU_IVF_FLAT", {"nlist": 128}),
        ("GPU_CAGRA", {}),
        ("AUTOINDEX", {}),
    ]
    for name, params in specs:
        coll = "probe_" + name.lower()
        try:
            fresh(coll)
            c.insert(collection_name=coll,
                     data=[{"pk": i, "vector": VECS[i].tolist()} for i in range(N)])
            c.flush(collection_name=coll)
            t0 = time.perf_counter()
            ip = IndexParams()
            ip.add_index(field_name="vector", index_type=name, metric_type="L2",
                         params=params if params else None)
            c.create_index(collection_name=coll, index_params=ip)
            c.load_collection(collection_name=coll)
            # 等索引就绪
            deadline = time.time() + 120
            state = "?"
            while time.time() < deadline:
                d = c.describe_index(collection_name=coll, index_name="vector")
                state = d.get("state")
                if state == "Finished" and d.get("pending_index_rows", 0) == 0:
                    break
                time.sleep(0.5)
            build_s = time.perf_counter() - t0
            got = c.describe_index(collection_name=coll, index_name="vector")
            real_type = got.get("index_type") or got.get("params", {}).get("index_type")
            # 查询侧参数
            qp = {"metric_type": "L2", "params": {}}
            if name.startswith("IVF"):
                qp["params"]["nprobe"] = 16
            elif name == "HNSW":
                qp["params"]["ef"] = 64
            try:
                r = c.search(collection_name=coll, data=[QUERIES[0].tolist()], limit=10,
                             search_params=qp, output_fields=["pk"])
                ok = len(r[0]) == 10
            except Exception as e:
                ok = f"query failed: {type(e).__name__}"
            print(f"  {name:<14} build={build_s:7.2f}s state={state:<10} "
                  f"type={real_type} query={ok}", flush=True)
        except Exception as e:
            print(f"  {name:<14} FAILED {type(e).__name__}: {str(e)[:120]}", flush=True)
        finally:
            try:
                if c.has_collection(coll):
                    c.drop_collection(coll)
            except Exception:
                pass


# ---------------------------------------------------------------- Qdrant

def probe_qdrant():
    from qdrant_client import QdrantClient
    from qdrant_client.models import (Distance, VectorParams, HnswConfigDiff,
                                      ScalarQuantization, ScalarQuantizationConfig,
                                      ScalarType, ProductQuantization,
                                      ProductQuantizationConfig, BinaryQuantization,
                                      BinaryQuantizationConfig, CompressionRatio)
    print("\n=== Qdrant ===", flush=True)
    c = QdrantClient(url=QDRANT_URI, timeout=60)
    specs = [
        ("HNSW m=16", VectorParams(size=DIM, distance=Distance.EUCLID,
                                   hnsw_config=HnswConfigDiff(m=16, ef_construct=200)), None),
        ("HNSW m=32", VectorParams(size=DIM, distance=Distance.EUCLID,
                                   hnsw_config=HnswConfigDiff(m=32, ef_construct=256)), None),
        ("FLAT (m=0)", VectorParams(size=DIM, distance=Distance.EUCLID,
                                    hnsw_config=HnswConfigDiff(m=0)), None),
        ("HNSW+ScalarI8", VectorParams(size=DIM, distance=Distance.EUCLID,
                                       hnsw_config=HnswConfigDiff(m=16, ef_construct=200)),
         ScalarQuantization(scalar=ScalarQuantizationConfig(type=ScalarType.INT8,
                                                            quantile=0.99, always_ram=True))),
        ("HNSW+PQ", VectorParams(size=DIM, distance=Distance.EUCLID,
                                 hnsw_config=HnswConfigDiff(m=16, ef_construct=200)),
         ProductQuantization(product=ProductQuantizationConfig(
             compression=CompressionRatio.X16, always_ram=True))),
        ("HNSW+Binary", VectorParams(size=DIM, distance=Distance.EUCLID,
                                     hnsw_config=HnswConfigDiff(m=16, ef_construct=200)),
         BinaryQuantization(binary=BinaryQuantizationConfig(always_ram=True))),
    ]
    from qdrant_client.models import PointStruct
    for name, vp, quant in specs:
        coll = "probe"
        try:
            if c.collection_exists(coll):
                c.delete_collection(coll)
            t0 = time.perf_counter()
            kw = dict(collection_name=coll, vectors_config=vp)
            if quant is not None:
                kw["quantization_config"] = quant
            c.create_collection(**kw)
            for s in range(0, N, 500):
                e = min(s + 500, N)
                c.upsert(collection_name=coll, points=[
                    PointStruct(id=i, vector=VECS[i].tolist(), payload={"pk": i})
                    for i in range(s, e)])
            build_s = time.perf_counter() - t0
            info = c.get_collection(coll)
            print(f"  {name:<16} build={build_s:7.2f}s "
                  f"status={info.status} indexed={info.indexed_vectors_count}", flush=True)
            r = c.query_points(collection_name=coll, query=QUERIES[0].tolist(), limit=10,
                               search_params={"hnsw_ef": 64, "exact": quant is None})
            print(f"                     query ok, {len(r.points)} pts", flush=True)
        except Exception as e:
            print(f"  {name:<16} FAILED {type(e).__name__}: {str(e)[:130]}", flush=True)
        finally:
            try:
                if c.collection_exists(coll):
                    c.delete_collection(coll)
            except Exception:
                pass


# ---------------------------------------------------------------- SurrealDB

def probe_surreal():
    from surrealdb import Surreal
    print("\n=== SurrealDB 3.2.4 ===", flush=True)
    db = Surreal(SURREAL_URI)
    db.signin({"username": "root", "password": "root"})
    specs = [
        ("HNSW M=16 EFC=200",
         "DEFINE INDEX idx ON probe FIELDS emb HNSW DIMENSION %d DIST EUCLIDEAN EFC 200 M 16;" % DIM),
        ("HNSW M=32 EFC=128",
         "DEFINE INDEX idx ON probe FIELDS emb HNSW DIMENSION %d DIST EUCLIDEAN EFC 128 M 32;" % DIM),
        ("MTREE",
         "DEFINE INDEX idx ON probe FIELDS emb MTREE DIMENSION %d DIST EUCLIDEAN;" % DIM),
        ("MTREE TYPE f64",
         "DEFINE INDEX idx ON probe FIELDS emb MTREE DIMENSION %d DIST EUCLIDEAN TYPE F64;" % DIM),
    ]
    for name, ddl in specs:
        try:
            db.use("probe", "main")
            db.query("REMOVE TABLE IF EXISTS probe;")
            db.query("DEFINE TABLE probe SCHEMALESS; DEFINE FIELD pk ON probe TYPE int; "
                     "DEFINE FIELD emb ON probe TYPE array<float>;")
            for s in range(0, N, 400):
                e = min(s + 400, N)
                parts = ["{pk: %d, emb: [%s]}" % (i, ",".join("%.6f" % x for x in VECS[i]))
                         for i in range(s, e)]
                db.query("INSERT INTO probe [%s];" % ",".join(parts))
            t0 = time.perf_counter()
            db.query(ddl)
            build_s = time.perf_counter() - t0
            res = db.query("SELECT pk FROM probe WHERE emb <|10,64|> [%s];"
                           % ",".join("%.6f" % x for x in QUERIES[0]))
            rows = res[0] if isinstance(res, list) and res and isinstance(res[0], list) else res
            print(f"  {name:<22} build={build_s:7.2f}s rows={len(rows) if rows else 0}",
                  flush=True)
        except Exception as e:
            print(f"  {name:<22} FAILED {type(e).__name__}: {str(e)[:130]}", flush=True)
    try:
        db.use("probe", "main")
        db.query("REMOVE TABLE IF EXISTS probe;")
    except Exception:
        pass
    db.close()


# ---------------------------------------------------------------- LanceDB

def probe_lance():
    import lancedb
    import shutil
    print("\n=== LanceDB 0.38.0 ===", flush=True)
    specs = [
        ("IVF_PQ part=128 sub=64", dict(num_partitions=128, num_sub_vectors=64)),
        ("IVF_PQ part=256 sub=32", dict(num_partitions=256, num_sub_vectors=32)),
        ("IVF_PQ part=64 sub=16", dict(num_partitions=64, num_sub_vectors=16)),
    ]
    for name, kw in specs:
        p = ROOT / "data" / ("lance_probe_idx")
        try:
            shutil.rmtree(p, ignore_errors=True)
            db = lancedb.connect(str(p))
            tbl = db.create_table(
                "vec",
                data=[{"pk": int(i), "vector": VECS[i].tolist()} for i in range(N)])
            t0 = time.perf_counter()
            tbl.create_index(metric="l2", **kw)
            build_s = time.perf_counter() - t0
            r = tbl.search(QUERIES[0].tolist()).limit(10).nprobes(16).to_list()
            print(f"  {name:<26} build={build_s:7.2f}s rows={len(r)}", flush=True)
        except Exception as e:
            print(f"  {name:<26} FAILED {type(e).__name__}: {str(e)[:130]}", flush=True)
    # 是否支持 IVF_FLAT / HNSW
    for name, kw in [("IVF_FLAT (sub_vectors=0)", dict(num_partitions=64, num_sub_vectors=0)),
                     ("HNSW", dict(index_type="IVF_PQ"))]:
        try:
            p = ROOT / "data" / ("lance_probe_idx2")
            shutil.rmtree(p, ignore_errors=True)
            db = lancedb.connect(str(p))
            tbl = db.create_table(
                "vec", data=[{"pk": int(i), "vector": VECS[i].tolist()} for i in range(N)])
            t0 = time.perf_counter()
            if name.startswith("IVF_FLAT"):
                tbl.create_index(metric="l2", **kw)
            else:
                tbl.create_index(metric="l2", num_partitions=64, num_sub_vectors=64,
                                 index_type="IVF_HNSW_SQ")
            print(f"  {name:<26} build={time.perf_counter() - t0:7.2f}s OK", flush=True)
        except Exception as e:
            print(f"  {name:<26} FAILED {type(e).__name__}: {str(e)[:130]}", flush=True)


def probe_sqlite_vec():
    import sqlite3
    import sqlite_vec
    print("\n=== sqlite-vec 0.1.9 ===", flush=True)
    print("  vec0 虚拟表，无 ANN 索引，只支持暴力扫描 + k 参数", flush=True)


if __name__ == "__main__":
    probe_milvus()
    probe_qdrant()
    probe_surreal()
    probe_lance()
    probe_sqlite_vec()
    print("\ndone")
