"""多向量库统一基准：同一份数据、同一查询集、同 ef，跑 SurrealDB / Milvus /
Qdrant / LanceDB / sqlite-vec。

每个库实现一个 adapter，暴露相同的四个方法：
    setup()        建表/建索引
    write(vecs)    批量写入，返回耗时秒
    query(q, ef)   单条 KNN，返回 (命中 pk 集合, 耗时 ms)
    teardown()     清理

用法:
  ./.venv/Scripts/python.exe scripts/multi_bench.py --n 30000 --dim 768
  ./.venv/Scripts/python.exe scripts/multi_bench.py --only qdrant lancedb
"""
import argparse
import importlib
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

MILVUS_URI = "http://127.0.0.1:19530"
QDRANT_URI = "http://127.0.0.1:6333"
SURREAL_URI = "ws://127.0.0.1:8000/rpc"


# ---------------------------------------------------------------- 数据集

def make_dataset(n, dim, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((n, dim), dtype="float32")


def make_queries(nq, dim, seed=99):
    return np.random.default_rng(seed).random((nq, dim), dtype="float32")


def ground_truth(vecs, queries, k):
    """暴力精确 top-k。用 |a-b|^2 = |a|^2+|b|^2-2ab 避免大中间张量。"""
    vv = np.sum(vecs**2, axis=1)
    qq = np.sum(queries**2, axis=1)
    gt = []
    for i in range(0, len(queries), 25):
        q = queries[i:i + 25]
        d = qq[i:i + 25, None] + vv[None, :] - 2.0 * (q @ vecs.T)
        gt.append(np.argsort(d, axis=1)[:, :k])
    return np.concatenate(gt, axis=0)


def load_real(n_queries=50, seed=0):
    """加载 20 Newsgroups 真实 embedding：train 入库，test 分层采样做查询。"""
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
    return vecs, test_emb[idx], int(vecs.shape[1])


# ---------------------------------------------------------------- adapters

class SurrealAdapter:
    name = "SurrealDB"
    version = "3.2.4"

    def __init__(self, dim, ns="multi"):
        self.dim = dim
        self.ns = ns

    def setup(self):
        from surrealdb import Surreal
        self.db = Surreal(SURREAL_URI)
        self.db.signin({"username": "root", "password": "root"})
        self.db.use(self.ns, "main")
        self.db.query("REMOVE TABLE IF EXISTS vec;")
        self.db.query(
            "DEFINE TABLE vec SCHEMALESS; DEFINE FIELD pk ON vec TYPE int; "
            "DEFINE FIELD emb ON vec TYPE array<float>; "
            "DEFINE INDEX emb_idx ON vec FIELDS emb "
            "HNSW DIMENSION %d DIST EUCLIDEAN EFC 200 M 16;" % self.dim)

    def write(self, vecs):
        def arr(v):
            return "[" + ",".join("%.6f" % x for x in v) + "]"
        t0 = time.perf_counter()
        CH = 400  # 更大批次会触发 HTTP 413 / WS 断连
        for s in range(0, len(vecs), CH):
            e = min(s + CH, len(vecs))
            parts = ["{pk: %d, emb: %s}" % (i, arr(vecs[i])) for i in range(s, e)]
            self.db.query("INSERT INTO vec [%s];" % ",".join(parts))
        return time.perf_counter() - t0

    def query(self, q, ef):
        def arr(v):
            return "[" + ",".join("%.6f" % x for x in v) + "]"
        t0 = time.perf_counter()
        res = self.db.query("SELECT pk FROM vec WHERE emb <|10,%d|> %s;" % (ef, arr(q)))
        dt = (time.perf_counter() - t0) * 1000
        rows = res[0] if isinstance(res, list) and res and isinstance(res[0], list) else res
        return {r["pk"] for r in rows}, dt

    def teardown(self):
        try:
            self.db.query("REMOVE TABLE IF EXISTS vec;")
            self.db.close()
        except Exception:
            pass


class MilvusAdapter:
    name = "Milvus"
    version = "v3.0.0"

    def __init__(self, dim, coll="multi"):
        self.dim = dim
        self.coll = coll

    def setup(self):
        from pymilvus import MilvusClient, CollectionSchema, FieldSchema, DataType
        self.c = MilvusClient(uri=MILVUS_URI)
        if self.c.has_collection(self.coll):
            self.c.drop_collection(self.coll)
        while self.c.has_collection(self.coll):
            time.sleep(0.3)
        # 走 schema 路径：create_collection 的快捷路径会强制建 AUTOINDEX，
        # 而 AUTOINDEX 不吃查询侧 ef
        schema = CollectionSchema(fields=[
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
        ])
        self.c.create_collection(collection_name=self.coll, schema=schema)

    def build_index(self):
        """索引必须在数据写入之后再建。

        在 setup() 里先建索引会导致后续写入的数据 pending 永不归零
        （describe_index 停在 InProgress, pending=全部行数），查询打在半成品
        索引上，实测召回率从 0.53 崩到 0.23，且 ef 变化无响应。
        """
        from pymilvus.milvus_client.index import IndexParams
        ip = IndexParams()
        ip.add_index(field_name="vector", index_type="HNSW", metric_type="L2",
                     params={"M": 16, "efConstruction": 200})
        self.c.create_index(collection_name=self.coll, index_params=ip)
        self.c.load_collection(collection_name=self.coll)
        self._wait_index()

    def _wait_index(self, timeout=300):
        deadline = time.time() + timeout
        while time.time() < deadline:
            d = self.c.describe_index(collection_name=self.coll, index_name="vector")
            if d.get("state") == "Finished" and d.get("pending_index_rows", 0) == 0:
                return
            time.sleep(1)
        raise TimeoutError("milvus index not ready")

    def write(self, vecs):
        t0 = time.perf_counter()
        for s in range(0, len(vecs), 5000):
            e = min(s + 5000, len(vecs))
            self.c.insert(collection_name=self.coll,
                          data=[{"pk": int(i), "vector": vecs[i].tolist()}
                                for i in range(s, e)])
        self.c.flush(collection_name=self.coll)
        return time.perf_counter() - t0

    def query(self, q, ef):
        t0 = time.perf_counter()
        r = self.c.search(collection_name=self.coll, data=[q.tolist()], limit=10,
                          search_params={"metric_type": "L2", "params": {"ef": ef}},
                          output_fields=["pk"])
        dt = (time.perf_counter() - t0) * 1000
        return {h["entity"]["pk"] for h in r[0]}, dt

    def teardown(self):
        try:
            if self.c.has_collection(self.coll):
                self.c.drop_collection(self.coll)
        except Exception:
            pass


class QdrantAdapter:
    name = "Qdrant"
    version = "1.19.0"

    def __init__(self, dim, coll="multi"):
        self.dim = dim
        self.coll = coll

    def setup(self):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, HnswConfigDiff
        self.c = QdrantClient(url=QDRANT_URI, timeout=120)
        if self.c.collection_exists(self.coll):
            self.c.delete_collection(self.coll)
        self.c.create_collection(
            collection_name=self.coll,
            vectors_config=VectorParams(size=self.dim, distance=Distance.EUCLID,
                                        hnsw_config=HnswConfigDiff(m=16, ef_construct=200)),
        )

    def write(self, vecs):
        from qdrant_client.models import PointStruct
        t0 = time.perf_counter()
        B = 1000
        for s in range(0, len(vecs), B):
            e = min(s + B, len(vecs))
            self.c.upsert(collection_name=self.coll, points=[
                PointStruct(id=i, vector=vecs[i].tolist(), payload={"pk": i})
                for i in range(s, e)])
        return time.perf_counter() - t0

    def query(self, q, ef):
        t0 = time.perf_counter()
        r = self.c.query_points(collection_name=self.coll, query=q.tolist(),
                                limit=10, search_params={"hnsw_ef": ef, "exact": False})
        dt = (time.perf_counter() - t0) * 1000
        return {p.payload["pk"] for p in r.points}, dt

    def teardown(self):
        try:
            if self.c.collection_exists(self.coll):
                self.c.delete_collection(self.coll)
        except Exception:
            pass


class LanceAdapter:
    name = "LanceDB"
    version = "0.38.0"

    def __init__(self, dim, uri=None):
        self.dim = dim
        self.uri = uri or str(ROOT / "data" / "lance")

    def setup(self):
        import lancedb
        self.tbl_name = "vec"
        try:
            self.db = lancedb.connect(self.uri)
            if self.tbl_name in self.db.table_names():
                self.db.drop_table(self.tbl_name)
        except Exception:
            pass

    def write(self, vecs):
        import pandas as pd
        t0 = time.perf_counter()
        self.tbl = self.db.create_table(
            self.tbl_name,
            data=[{"pk": int(i), "vector": vecs[i].tolist()} for i in range(len(vecs))],
        )
        return time.perf_counter() - t0

    # IVF_PQ 参数由 scripts/probe_lance.py 在真实数据上扫出来：
    #   sub=32/part=256（旧） -> recall 0.61，ef 从 16 到 256 无变化（PQ 量化误差封顶）
    #   sub=64/part=128      -> recall 0.78
    #   sub=96/part=64       -> recall 0.86
    #   sub=64 + refine=10   -> recall 0.98~1.00（PQ 粗排后拿原始向量精排，官方推荐用法）
    # 注意：refine 让 LanceDB 变成「粗排+精排」，与其余四库的纯 HNSW 不同构，
    # 报告里必须单列标注，否则延迟对比不公平。
    LANCE_SUB_VECTORS = 64
    LANCE_PARTITIONS = 128
    LANCE_REFINE = 10

    def build_index(self):
        self.tbl.create_index(metric="l2",
                              num_partitions=self.LANCE_PARTITIONS,
                              num_sub_vectors=self.LANCE_SUB_VECTORS)

    def query(self, q, ef):
        t0 = time.perf_counter()
        r = (self.tbl.search(q.tolist()).limit(10).nprobes(ef)
             .refine_factor(self.LANCE_REFINE).to_list())
        dt = (time.perf_counter() - t0) * 1000
        return {row["pk"] for row in r}, dt

    def teardown(self):
        try:
            if self.tbl_name in self.db.table_names():
                self.db.drop_table(self.tbl_name)
        except Exception:
            pass


class SqliteVecAdapter:
    name = "sqlite-vec"
    version = "0.1.9"

    def __init__(self, dim, path=None):
        self.dim = dim
        self.path = path or str(ROOT / "data" / "sqlite_vec.db")

    def setup(self):
        import sqlite3
        import sqlite_vec
        Path(self.path).unlink(missing_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)
        self.conn.execute("CREATE VIRTUAL TABLE vec USING vec0(pk integer primary key, embedding float[%d])" % self.dim)

    def write(self, vecs):
        import sqlite3
        t0 = time.perf_counter()
        rows = [(int(i), sqlite3.Binary(vecs[i].tobytes())) for i in range(len(vecs))]
        self.conn.executemany("INSERT INTO vec(pk, embedding) VALUES (?, ?)", rows)
        self.conn.commit()
        return time.perf_counter() - t0

    def query(self, q, ef):
        import sqlite3
        t0 = time.perf_counter()
        cur = self.conn.execute(
            "SELECT pk, distance FROM vec WHERE embedding MATCH ? AND k = 10",
            [sqlite3.Binary(q.tobytes())])
        rows = cur.fetchall()
        dt = (time.perf_counter() - t0) * 1000
        return {r[0] for r in rows}, dt

    def teardown(self):
        try:
            self.conn.close()
            Path(self.path).unlink(missing_ok=True)
        except Exception:
            pass


REGISTRY = {
    "surreal": SurrealAdapter,
    "milvus": MilvusAdapter,
    "qdrant": QdrantAdapter,
    "lance": LanceAdapter,
    "sqlite": SqliteVecAdapter,
}


# ---------------------------------------------------------------- 跑分

def run_one(key, cls, vecs, queries, gt, dim, efs, rounds=3):
    print(f"\n{'=' * 60}\n{cls.__name__}\n{'=' * 60}", flush=True)
    a = cls(dim=dim)
    try:
        a.setup()
        tw = a.write(vecs)
        if hasattr(a, "build_index"):
            a.build_index()
        print(f"insert {len(vecs)}: {tw:.1f}s ({len(vecs)/tw:.0f} vec/s)", flush=True)

        result = {
            "db": a.name,
            "version": a.version,
            "n": len(vecs),
            "dim": dim,
            "write_s": tw,
            "write_vec_per_s": len(vecs) / tw,
        }
        for ef in efs:
            lat, rc = [], []
            for _ in range(rounds):
                for i, q in enumerate(queries):
                    pks, dt = a.query(q, ef)
                    lat.append(dt)
                    rc.append(len(pks & set(gt[i].tolist())) / len(gt[i]))
            result[f"ef{ef}"] = {
                "p50_ms": float(np.median(lat)),
                "recall": float(np.mean(rc)),
            }
            print(f"  ef={ef:<4} p50={np.median(lat):7.0f}ms "
                  f"recall={np.mean(rc):.4f}", flush=True)
        return result
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
        return {"db": cls.__name__, "error": str(e)}
    finally:
        try:
            a.teardown()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--real", action="store_true",
                    help="用 20 Newsgroups 真实 embedding（train 入库 / test 采样查询）")
    ap.add_argument("--only", nargs="*", default=None,
                    help="只跑指定库: surreal milvus qdrant lance sqlite")
    args = ap.parse_args()

    keys = args.only or list(REGISTRY.keys())
    for k in keys:
        if k not in REGISTRY:
            raise SystemExit(f"未知库 '{k}'，可选: {', '.join(REGISTRY)}")

    if args.real:
        vecs, queries, dim = load_real()
        tag = "real-20news"
    else:
        vecs = make_dataset(args.n, args.dim)
        queries = make_queries(50, args.dim)
        dim = args.dim
        tag = f"{args.n}-d{args.dim}"
    gt = ground_truth(vecs, queries, 10)
    print(f"dataset: {len(vecs)} x {dim}, queries: {len(queries)}, k=10, mode={tag}", flush=True)

    results = []
    for k in keys:
        cls = REGISTRY[k]
        r = run_one(k, cls, vecs, queries, gt, dim,
                    efs=[64, 128, 256], rounds=args.rounds)
        r["mode"] = tag
        r["env"] = {"cpu": platform.processor(), "python": platform.python_version()}
        results.append(r)

    RESULTS.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = RESULTS / f"multi-{tag}-{stamp}.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
