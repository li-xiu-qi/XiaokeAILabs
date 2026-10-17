"""连接自检：确认两个库的客户端都能连通并执行基本向量操作。

用法: ./.venv/Scripts/python.exe scripts/smoke_test.py
"""
import sys
import time

import numpy as np


def check_surreal():
    from surrealdb import Surreal

    print("=== SurrealDB ===")
    t0 = time.perf_counter()
    with Surreal("ws://127.0.0.1:8000/rpc") as db:
        db.signin({"username": "root", "password": "root"})
        db.use("bench", "main")

        # 建表 + 向量索引
        db.query("REMOVE TABLE IF EXISTS smoke;")
        db.query("""
            DEFINE TABLE smoke SCHEMALESS;
            DEFINE FIELD emb ON smoke TYPE array<float>;
            DEFINE INDEX emb_idx ON smoke FIELDS emb HNSW DIMENSION 4 DIST EUCLIDEAN;
        """)

        vecs = np.random.rand(20, 4).astype("float32")
        rows = [
            {"pk": i, "emb": [float(x) for x in vecs[i]]}
            for i in range(20)
        ]
        t1 = time.perf_counter()
        stmts = "BEGIN;"
        for r in rows:
            stmts += f"CREATE smoke SET pk = {r['pk']}, emb = {r['emb']};"
        stmts += "COMMIT;"
        db.query(stmts)
        t_write = time.perf_counter() - t1

        q = [float(x) for x in np.random.rand(4).astype("float32")]
        t2 = time.perf_counter()
        res = db.query(
            "SELECT pk, emb, vector::distance::euclidean(emb, $q) AS d "
            "FROM smoke ORDER BY d LIMIT 5;",
            {"q": q},
        )
        t_query = time.perf_counter() - t2

        rows = res if isinstance(res, list) else res.get("result", [])
        print(f"connect+setup : {time.perf_counter() - t0:.3f}s")
        print(f"insert 20 rows: {t_write:.3f}s")
        print(f"knn k=5       : {t_query * 1000:.1f}ms")
        print(f"returned      : {len(rows) if rows else 0} rows")

        db.query("REMOVE TABLE IF EXISTS smoke;")
        return True


def check_milvus():
    from pymilvus import MilvusClient
    from pymilvus.milvus_client.index import IndexParams

    print("=== Milvus ===")
    t0 = time.perf_counter()
    client = MilvusClient(uri="http://127.0.0.1:19530")

    name = "smoke"
    if client.has_collection(name):
        client.drop_collection(name)

    index_params = IndexParams()
    index_params.add_index(
        field_name="vector",
        index_type="HNSW",
        metric_type="L2",
        params={"M": 16, "efConstruction": 200},
    )

    client.create_collection(
        collection_name=name,
        dimension=4,
        metric_type="L2",
        auto_id=True,
        index_params=index_params,
    )

    rng = np.random.default_rng(0)
    vecs = rng.random((20, 4), dtype="float32")
    rows = [{"pk": i, "vector": vecs[i].tolist()} for i in range(20)]

    t1 = time.perf_counter()
    client.insert(collection_name=name, data=rows)
    t_write = time.perf_counter() - t1

    client.flush(collection_name=name)
    client.load_collection(collection_name=name)

    q = rng.random((1, 4), dtype="float32")
    t2 = time.perf_counter()
    res = client.search(
        collection_name=name,
        data=q.tolist(),
        limit=5,
        search_params={"metric_type": "L2", "params": {"ef": 64}},
        output_fields=["pk"],
    )
    t_query = time.perf_counter() - t2

    hits = res[0] if res else []
    print(f"connect+setup : {time.perf_counter() - t0:.3f}s")
    print(f"insert 20 rows: {t_write:.3f}s")
    print(f"knn k=5       : {t_query * 1000:.1f}ms")
    print(f"returned      : {len(hits)} rows")

    client.drop_collection(name)
    return True


if __name__ == "__main__":
    ok_s = ok_m = False
    try:
        ok_s = check_surreal()
    except Exception as e:
        print(f"SurrealDB FAILED: {type(e).__name__}: {e}")
    try:
        ok_m = check_milvus()
    except Exception as e:
        print(f"Milvus FAILED: {type(e).__name__}: {e}")
    print(f"\nresult: surreal={'OK' if ok_s else 'FAIL'} milvus={'OK' if ok_m else 'FAIL'}")
    sys.exit(0 if (ok_s and ok_m) else 1)
