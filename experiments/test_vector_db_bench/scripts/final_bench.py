"""最终对比：同数据集、同查询集、同 ef，SurrealDB vs Milvus。"""
import sys, time
import numpy as np
from surrealdb import Surreal
from pymilvus import MilvusClient
from pymilvus.milvus_client.index import IndexParams

NS, DIM, N = "final_z", 768, 30000
rng = np.random.default_rng(0)
V = rng.random((N, DIM), dtype="float32")
qrng = np.random.default_rng(99)
QS = qrng.random((50, DIM), dtype="float32")
GT = []
for q in QS:
    GT.append(set(np.argsort(np.linalg.norm(V - q, axis=1))[:10].tolist()))


def arr(v):
    return "[" + ",".join("%.6f" % x for x in v) + "]"


# ---- SurrealDB ----
print("=== SurrealDB 3.2.4 ===")
db = Surreal("ws://127.0.0.1:8000/rpc")
db.signin({"username": "root", "password": "root"})
db.use(NS, "main")
db.query("REMOVE TABLE IF EXISTS vec;")
db.query("DEFINE TABLE vec SCHEMALESS; DEFINE FIELD pk ON vec TYPE int; "
         "DEFINE FIELD emb ON vec TYPE array<float>; "
         "DEFINE INDEX emb_idx ON vec FIELDS emb "
         "HNSW DIMENSION %d DIST EUCLIDEAN EFC 200 M 16;" % DIM)
t0 = time.perf_counter()
CH = 400
for s in range(0, N, CH):
    e = min(s + CH, N)
    parts = ["{pk: %d, emb: %s}" % (i, arr(V[i])) for i in range(s, e)]
    db.query("INSERT INTO vec [%s];" % ",".join(parts))
tw_s = time.perf_counter() - t0
print(f"insert 30k: {tw_s:.1f}s ({N/tw_s:.0f} vec/s)")

for ef in [64, 128, 256]:
    lat, rc = [], []
    for i, q in enumerate(QS):
        t = time.perf_counter()
        res = db.query("SELECT pk FROM vec WHERE emb <|10,%d|> %s;" % (ef, arr(q)))
        lat.append((time.perf_counter() - t) * 1000)
        rows = res[0] if isinstance(res, list) and res and isinstance(res[0], list) else res
        pks = {r["pk"] for r in rows}
        rc.append(len(pks & GT[i]) / 10)
    print(f"  ef={ef:<4} p50={np.median(lat):7.0f}ms recall={np.mean(rc):.4f}")
db.close()

# ---- Milvus ----
print("\n=== Milvus v3.0.0 (HNSW) ===")
c = MilvusClient(uri="http://127.0.0.1:19530")
if c.has_collection("mfin"):
    c.drop_collection("mfin")
while c.has_collection("mfin"):
    time.sleep(0.3)
from pymilvus import CollectionSchema, FieldSchema, DataType
schema = CollectionSchema(fields=[
    FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=DIM),
])
c.create_collection(collection_name="mfin", schema=schema)
t0 = time.perf_counter()
for s in range(0, N, 5000):
    e = min(s + 5000, N)
    c.insert(collection_name="mfin",
             data=[{"pk": int(i), "vector": V[i].tolist()} for i in range(s, e)])
c.flush(collection_name="mfin")
tw_m = time.perf_counter() - t0
ip = IndexParams()
ip.add_index(field_name="vector", index_type="HNSW", metric_type="L2",
             params={"M": 16, "efConstruction": 200})
c.create_index(collection_name="mfin", index_params=ip)
c.load_collection(collection_name="mfin")
while True:
    d = c.describe_index(collection_name="mfin", index_name="vector")
    if d.get("state") == "Finished" and d.get("pending_index_rows", 0) == 0:
        break
    time.sleep(1)
print(f"insert 30k: {tw_m:.1f}s ({N/tw_m:.0f} vec/s)")

for ef in [64, 128, 256]:
    lat, rc = [], []
    for i, q in enumerate(QS):
        t = time.perf_counter()
        r = c.search(collection_name="mfin", data=[q.tolist()], limit=10,
                     search_params={"metric_type": "L2", "params": {"ef": ef}},
                     output_fields=["pk"])
        lat.append((time.perf_counter() - t) * 1000)
        pks = {h["entity"]["pk"] for h in r[0]}
        rc.append(len(pks & GT[i]) / 10)
    print(f"  ef={ef:<4} p50={np.median(lat):7.0f}ms recall={np.mean(rc):.4f}")
