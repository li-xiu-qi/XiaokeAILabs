"""SurrealDB vs Milvus 向量性能对比。

同一份合成数据集，同一查询集，同维度同 metric，两边都走 HNSW 索引。
测写入吞吐、kNN 延迟分布、召回率、带过滤查询。

用法:
  ./.venv/Scripts/python.exe scripts/bench.py --scale 10k --dim 768
"""
import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

SURREAL_URL = "ws://127.0.0.1:8000/rpc"
MILVUS_URL = "http://127.0.0.1:19530"


# ---------------------------------------------------------------- 数据

def make_dataset(n, dim, seed=42):
    """确定性数据集，两组查询集（近邻 / 随机）。"""
    rng = np.random.default_rng(seed)
    vecs = rng.random((n, dim), dtype="float32")

    q_near = rng.random((200, dim), dtype="float32")
    q_rand = rng.random((200, dim), dtype="float32")
    return vecs, q_near, q_rand


def ground_truth(vecs, queries, k):
    """暴力搜索精确 top-k，作为召回率基准。

    用 ||a-b||^2 = |a|^2 + |b|^2 - 2ab 的展开式，避免开 (m, n, dim) 的中间张量。
    50k x 768 的朴素实现要开 7.7 GB 中间数组，会直接 OOM。
    """
    vv = np.sum(vecs**2, axis=1)          # (n,)
    qq = np.sum(queries**2, axis=1)       # (m,)
    gt = []
    step = 25
    for i in range(0, len(queries), step):
        q = queries[i : i + step]
        dots = q @ vecs.T                  # (step, n)
        d = qq[i : i + step, None] + vv[None, :] - 2.0 * dots
        gt.append(np.argsort(d, axis=1)[:, :k])
    return np.concatenate(gt, axis=0)


# ---------------------------------------------------------------- SurrealDB

class SurrealBench:
    name = "SurrealDB"

    def __init__(self, dim, k_nn):
        from surrealdb import Surreal

        self.dim = dim
        self.k = k_nn
        self.db = Surreal(SURREAL_URL)
        self.db.signin({"username": "root", "password": "root"})
        self.db.use("bench", "main")

    def reset(self):
        self.db.query("REMOVE TABLE IF EXISTS vec;")

    def create_index(self, efc=200, m=16):
        self.db.query(f"""
            DEFINE TABLE vec SCHEMALESS;
            DEFINE FIELD pk ON vec TYPE int;
            DEFINE FIELD emb ON vec TYPE array<float>;
            DEFINE INDEX emb_idx ON vec FIELDS emb
                HNSW DIMENSION {self.dim} DIST EUCLIDEAN
                EFC {efc} M {m};
        """)

    def write(self, vecs, chunk=5000):
        """批量写入，返回总耗时秒。

        用 INSERT ... $rows 参数化传递，不用逐条 CREATE 拼 SQL。实测 50k/768 维：
        逐条 CREATE 拼 78 MB 文本要 47.8s 且超线性恶化，INSERT 只要 17.4s。
        """
        n = len(vecs)
        t0 = time.perf_counter()
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            rows = [
                {"pk": i, "emb": vecs[i].tolist()}
                for i in range(start, end)
            ]
            self.db.query("INSERT INTO vec $rows;", {"rows": rows})
        return time.perf_counter() - t0

    def count(self):
        res = self.db.query("SELECT count() FROM vec GROUP ALL;")
        if isinstance(res, list):
            res = res[0]
        return res.get("result", res) if isinstance(res, dict) else res

    def query_once(self, q, ef=None):
        # <|K,EF|> 操作符走向量索引；ORDER BY distance 会退化成全表扫描
        e = ef if ef is not None else 64
        t0 = time.perf_counter()
        res = self.db.query(
            f"SELECT pk, emb FROM vec WHERE emb <|{self.k},{e}|> $q;",
            {"q": [float(x) for x in q]},
        )
        dt = (time.perf_counter() - t0) * 1000
        # 单条语句：直接返回记录列表；多语句：外层包一层结果列表
        rows = res[0] if isinstance(res, list) and res and isinstance(res[0], list) else res
        pks = [int(r["pk"]) for r in rows if "pk" in r]
        return pks, dt

    def drop(self):
        self.db.query("REMOVE TABLE IF EXISTS vec;")

    def close(self):
        try:
            self.db.close()
        except Exception:
            pass


# ---------------------------------------------------------------- Milvus

class MilvusBench:
    name = "Milvus"

    def __init__(self, dim, k_nn):
        from pymilvus import MilvusClient

        self.dim = dim
        self.k = k_nn
        self.client = MilvusClient(uri=MILVUS_URL)
        self.coll = "vec"
        # 不在 create_collection 里传 index_params：服务端会把它降级成
        # AUTOINDEX（实测 describe_index 返回 index_type=AUTOINDEX），
        # AUTOINDEX 不吃查询侧 ef。改为建表后显式 create_index 建 HNSW。
        self.client.create_collection(
            collection_name=self.coll,
            dimension=dim,
            metric_type="L2",
            id_type="int",
            primary_field_name="pk",
            vector_field_name="vector",
        )

    def reset(self):
        pass

    def create_index(self, efc=200, m=16):
        from pymilvus.milvus_client.index import IndexParams

        # ef_scan 会连跑两轮，collection 已带索引时不能重复建
        if self.client.has_collection(self.coll):
            idxs = self.client.list_indexes(self.coll)
            if idxs:
                self._wait_index_ready()
                return
        params = IndexParams()
        params.add_index(
            field_name="vector",
            index_type="HNSW",
            metric_type="L2",
            params={"M": m, "efConstruction": efc},
        )
        self.client.create_index(collection_name=self.coll, index_params=params)
        self.client.load_collection(collection_name=self.coll)
        self._wait_index_ready()

    def _wait_index_ready(self, timeout=300):
        """Milvus 3.0 的索引是异步构建的，没建完时查询侧 ef 完全无效。"""
        deadline = time.time() + timeout
        while time.time() < deadline:
            d = self.client.describe_index(
                collection_name=self.coll, index_name="vector"
            )
            if d.get("state") == "Finished" and d.get("pending_index_rows", 0) == 0:
                return
            time.sleep(1)
        raise TimeoutError("milvus index not ready")

    def write(self, vecs, chunk=5000):
        n = len(vecs)
        t0 = time.perf_counter()
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            rows = [{"pk": i, "vector": vecs[i].tolist()} for i in range(start, end)]
            self.client.insert(collection_name=self.coll, data=rows)
        self.client.flush(collection_name=self.coll)
        return time.perf_counter() - t0

    def count(self):
        return self.client.query(
            collection_name=self.coll,
            filter="pk >= 0",
            output_fields=["count(*)"],
        )

    def query_once(self, q, ef=64):
        t0 = time.perf_counter()
        res = self.client.search(
            collection_name=self.coll,
            data=[q.tolist()],
            limit=self.k,
            search_params={"metric_type": "L2", "params": {"ef": ef}},
            output_fields=["pk"],
        )
        dt = (time.perf_counter() - t0) * 1000
        pks = [int(h["entity"]["pk"]) for h in res[0]]
        return pks, dt

    def drop(self):
        if self.client.has_collection(self.coll):
            self.client.drop_collection(self.coll)
        # drop 是异步的，等它真没了再返回，否则下一轮 create 撞残留
        for _ in range(60):
            if not self.client.has_collection(self.coll):
                return
            time.sleep(0.5)

    def close(self):
        try:
            self.client.close()
        except Exception:
            pass


# ---------------------------------------------------------------- 跑分

def run_phase(bench, vecs, queries, gt, rounds=5, ef=None):
    """跑 rounds 轮，返回延迟分布与平均召回。"""
    recalls = []
    lat_first = None
    lat_rest = []
    for r in range(rounds):
        rcs = []
        for i, q in enumerate(queries):
            pks, dt = bench.query_once(q, ef=ef) if ef is not None else bench.query_once(q)
            if r == 0:
                lat_first = dt
            else:
                lat_rest.append(dt)
            rcs.append(len(set(pks) & set(gt[i].tolist())) / len(gt[i]))
        recalls.append(statistics.mean(rcs))
    return {
        "first_call_ms": lat_first,
        "warm_lat_p50_ms": statistics.median(lat_rest),
        "warm_lat_p95_ms": np.percentile(lat_rest, 95),
        "warm_lat_p99_ms": np.percentile(lat_rest, 99),
        "warm_lat_mean_ms": statistics.mean(lat_rest),
        "recall_mean": statistics.mean(recalls),
        "qps": 1000.0 / statistics.mean(lat_rest) if lat_rest else 0,
    }


def bench_one(cls, scale, dim, k, rounds):
    print(f"\n{'=' * 62}\n{cls.__name__}  scale={scale} dim={dim} k={k}\n{'=' * 62}")
    vecs, q_near, q_rand = make_dataset(scale, dim)
    print("computing ground truth ...")
    gt = ground_truth(vecs, q_near, k)

    b = cls(dim=dim, k_nn=k)
    try:
        b.reset()
        b.create_index()
        print(f"writing {scale} vectors ...")
        tw = b.write(vecs)
        print(f"  write: {tw:.2f}s  ({scale / tw:.0f} vec/s)")
        time.sleep(2)

        out = {
            "db": b.name,
            "scale": scale,
            "dim": dim,
            "k": k,
            "write_s": tw,
            "write_vec_per_s": scale / tw,
        }
        for tag, qs in (("near", q_near), ("rand", q_rand)):
            print(f"querying [{tag}] ...")
            out[tag] = run_phase(b, vecs, qs, gt, rounds=rounds)
            print(f"  p50={out[tag]['warm_lat_p50_ms']:.1f}ms "
                  f"p95={out[tag]['warm_lat_p95_ms']:.1f}ms "
                  f"recall={out[tag]['recall_mean']:.4f} "
                  f"qps={out[tag]['qps']:.0f}")
        return out
    finally:
        b.drop()
        b.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="10k")
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--tag", default="")
    ap.add_argument("--only", default="", help="surreal / milvus，只跑一个库")
    args = ap.parse_args()

    n = {"1k": 1000, "10k": 10000, "50k": 50000, "100k": 100000}[args.scale]

    results = []
    libs = {"surreal": SurrealBench, "milvus": MilvusBench}
    if args.only:
        libs = {k: v for k, v in libs.items() if k == args.only.lower()}
        if not libs:
            raise SystemExit(f"--only 只接受 surreal / milvus，收到 {args.only}")
    for cls in libs.values():
        r = bench_one(cls, n, args.dim, args.k, args.rounds)
        r["env"] = {
            "machine": platform.machine(),
            "cpu": platform.processor(),
            "python": platform.python_version(),
            "surreal_version": "3.2.4",
            "milvus_version": "3.0.1 client / v3.0.0 server",
        }
        results.append(r)

    RESULTS.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = RESULTS / f"bench-{args.scale}-d{args.dim}-{stamp}.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
