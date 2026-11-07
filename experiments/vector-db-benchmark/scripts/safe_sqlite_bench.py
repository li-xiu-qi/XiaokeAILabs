"""Small exact-search check using a new, process-local SQLite database."""
import argparse
import importlib.metadata
import sqlite3
import time
import numpy as np
import sqlite_vec


def run(vectors=1000, queries=20, dimensions=32, seed=0):
    if min(vectors, queries, dimensions) <= 0 or vectors < 10:
        raise ValueError('vectors must be at least 10; queries and dimensions must be positive')
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((vectors, dimensions)).astype(np.float32)
    query = rng.standard_normal((queries, dimensions)).astype(np.float32)
    times, recalls = [], []
    # :memory: creates a separate database per connection and cannot touch user files.
    with sqlite3.connect(':memory:') as db:
        db.enable_load_extension(True)
        try:
            sqlite_vec.load(db)
        finally:
            db.enable_load_extension(False)
        db.execute(f'CREATE VIRTUAL TABLE vec USING vec0(embedding float[{dimensions}])')
        db.executemany('INSERT INTO vec(rowid, embedding) VALUES (?, ?)',
                       [(i + 1, sqlite_vec.serialize_float32(v)) for i, v in enumerate(data)])
        for v in query:
            exact = set((np.argsort(np.sum((data-v)**2, axis=1))[:10] + 1).tolist())
            start = time.perf_counter()
            rows = db.execute('SELECT rowid FROM vec WHERE embedding MATCH ? AND k = 10 ORDER BY distance',
                              (sqlite_vec.serialize_float32(v),)).fetchall()
            times.append((time.perf_counter() - start) * 1000)
            recalls.append(len(exact.intersection(row[0] for row in rows)) / 10)
    return {'vectors': vectors, 'queries': queries, 'dimensions': dimensions, 'seed': seed,
            'distance': 'L2', 'k': 10, 'warmup_queries': 0,
            'numpy_version': np.__version__, 'sqlite_version': sqlite3.sqlite_version,
            'sqlite_vec_version': importlib.metadata.version('sqlite-vec'),
            'latency_scope': 'query serialization, SQLite execution and result fetch',
            'recall_at_10_mean': float(np.mean(recalls)),
            'latency_ms_median': float(np.median(times)),
            'note': 'in-memory exact search, synthetic data, local SQLite only'}


def main():
    import json
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--vectors', type=int, default=1000)
    p.add_argument('--queries', type=int, default=20)
    p.add_argument('--dimensions', type=int, default=32)
    a = p.parse_args()
    print(json.dumps(run(a.vectors, a.queries, a.dimensions), indent=2))


if __name__ == '__main__':
    main()
