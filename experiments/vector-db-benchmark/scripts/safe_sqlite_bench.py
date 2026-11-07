"""Small exact-search check using a new, process-local SQLite database."""
import argparse
import importlib.metadata
import sqlite3
import time
import numpy as np
import sqlite_vec


def recall_with_boundary_ties(distances, returned_ids, k=10):
    """Count strict neighbors first, then tied IDs up to the remaining places."""
    distances = np.asarray(distances)
    if distances.ndim != 1 or not np.issubdtype(distances.dtype, np.number) or np.iscomplexobj(distances):
        raise ValueError('distances must be a one-dimensional real numeric array')
    if not np.isfinite(distances).all() or np.any(distances < 0):
        raise ValueError('distances must be finite and non-negative')
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or not 1 <= k <= len(distances):
        raise ValueError('k must be an integer between 1 and the number of distances')
    boundary = np.partition(distances, k - 1)[k - 1]
    values = list(returned_ids)
    if any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer)) or i < 1 or i > len(distances) for i in values):
        raise ValueError('returned rowids must be integers within the dataset')
    ids = set(values)
    strictly_nearer = int(np.count_nonzero(distances < boundary))
    hits_nearer = sum(distances[i - 1] < boundary for i in ids)
    hits_tied = sum(distances[i - 1] == boundary for i in ids)
    return (hits_nearer + min(hits_tied, k - strictly_nearer)) / k


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
            distances = np.sum((data-v)**2, axis=1)
            start = time.perf_counter()
            rows = db.execute('SELECT rowid FROM vec WHERE embedding MATCH ? AND k = 10 ORDER BY distance',
                              (sqlite_vec.serialize_float32(v),)).fetchall()
            times.append((time.perf_counter() - start) * 1000)
            recalls.append(recall_with_boundary_ties(distances, (row[0] for row in rows)))
    return {'vectors': vectors, 'queries': queries, 'dimensions': dimensions, 'seed': seed,
            'distance': 'L2', 'k': 10, 'warmup_queries': 0,
            'numpy_version': np.__version__, 'sqlite_version': sqlite3.sqlite_version,
            'sqlite_vec_version': importlib.metadata.version('sqlite-vec'),
            'latency_scope': 'query serialization, SQLite execution and result fetch',
            'recall_at_10_mean': float(np.mean(recalls)),
            'recall_tie_policy': 'strict neighbors plus boundary ties capped to remaining places; float32 squared distance',
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
