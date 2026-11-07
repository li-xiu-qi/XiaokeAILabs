"""Recompute aggregate metrics and paired intervals from per-query evidence."""
import json
import math
import random
from pathlib import Path
from statistics import mean

rows = json.loads((Path(__file__).resolve().parents[1] / 'evidence/per-query.json').read_text(encoding='utf8'))
if set(rows) != {'teacher', 'baseline', 'supervised', 'distilled'}:
    raise ValueError('expected exactly four evaluation conditions')
ids = [r['query_id'] for r in rows['teacher']]
if len(ids) != 300 or len(set(ids)) != len(ids) or any([r['query_id'] for r in v] != ids for v in rows.values()):
    raise ValueError('expected the same 300 unique queries in every condition and order')
for name, data in rows.items():
    for row in data:
        for metric in ('ndcg10', 'mrr10'):
            value = row[metric]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'{name}: invalid {metric} for query {row["query_id"]}')
for name, data in rows.items():
    print(f"{name:12} nDCG@10={mean(x['ndcg10'] for x in data):.4f} MRR@10={mean(x['mrr10'] for x in data):.4f}")
for reference in ('baseline', 'supervised'):
    differences = [a['ndcg10'] - b['ndcg10'] for a,b in zip(rows['distilled'], rows[reference])]
    rng = random.Random(20261107)
    bootstrap = sorted(mean(rng.choices(differences, k=len(differences))) for _ in range(10000))
    print(f"distilled - {reference}: {mean(differences):+.4f}, 95% percentile [{bootstrap[250]:+.4f}, {bootstrap[9749]:+.4f}]")
