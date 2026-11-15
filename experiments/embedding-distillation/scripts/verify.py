"""Recompute aggregate metrics and paired intervals from per-query evidence."""
import json
import random
from pathlib import Path
from statistics import mean

rows = json.loads((Path(__file__).resolve().parents[1] / 'evidence/per-query.json').read_text(encoding='utf8'))
ids = [r['query_id'] for r in rows['teacher']]
assert len(ids) == 300 and all([r['query_id'] for r in v] == ids for v in rows.values())
for name, data in rows.items():
    print(f"{name:12} nDCG@10={mean(x['ndcg10'] for x in data):.4f} MRR@10={mean(x['mrr10'] for x in data):.4f}")
for reference in ('baseline', 'supervised'):
    differences = [a['ndcg10'] - b['ndcg10'] for a,b in zip(rows['distilled'], rows[reference])]
    rng = random.Random(20261107)
    bootstrap = sorted(mean(rng.choices(differences, k=len(differences))) for _ in range(10000))
    print(f"distilled - {reference}: {mean(differences):+.4f}, 95% percentile [{bootstrap[250]:+.4f}, {bootstrap[9749]:+.4f}]")
