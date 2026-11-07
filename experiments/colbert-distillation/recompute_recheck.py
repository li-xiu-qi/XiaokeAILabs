"""Recompute the 2026-11-07 paired rerun statistics from anonymous scores."""
import csv
import random
from pathlib import Path
from statistics import mean

path = Path(__file__).with_name('recheck-per-query.csv')
with path.open(encoding='utf8', newline='') as f:
    rows = list(csv.DictReader(f))
if len(rows) != 300 or [int(r['index']) for r in rows] != list(range(300)):
    raise ValueError('Expected 300 rows in paired order')
for name in ('baseline', 'supervised', 'distilled'):
    for metric in ('ndcg10', 'mrr10'):
        vals = [float(r[f'{name}_{metric}']) for r in rows]
        if not all(0 <= v <= 1 for v in vals):
            raise ValueError(f'Invalid {name} {metric}')
        print(f'{name} {metric}: {mean(vals):.8f}')
for other in ('baseline', 'supervised'):
    diffs = [float(r['distilled_ndcg10']) - float(r[f'{other}_ndcg10']) for r in rows]
    rng = random.Random(20261107)
    boot = sorted(mean(rng.choices(diffs, k=len(diffs))) for _ in range(10000))
    print(f'distilled - {other}: {mean(diffs):+.8f}, 95% [{boot[250]:+.8f}, {boot[9749]:+.8f}], '
          f'up/down/same {sum(x>0 for x in diffs)}/{sum(x<0 for x in diffs)}/{sum(x==0 for x in diffs)}')