"""Recompute published aggregates and paired bootstrap CIs from anonymized scores."""
import csv
import json
import random
from pathlib import Path

root = Path(__file__).resolve().parent
with (root / 'per-query-metrics.csv').open(encoding='utf-8', newline='') as f:
    rows = list(csv.DictReader(f))
assert len(rows) == 300
names = ('baseline', 'supervised', 'distilled', 'teacher')
metrics = {n: {m: sum(float(r[f'{n}_{m}']) for r in rows) / len(rows)
               for m in ('ndcg10', 'mrr10')} for n in names}
rng = random.Random(20261107)
def compare(a, b):
    differences = [float(r[f'{a}_ndcg10']) - float(r[f'{b}_ndcg10']) for r in rows]
    samples = sorted(sum(differences[rng.randrange(len(rows))] for _ in rows) / len(rows)
                     for _ in range(10000))
    return {'mean': sum(differences) / len(rows), 'paired_bootstrap_95': [samples[250], samples[9749]],
            'up': sum(x > 1e-12 for x in differences),
            'down': sum(x < -1e-12 for x in differences),
            'same': sum(abs(x) <= 1e-12 for x in differences)}
comparisons = {'distilled_vs_baseline': compare('distilled', 'baseline'),
               'distilled_vs_supervised': compare('distilled', 'supervised')}
expected = json.loads((root / 'summary.json').read_text(encoding='utf-8'))
for name in names:
    for metric in ('ndcg10', 'mrr10'):
        assert abs(metrics[name][metric] - expected['metrics'][name][metric]) < 1e-12
for name, result in comparisons.items():
    old = expected['comparisons'][name]
    assert abs(result['mean'] - old['mean']) < 1e-12
    assert all(abs(a-b) < 1e-12 for a,b in zip(result['paired_bootstrap_95'], old['paired_bootstrap_95']))
    assert all(result[k] == old[k] for k in ('up', 'down', 'same'))
print(json.dumps({'queries':len(rows), 'metrics':metrics, 'comparisons':comparisons}, indent=2))
print('All aggregate metrics and bootstrap comparisons match summary.json.')
