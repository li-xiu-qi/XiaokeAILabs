"""Recompute counts and matched comparisons from the revealed blind ratings."""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

metrics = ('single_camera', 'color_correct', 'background_correct', 'severe_artifact')
arms = ('eval-base', 'eval-self', 'eval-teacher')
with (Path(__file__).parent / 'revealed-ratings.csv').open(encoding='utf-8', newline='') as f:
    rows = list(csv.DictReader(f))
if len(rows) != 48 or len({r['blind_id'] for r in rows}) != 48:
    raise ValueError('expected 48 unique blind IDs')
pairs = defaultdict(dict)
for r in rows:
    if r['arm'] not in arms or r['arm'] in pairs[r['id']]:
        raise ValueError('invalid or duplicate arm')
    if any(r[m] not in ('0', '1', 'unknown') for m in metrics):
        raise ValueError('invalid rating')
    pairs[r['id']][r['arm']] = r
if len(pairs) != 16 or any(set(p) != set(arms) for p in pairs.values()):
    raise ValueError('expected 16 matched triples')
for triple in pairs.values():
    if len({r['prompt'] for r in triple.values()}) != 1 or len({r['seed'] for r in triple.values()}) != 1:
        raise ValueError('prompt or seed mismatch in a matched triple')
counts = {a: {m: dict(Counter(r[m] for r in rows if r['arm'] == a)) for m in metrics} for a in arms}
comparisons = {}
for left, right in (('eval-base', 'eval-self'), ('eval-base', 'eval-teacher'), ('eval-self', 'eval-teacher')):
    comparisons[f'{left} -> {right}'] = {}
    for m in metrics:
        result = Counter()
        for triple in pairs.values():
            x, y = triple[left][m], triple[right][m]
            if 'unknown' in (x, y): result['unknown'] += 1
            elif x == y: result['same'] += 1
            elif (y == '1') != (m == 'severe_artifact'): result['improved'] += 1
            else: result['worse'] += 1
        comparisons[f'{left} -> {right}'][m] = dict(result)
print(json.dumps({'arm_counts': counts, 'paired_comparisons': comparisons}, indent=2))
