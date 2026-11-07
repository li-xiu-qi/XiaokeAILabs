"""Recompute counts and matched comparisons from the revealed blind ratings."""
import csv
import json
import re
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
    if not re.fullmatch(r'[0-9a-f]{64}', r['sha256']):
        raise ValueError(f"invalid image SHA-256 for {r['blind_id']}")
    pairs[r['id']][r['arm']] = r
if len(pairs) != 16 or any(set(p) != set(arms) for p in pairs.values()):
    raise ValueError('expected 16 matched triples')
for triple in pairs.values():
    if len({r['prompt'] for r in triple.values()}) != 1 or len({r['seed'] for r in triple.values()}) != 1:
        raise ValueError('prompt or seed mismatch in a matched triple')
prompt_seeds = defaultdict(set)
for triple in pairs.values():
    row = triple['eval-base']
    prompt_seeds[row['prompt']].add(row['seed'])
if len(prompt_seeds) != 8 or any(len(seeds) != 2 for seeds in prompt_seeds.values()):
    raise ValueError('expected eight prompts with two distinct seeds each')
if len({(triple['eval-base']['prompt'], triple['eval-base']['seed']) for triple in pairs.values()}) != 16:
    raise ValueError('duplicate prompt/seed pair')
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
result = {'arm_counts': counts, 'paired_comparisons': comparisons}
recorded = json.loads((Path(__file__).parent / 'summary.json').read_text(encoding='utf-8'))
if result != recorded:
    raise ValueError('recomputed counts differ from summary.json')
print(json.dumps(result, indent=2))
