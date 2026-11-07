"""Validate and summarize the released per-frame ratings and blind mapping."""
import csv
import json
import re
from pathlib import Path

root = Path(__file__).parent
plan = json.loads((root / 'eval-plan.json').read_text(encoding='utf-8'))
mapping = json.loads((root / 'blind-mapping.json').read_text(encoding='utf-8'))
with (root / 'scores.csv').open(encoding='utf-8', newline='') as f:
    rows = list(csv.DictReader(f))
ids = {item['blind_id'] for item in mapping}
if not (len(mapping) == len(ids) == 8 and len(rows) == 40): raise ValueError('expected 8 unique videos and 40 frame ratings')
jobs = {f"{prompt['id']}-s{seed}-{arm}"
        for prompt in plan['prompts'] for seed in plan['seeds'] for arm in plan['arms']}
if len(jobs) != len(mapping): raise ValueError('job count mismatch')
if {item['job'] for item in mapping} != jobs: raise ValueError('job mapping incomplete')
if not all(re.fullmatch(r'[0-9a-f]{64}', item['sha256']) for item in mapping): raise ValueError('invalid SHA-256')
if len({item['sha256'] for item in mapping}) != len(mapping): raise ValueError('duplicate SHA-256')
expected = {(i, n) for i in ids for n in plan['frames_for_review']}
actual = {(r['blind_id'], int(r['frame'])) for r in rows}
if actual != expected or len(actual) != len(rows): raise ValueError('frame ratings incomplete or duplicated')
fields = ['scene_usable', 'dark_hair', 'red_shirt', 'yellow_neck_scarf']
if not all(r[f] in ('present', 'absent', 'unknown') for r in rows for f in fields): raise ValueError('invalid rating')
for item in sorted(mapping, key=lambda x: x['job']):
    sample = [r for r in rows if r['blind_id'] == item['blind_id']]
    print(item['job'], {f: {v: sum(r[f] == v for r in sample) for v in ('present', 'absent', 'unknown')} for f in fields})
