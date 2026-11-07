"""Validate and summarize the released per-frame ratings and blind mapping."""
import csv
import json
from pathlib import Path

root = Path(__file__).parent
plan = json.loads((root / 'eval-plan.json').read_text(encoding='utf-8'))
mapping = json.loads((root / 'blind-mapping.json').read_text(encoding='utf-8'))
with (root / 'scores.csv').open(encoding='utf-8', newline='') as f:
    rows = list(csv.DictReader(f))
ids = {item['blind_id'] for item in mapping}
assert len(mapping) == len(ids) == 8 and len(rows) == 40
expected = {(i, n) for i in ids for n in plan['frames_for_review']}
actual = {(r['blind_id'], int(r['frame'])) for r in rows}
assert actual == expected and len(actual) == len(rows)
fields = ['scene_usable', 'dark_hair', 'red_shirt', 'yellow_neck_scarf']
assert all(r[f] in ('present', 'absent', 'unknown') for r in rows for f in fields)
for item in sorted(mapping, key=lambda x: x['job']):
    sample = [r for r in rows if r['blind_id'] == item['blind_id']]
    print(item['job'], {f: {v: sum(r[f] == v for r in sample) for v in ('present', 'absent', 'unknown')} for f in fields})
