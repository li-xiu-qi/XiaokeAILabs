"""Evaluate the locally recorded gate after a blinded, per-image development review.

CSV requires id, arm, single_camera, color_correct, background_correct,
severe_artifact. Values are 0, 1 or unknown; arms are teacher and base.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

FIELDS = ('single_camera', 'color_correct', 'background_correct', 'severe_artifact')

def evaluate(rows, expected):
    pairs = defaultdict(dict)
    for row in rows:
        if row['id'] not in expected or row['arm'] not in ('teacher', 'base'):
            raise ValueError('unexpected id or arm')
        if row['arm'] in pairs[row['id']]:
            raise ValueError('duplicate id and arm')
        if any(row[field] not in ('0', '1', 'unknown') for field in FIELDS):
            raise ValueError('invalid rating')
        values = [row[field] for field in FIELDS]
        pairs[row['id']][row['arm']] = None if 'unknown' in values else values == ['1', '1', '1', '0']
    if set(pairs) != expected or any(set(x) != {'teacher', 'base'} for x in pairs.values()):
        raise ValueError('incomplete development ratings')
    valid = [(p['teacher'], p['base']) for p in pairs.values() if p['teacher'] is not None and p['base'] is not None]
    teacher = sum(t for t, _ in valid)
    base = sum(b for _, b in valid)
    return {'valid_pairs': len(valid), 'unknown_pairs': len(pairs)-len(valid),
            'teacher_success': teacher, 'base_success': base,
            'teacher_only': sum(t and not b for t, b in valid),
            'base_only': sum(b and not t for t, b in valid),
            'passed': len(valid) >= 28 and teacher >= 20 and teacher-base >= 4}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scores', type=Path, required=True)
    p.add_argument('--development', type=Path, default=Path(__file__).parent/'frozen/development.jsonl')
    args = p.parse_args()
    data = [json.loads(line) for line in args.development.read_text(encoding='utf-8').splitlines()]
    ids = {r['id'] for r in data}
    if len(data) != 32 or len(ids) != 32:
        raise ValueError('expected 32 distinct development prompts')
    with args.scores.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    print(json.dumps(evaluate(rows, ids), indent=2))

if __name__ == '__main__':
    main()
