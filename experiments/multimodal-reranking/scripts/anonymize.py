"""Reduce local candidate files to anonymous paired ranks and shuffle their order."""
import argparse
import json
import random
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--candidates', type=Path, required=True)
p.add_argument('--reranked', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
before = json.loads(a.candidates.read_text(encoding='utf-8'))
after = [json.loads(x) for x in a.reranked.read_text(encoding='utf-8').splitlines()]
assert len(before) == len(after) == 100
out = []
for x, y in zip(before, after):
    assert x['target_filename'] == y['target_filename']
    assert [z['filename'] for z in x['candidates']] == [z['filename'] for z in y['candidates']]
    target = x['target_filename']
    def rank(items):
        return next((i + 1 for i, item in enumerate(items) if item['filename'] == target), None)
    out.append({'before_rank_at_5': rank(x['candidates']), 'after_rank_at_5': rank(sorted(y['candidates'], key=lambda z: z['reranker_score'], reverse=True))})
random.Random(938217).shuffle(out)
if a.output.exists():
    raise FileExistsError(a.output)
a.output.write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8')