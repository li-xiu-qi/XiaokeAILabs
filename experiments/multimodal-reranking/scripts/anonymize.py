"""Reduce local candidate files to anonymous paired ranks and shuffle their order."""
import argparse
import json
import math
import random
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--candidates', type=Path, required=True)
p.add_argument('--reranked', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
before = json.loads(a.candidates.read_text(encoding='utf-8'))
after = [json.loads(x) for x in a.reranked.read_text(encoding='utf-8').splitlines()]
if len(before) != 100 or len(after) != 100:
    raise ValueError('Expected 100 paired queries in both input files')
out = []
for x, y in zip(before, after):
    if x['target_filename'] != y.get('target_filename') or x['query'] != y.get('query'):
        raise ValueError('candidate and reranked query/target differ; regenerate incompatible older rerank output')
    candidates, ranked = x['candidates'], y.get('candidates', [])
    if len(candidates) != 5 or len(ranked) != len(candidates) or len({z['filename'] for z in candidates}) != len(candidates):
        raise ValueError('expected five unique matching candidates per query')
    for original, scored in zip(candidates, ranked):
        score = scored.get('reranker_score')
        if ({k: v for k, v in scored.items() if k != 'reranker_score'} != original
                or isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score)):
            raise ValueError('candidate fields or reranker scores are invalid')
    target = x['target_filename']
    def rank(items):
        return next((i + 1 for i, item in enumerate(items) if item['filename'] == target), None)
    out.append({'before_rank_at_5': rank(x['candidates']), 'after_rank_at_5': rank(sorted(y['candidates'], key=lambda z: z['reranker_score'], reverse=True))})
random.Random(938217).shuffle(out)
if a.output.exists():
    raise FileExistsError(a.output)
a.output.write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8')
