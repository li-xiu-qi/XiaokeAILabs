"""Recompute fixed-candidate ranking metrics from anonymous rank pairs."""
import argparse
import json
import random
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('path', nargs='?', default=str(Path(__file__).resolve().parent.parent / 'evidence/anonymous-ranks.json'))
args = parser.parse_args()
rows = json.loads(Path(args.path).read_text(encoding='utf-8'))
assert len(rows) == 100 and all(set(r) == {'before_rank_at_5', 'after_rank_at_5'} for r in rows)
before = [r['before_rank_at_5'] for r in rows]
after = [r['after_rank_at_5'] for r in rows]
assert all(v is None or 1 <= v <= 5 for v in before + after)
assert all((a is None) == (b is None) for a, b in zip(before, after))
changes = [int(b == 1) - int(a == 1) for a, b in zip(before, after)]
rng = random.Random(20261107)
boot = sorted(sum(rng.choice(changes) for _ in changes) / len(changes) for _ in range(10000))
print(json.dumps({'n': len(rows), 'before_hit1': sum(r == 1 for r in before),
 'after_hit1': sum(r == 1 for r in after), 'top5_covered': sum(r is not None for r in before),
 'improved_hit1': changes.count(1), 'worsened_hit1': changes.count(-1),
 'before_mrr5': sum(1/r if r else 0 for r in before) / len(rows),
 'after_mrr5': sum(1/r if r else 0 for r in after) / len(rows),
 'delta_hit1_bootstrap_95': [boot[250], boot[9749]]}, indent=2))