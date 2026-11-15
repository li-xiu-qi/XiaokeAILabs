"""Recompute student metrics from the recorded paired test ranks."""
import json
import random
from pathlib import Path
from statistics import mean

path = Path(__file__).resolve().parents[1] / 'evidence/student-test-ranks.json'
rows = json.loads(path.read_text(encoding='utf8'))
assert all(len(ranks) == 100 and all(isinstance(x, int) and 1 <= x <= 100 for x in ranks)
           for ranks in rows.values())
for name, ranks in rows.items():
    print(f'{name:12} R@1={mean(x == 1 for x in ranks):.4f} '
          f'R@5={mean(x <= 5 for x in ranks):.4f} MRR={mean(1 / x for x in ranks):.4f}')
deltas = [int(a == 1) - int(b == 1) for a, b in zip(rows['distilled'], rows['supervised'])]
rng = random.Random(20261107)
interval = sorted(mean(rng.choices(deltas, k=len(deltas))) for _ in range(10000))
print(f'distilled - supervised R@1 = {mean(deltas):+.4f}, '
      f'95% percentile [{interval[250]:+.4f}, {interval[9749]:+.4f}]')
