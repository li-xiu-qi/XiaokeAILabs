import json, random
from collections import Counter
from pathlib import Path

path=Path(__file__).resolve().parents[1]/'evidence'/'holdout-48-paired.json'
rows=json.loads(path.read_text(encoding='utf8'))['samples']
if len(rows)!=48 or len({r['sample'] for r in rows})!=48:
    raise ValueError('Expected 48 uniquely numbered samples')
if any(type(r[k]) is not bool for r in rows for k in ('baseline_hit','lora_hit')):
    raise ValueError('Hit fields must be booleans')
deltas=[int(r['lora_hit'])-int(r['baseline_hit']) for r in rows]
n=len(rows); rng=random.Random(20261107)
boot=sorted(sum(deltas[rng.randrange(n)] for _ in range(n))/n for _ in range(10000))
print(json.dumps({'n':n,'baseline_hits':sum(r['baseline_hit'] for r in rows),'lora_hits':sum(r['lora_hit'] for r in rows),'improved':deltas.count(1),'regressed':deltas.count(-1),'delta':sum(deltas)/n,'bootstrap_95':[boot[250],boot[9749]],'by_source':dict(sorted(Counter(r['source'] for r in rows).items()))},indent=2))
