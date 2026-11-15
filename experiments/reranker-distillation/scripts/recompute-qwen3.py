"""Recompute the paired query bootstrap interval from saved metrics."""
import json
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parents[1] / 'evidence' / 'qwen3-8b'
supervised = json.loads((root / 'supervised-eval.json').read_text(encoding='utf-8'))
distilled = json.loads((root / 'distilled-eval.json').read_text(encoding='utf-8'))
for split in ('dev', 'test'):
    before = {r['qid']: r['ndcg10'] for r in supervised[split]['per_query']}
    after = {r['qid']: r['ndcg10'] for r in distilled[split]['per_query']}
    if before.keys() != after.keys():
        raise ValueError('Query IDs differ')
    differences = np.array([after[q] - before[q] for q in sorted(before)])
    samples = np.random.default_rng(20261107).choice(
        differences, (10000, len(differences)), replace=True
    ).mean(axis=1)
    print(split, 'n=', len(differences), 'mean=', float(differences.mean()),
          'paired 95% interval=', np.quantile(samples, [.025, .975]).tolist())
