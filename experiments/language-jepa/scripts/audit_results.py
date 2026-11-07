"""Recheck recorded toy results and detect commutative train/test overlap."""
import json
from pathlib import Path
import random

root = Path(__file__).resolve().parents[1]
pairs = [(a, b) for a in range(10) for b in range(20)]
random.Random(42).shuffle(pairs)
train, test = pairs[:160], pairs[160:]
train_unordered = {tuple(sorted(pair)) for pair in train}
overlap = {f'What is {a} + {b}?' for a, b in test if tuple(sorted((a, b))) in train_unordered}
expected = {f'What is {a} + {b}?': str(a + b) for a, b in test}
print(f'Commutative train/test overlap: {len(overlap)}/{len(test)} test prompts')
for name in ('untrained', 'baseline', 'jepa'):
    result = json.loads((root / 'results' / f'{name}-eval.json').read_text(encoding='utf-8'))
    records = result['records']
    actual = {r['input']: r['expected'] for r in records}
    if len(records) != len(expected) or len(actual) != len(records) or actual != expected:
        raise ValueError(f'{name}: evaluation inputs do not match generated test set')
    for row in records:
        if row['strict_exact_match'] != (row['generated'].strip() == row['expected']):
            raise ValueError(f"{name}: incorrect match for {row['input']}")
    correct = sum(row['strict_exact_match'] for row in records)
    if result['count'] != len(records) or abs(result['strict_exact_match'] - correct / len(records)) > 1e-12:
        raise ValueError(f'{name}: aggregate does not match records')
    print(f'{name}: {correct}/{len(records)}')
