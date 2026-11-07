"""Recalculate exact matches and verify targets against the generated test split."""
import json
from pathlib import Path
from make_grouped_data import make_rows

root = Path(__file__).resolve().parents[1]
records = json.loads((root / 'results' / 'grouped-base-eval.json').read_text(encoding='utf8'))['records']
_, test = make_rows(seed=20261107, train_count=320, test_count=80)
expected = [r['messages'][2]['content'] for r in test]
if len(records) != len(expected) or any(r['expected'] != target for r, target in zip(records, expected)):
    raise ValueError('recorded targets differ from the deterministic test split')
correct = sum(r['generated'].strip() == r['expected'] for r in records)
print(json.dumps({'count': len(records), 'strict_exact_match_hits': correct,
                  'strict_exact_match': correct / len(records)}, indent=2))
