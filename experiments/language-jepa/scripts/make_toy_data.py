"""Generate tiny deterministic arithmetic pairs for plumbing checks, not benchmarking."""
import argparse
import json
import random
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
args = p.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
pairs = [(a, b) for a in range(10) for b in range(20)]
random.Random(42).shuffle(pairs)
for name, selected in [('train', pairs[:160]), ('test', pairs[160:])]:
    with (args.output / f'{name}.jsonl').open('x', encoding='utf-8') as f:
        for a, b in selected:
            row = {'messages': [
                {'role': 'system', 'content': 'Answer with only the integer.'},
                {'role': 'user', 'content': f'What is {a} + {b}?'},
                {'role': 'assistant', 'content': str(a + b)},
            ]}
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
print('wrote 160 train and 40 test unique ordered pairs, seed=42')
