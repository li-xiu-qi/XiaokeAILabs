"""Generate a harder arithmetic plumbing set grouped by commutative equivalence.

This is a proposed follow-up, not evidence that a model benefits from training.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

SYSTEM = 'Answer with only the integer.'


def make_rows(seed, train_count, test_count):
    rng = random.Random(seed)
    seen = set()
    rows = []
    while len(rows) < train_count + test_count:
        a, b = rng.randint(100, 999), rng.randint(100, 999)
        # Force carries in the ones and tens places; still no guarantee that
        # the chosen base model will find these examples challenging.
        if (a % 10 + b % 10 < 10 or
                (a // 10) % 10 + (b // 10) % 10 + 1 < 10):
            continue
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        rows.append({'messages': [
            {'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': f'What is {a} + {b}?'},
            {'role': 'assistant', 'content': str(a + b)},
        ]})
    return rows[:train_count], rows[train_count:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261107)
    parser.add_argument('--train-count', type=int, default=320)
    parser.add_argument('--test-count', type=int, default=80)
    args = parser.parse_args()
    if not 1 <= args.train_count <= 10000 or not 1 <= args.test_count <= 10000:
        parser.error('counts must be between 1 and 10000')
    train, test = make_rows(args.seed, args.train_count, args.test_count)
    args.output.mkdir(parents=True, exist_ok=False)
    files = {}
    for name, rows in [('train', train), ('test', test)]:
        path = args.output / f'{name}.jsonl'
        with path.open('x', encoding='utf8', newline='\n') as out:
            for row in rows:
                out.write(json.dumps(row, ensure_ascii=False) + '\n')
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    meta = {'seed': args.seed, 'train_count': len(train), 'test_count': len(test),
            'operand_range': [100, 999], 'constraint': 'carry in ones and tens',
            'split_unit': 'unordered operand pair', 'sha256': files}
    (args.output / 'metadata.json').write_text(json.dumps(meta, indent=2) + '\n', encoding='utf8')
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
