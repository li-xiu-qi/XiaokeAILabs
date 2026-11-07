"""Validate labels and commutative split isolation without loading a model."""
import argparse
import json
import re
from pathlib import Path

PATTERN = re.compile(r'What is ([0-9]+) \+ ([0-9]+)\?')


def read_split(path):
    groups = set()
    for number, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        messages = json.loads(line)['messages']
        if len(messages) != 3 or [m['role'] for m in messages] != ['system', 'user', 'assistant']:
            raise ValueError(f'{path.name}:{number}: invalid message structure')
        match = PATTERN.fullmatch(messages[1]['content'])
        if not match:
            raise ValueError(f'{path.name}:{number}: invalid question')
        a, b = map(int, match.groups())
        if not (100 <= a <= 999 and 100 <= b <= 999 and a % 10 + b % 10 >= 10 and
                (a // 10) % 10 + (b // 10) % 10 + 1 >= 10):
            raise ValueError(f'{path.name}:{number}: operand or carry constraint failed')
        if messages[2]['content'] != str(a+b):
            raise ValueError(f'{path.name}:{number}: incorrect answer')
        key = tuple(sorted((a, b)))
        if key in groups:
            raise ValueError(f'{path.name}:{number}: duplicate or swapped equivalent')
        groups.add(key)
    if not groups:
        raise ValueError(f'{path.name}: empty split')
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    args = parser.parse_args()
    train = read_split(args.data / 'train.jsonl')
    test = read_split(args.data / 'test.jsonl')
    if train & test:
        raise ValueError(f'commutative train/test overlap: {len(train & test)}')
    print(f'train={len(train)} test={len(test)} commutative overlap=0; labels verified')


if __name__ == '__main__':
    main()
