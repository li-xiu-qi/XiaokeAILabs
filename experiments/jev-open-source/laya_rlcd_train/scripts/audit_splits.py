"""Inspect exact case and state overlap between decision JSONL splits."""
import hashlib
import json
from pathlib import Path


def fingerprints(path):
    cases, states = set(), set()
    count = 0
    with Path(path).open(encoding='utf-8-sig') as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not {'state', 'questions', 'gold'} <= row.keys():
                raise ValueError(f'{path}:{line_number}: missing state, questions or gold')
            def fingerprint(value):
                canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
                return hashlib.sha256(canonical.encode('utf-8')).hexdigest()
            cases.add(fingerprint({key: row[key] for key in ('state', 'questions', 'gold')}))
            states.add(fingerprint(row['state']))
            count += 1
    if not count:
        raise ValueError(f'No records in {path}')
    return count, cases, states


def audit(train_path, test_path):
    train_n, train_cases, train_states = fingerprints(train_path)
    test_n, test_cases, test_states = fingerprints(test_path)
    return {'train_records': train_n, 'test_records': test_n,
            'shared_exact_cases': len(train_cases & test_cases),
            'shared_exact_states': len(train_states & test_states)}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('train')
    parser.add_argument('test')
    args = parser.parse_args()
    result = audit(args.train, args.test)
    print(json.dumps(result, indent=2))
    if result['shared_exact_cases']:
        raise SystemExit('Identical cases found in training and evaluation')
