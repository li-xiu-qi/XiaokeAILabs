"""Recompute paired image-review changes by object and exploratory sign test."""
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def exact_two_sided(changed_for_better, changed_for_worse):
    n = changed_for_better + changed_for_worse
    if not n:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(changed_for_better, changed_for_worse) + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prompts', type=Path, required=True)
    p.add_argument('--key', type=Path, required=True)
    p.add_argument('--scores', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    prompts = [json.loads(line) for line in args.prompts.read_text(encoding='utf-8').splitlines() if line.strip()]
    ids = {row['id'] for row in prompts}
    if len(ids) != len(prompts):
        raise ValueError('duplicate prompt ID')
    key = json.loads(args.key.read_text(encoding='utf-8'))
    with args.scores.open(encoding='utf-8-sig', newline='') as handle:
        scores = list(csv.DictReader(handle))
    by_pair = {}
    for row in scores:
        review_id = row['review_id']
        if review_id not in key:
            raise ValueError('score has no mapping')
        mapping = key[review_id]
        pair = (mapping['id'], mapping['arm'])
        if pair in by_pair:
            raise ValueError('duplicate prompt/arm pair')
        by_pair[pair] = row
    expected = {(i, arm) for i in ids for arm in ('teacher', 'student-before', 'student-after')}
    if set(by_pair) != expected or len(scores) != len(key):
        raise ValueError('incomplete or extra score/mapping')
    metrics = {'object_correct': False, 'background_correct': False,
               'complete_subject': False, 'obvious_artifact': True}
    results = {}
    for metric, reverse in metrics.items():
        total = Counter()
        groups = defaultdict(Counter)
        for prompt in prompts:
            before = by_pair[(prompt['id'], 'student-before')][metric]
            after = by_pair[(prompt['id'], 'student-after')][metric]
            if before not in ('0', '1') or after not in ('0', '1'):
                raise ValueError(f'missing or invalid {metric} for {prompt["id"]}')
            b, a = int(before), int(after)
            if reverse:
                b, a = 1 - b, 1 - a
            change = 'improved' if a > b else 'worsened' if a < b else 'unchanged'
            for count in (total, groups[prompt['object']]):
                count[change] += 1
                count['before_positive'] += b
                count['after_positive'] += a
                count['n'] += 1
        results[metric] = {'overall': dict(total), 'by_object': {k: dict(v) for k, v in sorted(groups.items())},
                           'exploratory_exact_sign_p_two_sided': exact_two_sided(total['improved'], total['worsened'])}
    result = {'n_prompts': len(prompts), 'n_reviewed_images': len(scores),
              'note': 'Single AI reviewer; four object categories share one prompt template. Sign test is exploratory and assumes exchangeable paired changes; it does not establish model generalization or human preference.',
              'metrics': results}
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
