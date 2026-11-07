"""Score fixed-answer runs; does not execute generated code or load any model."""
import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path


def assess(tasks, result):
    prompts = {task['prompt']: task for task in tasks}
    if len(prompts) != len(tasks) or not tasks:
        raise ValueError('task prompts must be nonempty and unique')
    runs = result['runs']
    seen = Counter(row['prompt'] for row in runs)
    if set(seen) != set(prompts) or len(set(seen.values())) != 1:
        raise ValueError('each task must have the same positive number of runs')
    pairs = [(row['prompt'], row['repetition']) for row in runs]
    if any(type(rep) is not int or rep < 0 for _, rep in pairs):
        raise ValueError('repetition IDs must be nonnegative integers')
    if len(set(pairs)) != len(pairs):
        raise ValueError('task repetition numbers must not be duplicated')
    repetitions_by_task = [{rep for prompt, rep in pairs if prompt == task_prompt} for task_prompt in prompts]
    if any(reps != repetitions_by_task[0] for reps in repetitions_by_task[1:]):
        raise ValueError('each task must have the same repetition IDs')
    rows = []
    for row in runs:
        task = prompts[row['prompt']]
        seconds = row['seconds']
        if type(seconds) not in (float, int) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('every run must have positive generation time')
        text = row['text']
        rows.append({'task_id': task['id'], 'repetition': row['repetition'],
                     'answer': text, 'expected': task['answer'],
                     'correct': text.strip().casefold() == str(task['answer']).strip().casefold(),
                     'seconds': seconds, 'output_utf8_bytes': row.get('output_utf8_bytes'),
                     'text_tokens_reencoded': row.get('text_tokens_reencoded'),
                     'returned_token_ids': row.get('returned_token_ids'),
                     'eos_count': row.get('eos_count'),
                     'contains_mask_token': row.get('contains_mask_token')})
    return {'model': result.get('model'), 'revision': result.get('revision'),
            'kind': result.get('kind', result.get('algorithm')), 'device': result.get('device'),
            'torch': result.get('torch'), 'transformers': result.get('transformers'),
            'dtype': result.get('dtype'), 'input_output_timing': result.get('timing'),
            'output_budget': result.get('max_new_tokens', result.get('gen_length')),
            'warmup_runs': len(result.get('warmup', [])), 'repetitions_per_task': next(iter(seen.values())),
            'repetition_ids': sorted(repetitions_by_task[0]),
            'generation_settings': result.get('diffusion_settings', result.get('algorithm')),
            'steps': result.get('steps'), 'block_length': result.get('block_length'),
            'exact_match_rate': sum(r['correct'] for r in rows) / len(rows),
            'generation_seconds_median': statistics.median(r['seconds'] for r in rows),
            'runs': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', required=True)
    parser.add_argument('--runs', required=True, nargs='+')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    tasks = json.loads(Path(args.tasks).read_text(encoding='utf8'))
    if any(not all(key in task for key in ('id', 'prompt', 'answer')) for task in tasks):
        raise ValueError('each task requires id, prompt and answer')
    if len({t['id'] for t in tasks}) != len(tasks):
        raise ValueError('task ids must be unique')
    results = [assess(tasks, json.loads(Path(p).read_text(encoding='utf8'))) for p in args.runs]
    budgets = {r['output_budget'] for r in results}
    repetitions = {r['repetitions_per_task'] for r in results}
    repetition_ids = {tuple(r['repetition_ids']) for r in results}
    if None in budgets or len(budgets) != 1 or len(repetitions) != 1 or len(repetition_ids) != 1:
        raise ValueError('runs must have the same recorded output token budget and repetition count')
    if any(len({r[key] for r in results}) != 1 for key in ('device', 'dtype', 'warmup_runs')):
        raise ValueError('runs must use the same recorded device, dtype, and warmup count')
    if any(r['warmup_runs'] < 1 or not r['device'] or not r['dtype'] for r in results):
        raise ValueError('device, dtype, and at least one warmup must be recorded')
    payload = {'scope': 'small exact-answer diagnostic; not a general capability benchmark or architecture-only comparison',
               'matching': 'case-insensitive exact match after stripping leading/trailing whitespace',
               'task_count': len(tasks), 'results': results}
    with Path(args.out).open('x', encoding='utf8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps([{'model': r['model'], 'exact_match_rate': r['exact_match_rate'],
                       'generation_seconds_median': r['generation_seconds_median']} for r in results], indent=2))

if __name__ == '__main__':
    main()
