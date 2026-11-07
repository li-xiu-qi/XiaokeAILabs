"""Recompute exact and whitespace-insensitive rates from the recorded outputs."""
import json
import math
import pathlib
import re
import statistics

root = pathlib.Path(__file__).resolve().parent
s = json.loads((root / 'results.json').read_text(encoding='utf-8'))
tasks = {t['id']: t['answer'] for t in json.loads((root / 'tasks.json').read_text(encoding='utf-8'))}
normalize = lambda text: re.sub(r'\s+', '', text).casefold()
for model in s['results']:
    runs = model['runs']
    assert len(runs) == 24 and len(tasks) == 8
    assert {(r['task_id'], r['repetition']) for r in runs} == {(t, i) for t in tasks for i in range(3)}
    for r in runs:
        seconds = r['seconds']
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError(f"invalid generation time for {model['model']}: {seconds!r}")
        assert r['expected'] == tasks[r['task_id']]
        assert r['correct'] == (r['answer'].strip().casefold() == r['expected'].strip().casefold())
    strict = sum(r['correct'] for r in runs)
    loose = sum(normalize(r['answer']) == normalize(r['expected']) for r in runs)
    median = statistics.median(r['seconds'] for r in runs)
    assert abs(strict / len(runs) - model['exact_match_rate']) < 1e-12
    assert abs(median - model['generation_seconds_median']) < 1e-12
    print(f"{model['model']}: strict={strict}/24 whitespace={loose}/24 median={median:.6f}s")
