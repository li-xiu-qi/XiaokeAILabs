import os
import json, random, statistics
from pathlib import Path
root=Path(os.environ['GUI_ROOT']).expanduser()
def read(name):
    rows=[json.loads(line) for line in (root/name).read_text(encoding='utf8').splitlines()]
    if not rows or len({r['row_idx'] for r in rows}) != len(rows):
        raise ValueError(f'Empty or duplicate evaluation rows: {name}')
    return rows
baseline_name='qwen3vl-2b-holdout.jsonl'
finetuned_name='qwen3vl-2b-lora-holdout.jsonl'
before=read(baseline_name); after=read(finetuned_name)

def context(name):
    path=(root/name).with_suffix('.context.json')
    if not path.is_file():
        raise ValueError(f'Missing evaluation context: {path}')
    return json.loads(path.read_text(encoding='utf8'))

base_context=context(baseline_name); tuned_context=context(finetuned_name)
for key in ('manifest_sha256', 'model_path', 'prompt_sha256'):
    if not base_context.get(key) or base_context.get(key)!=tuned_context.get(key):
        raise ValueError(f'Evaluation contexts differ or lack {key}; do not compare these runs')
if base_context.get('adapter_path') is not None or not tuned_context.get('adapter_path'):
    raise ValueError('Expected a base run and a run with an adapter')
if [x['row_idx'] for x in before]!=[x['row_idx'] for x in after]:
    raise ValueError('Evaluation rows differ or are not in the same order')
if any((a['source'],a['type'],a['bbox']) != (b['source'],b['type'],b['bbox']) for a,b in zip(before,after)):
    raise ValueError('Evaluation labels differ; do not compare these runs')
n=len(before)
delta=[int(y['hit'])-int(x['hit']) for x,y in zip(before,after)]; rng=random.Random(20261107)
boot=sorted(sum(delta[rng.randrange(n)] for _ in range(n))/n for _ in range(10000))
def summary(rows):
    return {'hit':sum(x['hit'] for x in rows),'total':len(rows),'invalid_coordinates':sum(x['point_normalized'] is None for x in rows),
            'per_source':{s:{'hits':sum(x['hit'] for x in rows if x['source']==s),'total':sum(x['source']==s for x in rows)} for s in sorted({x['source'] for x in rows})},
            'median_generation_seconds':statistics.median(x['generation_s'] for x in rows)}
out={'baseline':summary(before),'finetuned':summary(after),'paired_improvements':sum(d==1 for d in delta),
     'paired_regressions':sum(d==-1 for d in delta),'hit_rate_delta':sum(delta)/n,'paired_bootstrap_95':[boot[250],boot[9749]],
     'limits':'Stratified examples are non-random after image exclusions; this interval describes only resampling of these examples. Point-in-box is not task success.'}
with (root/'qwen3vl-2b-holdout-comparison.json').open('x',encoding='utf8') as output:
    json.dump(out,output,indent=2)
print(json.dumps(out,indent=2))
