import os
import json, random, statistics
from pathlib import Path
root=Path(os.environ['GUI_ROOT']).expanduser()
def read(name):
    rows=[json.loads(line) for line in (root/name).read_text(encoding='utf8').splitlines()]
    assert len(rows)==30 and len({r['row_idx'] for r in rows})==30
    return rows
before=read('qwen3vl-2b-screenspot-30.jsonl'); after=read('qwen3vl-2b-lora-screenspot-30.jsonl')
assert [x['row_idx'] for x in before]==[x['row_idx'] for x in after]
delta=[int(y['hit'])-int(x['hit']) for x,y in zip(before,after)]; rng=random.Random(20261107)
boot=sorted(sum(delta[rng.randrange(30)] for _ in range(30))/30 for _ in range(10000))
def summary(rows):
    return {'hit':sum(x['hit'] for x in rows),'total':len(rows),'invalid_coordinates':sum(x['point_normalized'] is None for x in rows),
            'per_source':{s:{'hits':sum(x['hit'] for x in rows if x['source']==s),'total':sum(x['source']==s for x in rows)} for s in sorted({x['source'] for x in rows})},
            'median_generation_seconds':statistics.median(x['generation_s'] for x in rows)}
out={'baseline':summary(before),'finetuned':summary(after),'paired_improvements':sum(d==1 for d in delta),
     'paired_regressions':sum(d==-1 for d in delta),'hit_rate_delta':sum(delta)/30,'paired_bootstrap_95':[boot[250],boot[9749]],
     'limits':'30 non-random examples; confidence interval samples only these examples; point-in-box is not task success.'}
(root/'qwen3vl-2b-screenspot-comparison.json').write_text(json.dumps(out,indent=2),encoding='utf8'); print(json.dumps(out,indent=2))
