"""Recompute grouped-test results from answers and deterministic targets."""
import json
from pathlib import Path
from make_grouped_data import make_rows

root=Path(__file__).resolve().parents[1]
rows=json.loads((root/'results'/'grouped-comparison.json').read_text(encoding='utf8'))['records']
_, test=make_rows(seed=20261107,train_count=320,test_count=80)
targets=[r['messages'][2]['content'] for r in test]
if len(rows)!=len(targets) or any(r['expected']!=target for r,target in zip(rows,targets)):
    raise ValueError('test target sequence mismatch')
modes=('base','baseline','jepa')
hits={m:[r[m].strip()==r['expected'] for r in rows] for m in modes}
result={'count':len(rows),'hits':{m:sum(hits[m]) for m in modes},'paired':{}}
for left,right in [('base','baseline'),('base','jepa'),('baseline','jepa')]:
    d=[int(b)-int(a) for a,b in zip(hits[left],hits[right])]
    result['paired'][left+'_to_'+right]={'improved':d.count(1),'regressed':d.count(-1),'delta':sum(d)/len(d)}
print(json.dumps(result,indent=2))
