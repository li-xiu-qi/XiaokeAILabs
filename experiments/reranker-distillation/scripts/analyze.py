import json,random,statistics,re
from pathlib import Path
import sys
p=Path(sys.argv[1] if len(sys.argv)>1 else 'docs/evidence')
files={'teacher':'teacher-eval.json','baseline':'student-before.json','supervised':'supervised-eval.json','distilled':'distilled-eval.json'}
d={k:json.loads((p/v).read_text()) for k,v in files.items()}
rng=random.Random(20261107)
out={}
for a,b in [('distilled','baseline'),('supervised','baseline'),('distilled','supervised'),('teacher','baseline')]:
    left={x['qid']:x['ndcg10'] for x in d[a]['test']['per_query']}; right={x['qid']:x['ndcg10'] for x in d[b]['test']['per_query']}; assert left.keys()==right.keys()
    vals=[left[q]-right[q] for q in left]; samples=sorted(statistics.mean(rng.choices(vals,k=len(vals))) for _ in range(10000))
    out[f'{a}_minus_{b}']={'mean_difference':statistics.mean(vals),'bootstrap95':[samples[249],samples[9749]],'n_queries':len(vals)}
monitor=p/'memory-monitor.log'
if monitor.exists():
    mem=[int(x) for x in re.findall(r'available_kib=(\d+)',monitor.read_text())]
    out['monitor']={'min_memavailable_gib':min(mem)/1024**2,'samples':len(mem),'interval_seconds':2}
out['timing']={k:{'seconds_dev_and_test':v['seconds'],'caveat':'Single sequential scoring pass, includes tokenization, not a production latency benchmark'} for k,v in d.items()}

print(json.dumps(out,indent=2))
