import json,random,statistics,re,math
from pathlib import Path
import sys
p=Path(sys.argv[1] if len(sys.argv)>1 else 'docs/evidence')
files={'teacher':'teacher-eval.json','baseline':'student-before.json','supervised':'supervised-eval.json','distilled':'distilled-eval.json'}
d={k:json.loads((p/v).read_text()) for k,v in files.items()}
def per_query(condition):
    part = d[condition]['test']
    rows = part['per_query']
    indexed = {row['qid']: row for row in rows}
    if not rows or len(indexed) != len(rows) or part['n'] != len(rows):
        raise ValueError(f'{condition}: invalid or duplicate test query IDs')
    for row in rows:
        if any(isinstance(row[m], bool) or not isinstance(row[m], (int, float)) or not math.isfinite(row[m]) or not 0 <= row[m] <= 1 for m in ('ndcg10', 'mrr10')):
            raise ValueError(f"{condition}: invalid metric for query {row['qid']}")
    return indexed
indexed = {name: per_query(name) for name in files}
reference = indexed['baseline']
for condition, rows in indexed.items():
    if rows.keys() != reference.keys():
        raise ValueError(f'{condition}: test query IDs do not match baseline')
    for qid, row in rows.items():
        if any(row[field] != reference[qid][field] for field in ('relevant_in_pool', 'relevant_total')):
            raise ValueError(f'{condition}: relevance counts differ for query {qid}')
rng=random.Random(20261107)
out={}
for a,b in [('distilled','baseline'),('supervised','baseline'),('distilled','supervised'),('teacher','baseline')]:
    left={qid:row['ndcg10'] for qid,row in indexed[a].items()}; right={qid:row['ndcg10'] for qid,row in indexed[b].items()}
    vals=[left[q]-right[q] for q in left]; samples=sorted(statistics.mean(rng.choices(vals,k=len(vals))) for _ in range(10000))
    out[f'{a}_minus_{b}']={'mean_difference':statistics.mean(vals),'bootstrap95':[samples[249],samples[9749]],'n_queries':len(vals)}
monitor=p/'memory-monitor.log'
if monitor.exists():
    mem=[int(x) for x in re.findall(r'available_kib=(\d+)',monitor.read_text())]
    out['monitor']={'min_memavailable_gib':min(mem)/1024**2,'samples':len(mem),'interval_seconds':2}
out['timing']={k:{'seconds_dev_and_test':v['seconds'],'caveat':'Single sequential scoring pass, includes tokenization, not a production latency benchmark'} for k,v in d.items()}

print(json.dumps(out,indent=2))
