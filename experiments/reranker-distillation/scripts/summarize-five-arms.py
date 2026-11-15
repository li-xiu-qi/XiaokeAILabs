import json,statistics
from pathlib import Path
import numpy as np
p=Path(__file__).resolve().parents[1]/'evidence'/'hard-soft-five-arms'
groups=['human','teacher_hard','teacher_soft','human_hard','human_soft'];seeds=[20261107,20261108,20261109]
data={g:{s:json.loads((p/f'{g}-{s}-eval.json').read_text()) for s in seeds} for g in groups}
result={}
for split in ['dev','test']:
 result[split]={}
 for g in groups:
    vals=[data[g][s][split]['ndcg10'] for s in seeds]
    result[split][g]={'seeds':vals,'mean':statistics.mean(vals),'std':statistics.stdev(vals)}
 for a,b in [('teacher_hard','teacher_soft'),('human','human_hard'),('human','human_soft')]:
    qids=[x['qid'] for x in data[a][seeds[0]][split]['per_query']]
    by={g:{s:{x['qid']:x['ndcg10'] for x in data[g][s][split]['per_query']} for s in seeds} for g in [a,b]}
    delta=np.array([np.mean([by[b][s][q]-by[a][s][q] for s in seeds]) for q in qids]);rng=np.random.default_rng(20261107)
    means=rng.choice(delta,(10000,len(delta)),replace=True).mean(axis=1)
    seed_differences=[data[b][s][split]['ndcg10']-data[a][s][split]['ndcg10'] for s in seeds]
    result[split][f'{b}-minus-{a}']={'difference':float(delta.mean()),'per_seed_differences':seed_differences,'seed_difference_std':statistics.stdev(seed_differences),'query_bootstrap_95':np.quantile(means,[.025,.975]).tolist(),'note':'unadjusted query bootstrap on seed-mean differences; does not estimate uncertainty across training seeds or correct multiple comparisons'}
print(json.dumps(result,indent=2))
