import json, math
from pathlib import Path
x=json.loads((Path(__file__).parent/'baseline100.json').read_text()); rows=x['records']
assert len(rows)==100 and [r['row'] for r in rows]==list(range(100))
for r in rows:
 rank=r['rank']; assert 1<=rank<=100
 assert r['r1']==int(rank<=1) and r['r5']==int(rank<=5)
 assert abs(r['ndcg5']-(1/math.log2(rank+1) if rank<=5 else 0))<1e-12
for metric in ('r1','r5','ndcg5'):
 assert abs(x[metric]-sum(r[metric] for r in rows)/len(rows))<1e-12
print('Ranks and aggregate metrics match.')
