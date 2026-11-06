import json,math
from pathlib import Path
root=Path(__file__).parent
sup=json.loads((root/'synthetic-train-results.json').read_text());kd=json.loads((root/'synthetic-distill-results.json').read_text());teacher=json.loads((root/'qwen-train-scores.json').read_text())
assert len(teacher['batches'])==8
for b in teacher['batches']:
 assert len(b['batch'])==2 and len({r['image'] for r in b['batch']})==2 and all(r['split']=='train' for r in b['batch'])
 assert len(b['scores'])==2 and all(len(v)==2 and all(math.isfinite(x) and 0<=x<=1 for x in v) for v in b['scores'])
assert sup['baseline']['ranks']==kd['baseline']['ranks']
for label,r in [('original',sup['baseline']),('supervised',sup['trained']),('distilled',kd['trained'])]:
 assert len(r['ranks'])==96 and r['candidates']==32
 assert abs(r['r1']-sum(x==1 for x in r['ranks'])/96)<1e-12 and abs(r['r5']-sum(x<=5 for x in r['ranks'])/96)<1e-12
 print(label,'R@1',sum(x==1 for x in r['ranks']),'/96')
a=sup['trained']['ranks'];b=kd['trained']['ranks'];up=sum(x!=1 and y==1 for x,y in zip(a,b));down=sum(x==1 and y!=1 for x,y in zip(a,b));print('distilled vs supervised:',up,'up,',down,'down, net',up-down)
