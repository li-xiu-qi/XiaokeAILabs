import json
from pathlib import Path
x=json.loads((Path(__file__).parent/'synthetic-train-results.json').read_text())
a,b=x['baseline'],x['trained']; assert a['n']==b['n']==96 and a['candidates']==b['candidates']==32
for r in (a,b):
 ranks=r['ranks']; assert len(ranks)==96 and all(1<=n<=32 for n in ranks)
 assert abs(r['r1']-sum(n==1 for n in ranks)/96)<1e-12
 assert abs(r['r5']-sum(n<=5 for n in ranks)/96)<1e-12
up=sum(aa!=1 and bb==1 for aa,bb in zip(a['ranks'],b['ranks']))
down=sum(aa==1 and bb!=1 for aa,bb in zip(a['ranks'],b['ranks']))
print('R@1 improved:',up,'worsened:',down,'net:',up-down,'of 96')
