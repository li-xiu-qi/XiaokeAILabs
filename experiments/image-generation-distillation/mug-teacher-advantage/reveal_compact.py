"""Reproduce score-to-ID mapping without images; image content cannot be checked."""
import argparse,csv,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--review',type=Path,required=True);p.add_argument('--key',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
key=json.loads(a.key.read_text(encoding='utf-8'))
with a.review.open(encoding='utf-8-sig',newline='') as f: ratings=list(csv.DictReader(f))
k={r['blind_id']:r for r in key};scores={r['blind_id']:r for r in ratings}
if len(key)!=64 or len(k)!=64 or len(ratings)!=64 or len(scores)!=64 or set(k)!=set(scores):raise SystemExit('invalid IDs')
fields=['id','arm','single_mug','color_correct','background_correct','severe_artifact'];rows=[]
for blind,r in scores.items():
    item=k[blind]
    if r['prompt']!=item['prompt'] or any(r[x] not in ('0','1','unknown') for x in fields[2:]):raise SystemExit('invalid prompt or score')
    rows.append({'id':item['id'],'arm':item['arm'],**{f:r[f] for f in fields[2:]}})
with a.out.open('x',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
print('64 ratings mapped; no image-content or image-hash verification performed')
