import argparse, csv, json
from collections import Counter
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--review',type=Path,required=True);p.add_argument('--key',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
if a.out.exists(): raise SystemExit('refusing overwrite')
key=json.loads(a.key.read_text(encoding='utf-8'))
with a.review.open(newline='',encoding='utf-8-sig') as f: scores=list(csv.DictReader(f))
if len(key)!=48 or len(scores)!=48: raise SystemExit('expected 48 scored images')
lookup={x['blind_id']:x for x in key}
if len(lookup)!=48 or len({x['blind_id'] for x in scores})!=48 or set(lookup)!={x['blind_id'] for x in scores}:raise SystemExit('identifier mismatch')
metrics=['single_camera','color_correct','background_correct','severe_artifact']
valid={'0','1','?'}
if any(s[m] not in valid for s in scores for m in metrics):raise SystemExit('all fields must be 0, 1, or ?')
joined=[{**lookup[s['blind_id']],**s} for s in scores]
summary={'by_arm':{},'paired':{},'notes':'8 prompts x 2 fixed seeds; paired rows share prompts; exploratory follow-up to a post-hoc camera result; one reviewer.'}
for arm in ('teacher','before','after'):
    armrows=[s for s in joined if s['arm']==arm]
    if len(armrows)!=16:raise SystemExit('bad arm counts')
    summary['by_arm'][arm]={m:dict(Counter(s[m] for s in armrows)) for m in metrics}
byid={arm:{s['id']:s for s in joined if s['arm']==arm} for arm in ('before','after')}
if len(byid['before'])!=16 or set(byid['before'])!=set(byid['after']):raise SystemExit('bad pairing')
for m in metrics:
    counts=Counter()
    for ident,b in byid['before'].items():
        x,y=b[m],byid['after'][ident][m]
        if '?' in (x,y): counts['uncertain']+=1
        elif x==y: counts['unchanged']+=1
        elif (m!='severe_artifact' and y=='1') or (m=='severe_artifact' and y=='0'): counts['improved']+=1
        else: counts['worsened']+=1
    summary['paired'][m]=dict(counts)
a.out.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
