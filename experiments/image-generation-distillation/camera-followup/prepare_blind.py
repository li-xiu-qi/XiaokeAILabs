import csv, hashlib, json, random, shutil
from pathlib import Path
import argparse

p=argparse.ArgumentParser()
p.add_argument('--source', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--key', type=Path, required=True)
args=p.parse_args()
if args.out.exists() or args.key.exists(): raise SystemExit('refusing to overwrite')
if args.key.resolve().is_relative_to(args.out.resolve()): raise SystemExit('mapping must be outside review directory')
rows=[json.loads(x) for x in (args.source/'prompts.jsonl').read_text(encoding='utf-8-sig').splitlines()]
if len(rows)!=16 or len({r['id'] for r in rows})!=16: raise SystemExit('expected 16 unique records')
items=[(arm,row) for arm in ('teacher','before','after') for row in rows]
random.Random(442915).shuffle(items)
args.out.mkdir(parents=True)
(args.out/'images').mkdir()
key=[]
with (args.out/'review.csv').open('x',newline='',encoding='utf-8') as f:
    writer=csv.writer(f)
    writer.writerow(['blind_id','prompt','single_camera','color_correct','background_correct','severe_artifact','notes'])
    for n,(arm,row) in enumerate(items):
        identifier=f'item-{n+1:03d}'
        source=args.source/arm/(row['id']+'.png')
        if not source.is_file(): raise SystemExit(f'missing {source}')
        dest=args.out/'images'/(identifier+'.png')
        shutil.copyfile(source,dest)
        sha=hashlib.sha256(dest.read_bytes()).hexdigest()
        key.append({'blind_id':identifier,'arm':arm,'id':row['id'],'seed':row['seed'],'sha256':sha})
        writer.writerow([identifier,row['prompt'],'','','','',''])
args.key.parent.mkdir(parents=True, exist_ok=True)
args.key.write_text(json.dumps(key,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(f'Created {len(key)} blinded images')
