"""Verify all frozen pairs; downsample teacher originals without filtering."""
import argparse,hashlib,json,shutil
from pathlib import Path
from PIL import Image
p=argparse.ArgumentParser(); p.add_argument('--prompts',type=Path,required=True); p.add_argument('--source',type=Path,required=True); p.add_argument('--out',type=Path,required=True); p.add_argument('--size',type=int,choices=[512,1024],required=True); a=p.parse_args()
expected=[json.loads(x) for x in a.prompts.read_text(encoding='utf-8').splitlines()]
actual=[json.loads(x) for x in (a.source/'metadata.jsonl').read_text(encoding='utf-8').splitlines()]
rows={r['id']:r for r in expected}; indexed={r['id']:r for r in actual}
if len(rows)!=128 or len(expected)!=128 or len(indexed)!=128 or len(actual)!=128 or set(rows)!=set(indexed): raise SystemExit('expected 128 unique matching IDs')
if a.out.exists(): raise SystemExit('refusing overwrite')
for id,row in rows.items():
    r=indexed[id]; filename=id+'.png'
    if Path(filename).name!=filename or r['file_name']!=filename or r['prompt']!=row['prompt'] or r['text']!=row['prompt'] or r['seed']!=row['seed']: raise SystemExit('mismatched metadata')
    with Image.open(a.source/filename) as im:
        if im.size!=(a.size,a.size): raise SystemExit('incorrect image size')
        im.verify()
a.out.mkdir(parents=True); manifest=[]
for id,row in rows.items():
    r=indexed[id]; src=a.source/(id+'.png'); dst=a.out/src.name
    if a.size==512: shutil.copyfile(src,dst)
    else:
        with Image.open(src) as im: im.convert('RGB').resize((512,512),Image.Resampling.LANCZOS).save(dst)
    manifest.append({'id':id,'original_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'training_sha256':hashlib.sha256(dst.read_bytes()).hexdigest()})
(a.out/'metadata.jsonl').write_text(''.join(json.dumps(indexed[id],ensure_ascii=False)+'\n' for id in rows),encoding='utf-8')
(a.out/'image-sha256.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print('validated and prepared 128 image-caption pairs; no quality filtering')
