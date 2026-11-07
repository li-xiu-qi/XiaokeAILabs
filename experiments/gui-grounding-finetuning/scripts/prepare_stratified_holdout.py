import argparse, hashlib, io, json, random
from pathlib import Path
from collections import defaultdict, Counter
import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API='https://datasets-server.huggingface.co/rows'
def dhash(image):
    p=list(image.convert('L').resize((9,8)).getdata())
    return sum((p[y*9+x]>p[y*9+x+1]) << (y*8+x) for y in range(8) for x in range(8))

def main():
    parser=argparse.ArgumentParser(description='Build a local, stratified ScreenSpot holdout without modifying older runs')
    parser.add_argument('--training',required=True,type=Path,help='directory containing GroundGUI training manifest and images')
    parser.add_argument('--previous',required=True,type=Path,help='directory containing previous ScreenSpot manifest and images')
    parser.add_argument('--output',required=True,type=Path,help='new private output directory')
    parser.add_argument('--per-stratum',type=int,default=3)
    parser.add_argument('--seed',type=int,default=20261107)
    args=parser.parse_args()
    if args.per_stratum<1: parser.error('--per-stratum must be positive')
    if args.output.exists(): raise FileExistsError(args.output)
    blocked_sha=set(); blocked_hash=[]
    for directory in (args.training,args.previous):
        for row in json.loads((directory/'manifest.json').read_text(encoding='utf8')):
            raw=(directory/row['file']).read_bytes(); actual=hashlib.sha256(raw).hexdigest()
            if actual!=row['sha256']: raise ValueError('Reference image hash mismatch')
            blocked_sha.add(actual); blocked_hash.append(dhash(Image.open(io.BytesIO(raw))))
    session=requests.Session(); session.mount("https://", HTTPAdapter(max_retries=Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])))
    candidates=[]; offset=0
    while True:
        res=session.get(API,params={'dataset':'bevaya/ScreenSpot','config':'default','split':'test','offset':offset,'length':100},timeout=90); res.raise_for_status()
        data=res.json(); rows=data['rows']; candidates.extend(rows)
        offset+=len(rows)
        if offset>=data['num_rows_total'] or not rows: break
    groups=defaultdict(list)
    for item in candidates:
        row=item['row']; groups[(row['data_source'],row['data_type'])].append(item)
    rng=random.Random(args.seed)
    args.output.mkdir(parents=True)
    selected=[]; rejected=Counter()
    for stratum in sorted(groups):
        items=groups[stratum]; rng.shuffle(items); n=0
        for item in items:
            row=item['row']; img=row['image']; response=session.get(img['src'],timeout=90); response.raise_for_status()
            raw=response.content; digest=hashlib.sha256(raw).hexdigest()
            if digest in blocked_sha: rejected['exact']+=1; continue
            image=Image.open(io.BytesIO(raw)).convert('RGB'); sig=dhash(image)
            if any((sig ^ x).bit_count()<=5 for x in blocked_hash): rejected['near']+=1; continue
            width,height=image.size; bbox=row['bbox']
            if (width,height)!=(img['width'],img['height']) or len(bbox)!=4 or not (0<=bbox[0]<bbox[2]<=1 and 0<=bbox[1]<bbox[3]<=1): raise ValueError('Invalid image dimensions or bounding box')
            filename=f"{item['row_idx']:04d}.png"; (args.output/filename).write_bytes(raw)
            selected.append({'row_idx':item['row_idx'],'file':filename,'sha256':digest,'bbox_xyxy_normalized':bbox,'instruction':row['instruction'],'data_type':row['data_type'],'data_source':row['data_source'],'width':width,'height':height})
            blocked_sha.add(digest); blocked_hash.append(sig); n+=1
            if n>=args.per_stratum: break
        print(f'{stratum}: selected {n}/{args.per_stratum}',flush=True)
    (args.output/'manifest.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({'dataset':'bevaya/ScreenSpot','source_api_revision':'unavailable','seed':args.seed,'requested_per_stratum':args.per_stratum,'selected':len(selected),'rejected':dict(rejected),'strata':{f'{a}/{b}':sum(r['data_source']==a and r['data_type']==b for r in selected) for a,b in sorted(groups)}},indent=2))
if __name__=='__main__': main()
