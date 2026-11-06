import argparse,hashlib,json,random,threading,time,os
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont

def check():
    for line in open('/proc/meminfo'):
        if line.startswith('MemAvailable:') and int(line.split()[1])<50*1024*1024:
            raise RuntimeError('available memory below 50 GiB')
def watch():
    while True:
        try: check()
        except RuntimeError as exc: print(exc,flush=True);os._exit(3)
        time.sleep(1)

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--base',type=Path,required=True);p.add_argument('--supervised',type=Path,required=True);p.add_argument('--distilled',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    check();threading.Thread(target=watch,daemon=True).start()
    import torch
    from peft import PeftModel
    from colpali_engine.models import ColIdefics3,ColIdefics3Processor
    rng=random.Random(20261108);codes=rng.sample(range(100000,999999),64);vals=iter(rng.sample(sorted(set(range(100000,999999))-set(codes)),192))
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',25);bold=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',35)
    pages=[];rows=[]; labels=['Revenue','Units shipped','Inventory']
    for i,code in enumerate(codes):
        im=Image.new('RGB',(900,700),'white');d=ImageDraw.Draw(im);d.rectangle((0,0,900,95),fill='#143a43');d.text((35,25),'OPERATIONS REPORT',font=bold,fill='white');d.text((45,125),f'Site ID: ST-{code}',font=bold,fill='#193b3a')
        numbers=[next(vals) for _ in labels]
        for j,(label,n) in enumerate(zip(labels,numbers)):
            y=240+j*90;d.rectangle((45,y-10,850,y+55),outline='#bfd1ce',width=2);d.text((65,y),label,font=font,fill='#1b302f');d.text((580,y),str(n),font=font,fill='#1b302f')
            rows.append({'query':f'Which report lists {label.lower()} as {n}?','page':i,'value':n,'entity':f'ST-{code}'})
        path=a.out/f'page-{i:03}.png';im.save(path);pages.append(path)
    old=json.loads(a.source.read_text());assert not ({r['entity'] for r in old}&{r['entity'] for r in rows});assert not ({r['value'] for r in old}&{r['value'] for r in rows})
    oldhash={hashlib.sha256(p.read_bytes()).hexdigest() for p in a.source.parent.glob('*/*.png')};assert oldhash, 'no original PNG pages found to compare';hashes=[hashlib.sha256(p.read_bytes()).hexdigest() for p in pages];assert len(set(hashes))==64 and not oldhash.intersection(hashes)
    (a.out/'manifest.json').write_text(json.dumps({'seed':20261108,'records':rows,'page_sha256':hashes},indent=2))
    print('data prepared: 64 pages 192 queries, distinct entity values and image hashes',flush=True)
    torch.cuda.set_per_process_memory_fraction(.15)
    proc=ColIdefics3Processor.from_pretrained(str(a.model),local_files_only=True)
    results={}
    for name,adapter in [('original',a.model),('supervised',a.supervised),('distilled',a.distilled)]:
        check();m=ColIdefics3.from_pretrained(str(a.base),torch_dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='eager',local_files_only=True);m=PeftModel.from_pretrained(m,str(adapter),local_files_only=True);m.eval();emb=[];qs=[]
        with torch.inference_mode():
            for path in pages:
                check()
                with Image.open(path) as image: x=proc.process_images([image.convert('RGB')]).to(m.device)
                emb.append(m(**x)[0].cpu())
            for r in rows:
                check();x=proc.process_queries([r['query']]).to(m.device);qs.append(m(**x)[0].cpu())
            scores=proc.score_multi_vector(qs,emb,batch_size=1)
        ranks=[torch.argsort(scores[i],descending=True).tolist().index(r['page'])+1 for i,r in enumerate(rows)]
        results[name]={'ranks':ranks,'r1_count':sum(x==1 for x in ranks),'r5_count':sum(x<=5 for x in ranks),'n':len(rows),'candidates':len(pages)}
        print(name,results[name]['r1_count'],results[name]['r5_count'],flush=True)
        (a.out/'results.partial.json').write_text(json.dumps(results,indent=2));del m,emb,qs,scores;torch.cuda.empty_cache()
    results['peak_allocated_bytes']=torch.cuda.max_memory_allocated();(a.out/'results.json').write_text(json.dumps(results,indent=2))
if __name__=='__main__':main()
