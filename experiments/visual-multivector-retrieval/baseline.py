import argparse, json, math, os, threading, time
from pathlib import Path

def check():
    for line in open('/proc/meminfo'):
        if line.startswith('MemAvailable:') and int(line.split()[1]) < 50*1024*1024:
            raise RuntimeError('available memory below 50 GiB')
def watch():
    while True:
        try: check()
        except RuntimeError as e: print(e,flush=True); os._exit(3)
        time.sleep(1)

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--model',required=True);p.add_argument('--base',required=True);p.add_argument('--out',required=True);p.add_argument('--count',type=int,default=100);args=p.parse_args()
    if Path(args.out).exists(): p.error('output already exists; choose a new path')
    check();threading.Thread(target=watch,daemon=True).start()
    import torch
    from datasets import load_from_disk
    from colpali_engine.models import ColIdefics3,ColIdefics3Processor
    from peft import PeftModel
    torch.cuda.set_per_process_memory_fraction(.15)
    data=load_from_disk(args.data); rows=[];seen=set()
    for r in data:
        key=r['image_filename']
        if key not in seen:
            seen.add(key); rows.append(r)
            if len(rows)==args.count:break
    assert len(rows)==args.count
    model=ColIdefics3.from_pretrained(args.base,torch_dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='eager',local_files_only=True)
    model=PeftModel.from_pretrained(model,args.model,local_files_only=True).eval()
    proc=ColIdefics3Processor.from_pretrained(args.model,local_files_only=True)
    ims=[];qs=[];start=time.perf_counter()
    with torch.inference_mode():
        for i,r in enumerate(rows):
            check(); x=proc.process_images([r['image'].convert('RGB')]).to(model.device);ims.append(model(**x)[0].cpu())
            if i%10==0:print('images',i,flush=True)
        for i,r in enumerate(rows):
            check();x=proc.process_queries([r['query']]).to(model.device);qs.append(model(**x)[0].cpu())
        scores=proc.score_multi_vector(qs,ims,batch_size=1).cpu()
    rankings=torch.argsort(scores,dim=1,descending=True); records=[]
    for i,row in enumerate(rankings):
        pos=(row==i).nonzero(as_tuple=True)[0].item()+1
        records.append({'row':i,'rank':pos,'r1':int(pos<=1),'r5':int(pos<=5),'ndcg5':1/math.log2(pos+1) if pos<=5 else 0.})
    result={'n':len(rows),'r1':sum(x['r1'] for x in records)/len(rows),'r5':sum(x['r5'] for x in records)/len(rows),'ndcg5':sum(x['ndcg5'] for x in records)/len(rows),'seconds':time.perf_counter()-start,'peak_bytes':torch.cuda.max_memory_allocated(),'library':'colpali-engine 0.3.13','precision':'bfloat16','records':records}
    with Path(args.out).open('x', encoding='utf8') as f: json.dump(result, f, indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)
if __name__=='__main__':main()
