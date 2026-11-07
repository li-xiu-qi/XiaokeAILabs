"""Score exactly the fixed in-batch training pairs with Qwen; never load test data."""
import argparse,json,random,os,sys,threading,math,time
from pathlib import Path
from teacher_binding import image_hashes

def check():
 for s in open('/proc/meminfo'):
  if s.startswith('MemAvailable:') and int(s.split()[1])<50*1024*1024:raise RuntimeError('memory < 50 GiB')
def watch():
 while True:
  try:check()
  except RuntimeError as e:print(e,flush=True);os._exit(3)
  time.sleep(1)
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--steps',type=int,default=8);p.add_argument('--model',required=True);args=p.parse_args()
 if args.out.exists(): p.error('output already exists; choose a new path')
 check();threading.Thread(target=watch,daemon=True).start();import torch
 sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'multimodal-reranking'/'scripts'));from qwen3_vl_reranker import Qwen3VLReranker
 torch.cuda.set_per_process_memory_fraction(.15);rng=random.Random(20261107)
 rows=json.loads((args.root/'records.json').read_text());train=[r for r in rows if r['split']=='train'];by={i:[r for r in train if r['image']==i] for i in {r['image'] for r in train}};imgs=sorted(by)
 batches=[[rng.choice(by[i]) for i in rng.sample(imgs,2)] for _ in range(args.steps)]
 hashes=image_hashes(args.root,batches)
 model=Qwen3VLReranker(args.model,min_pixels=4096,max_pixels=65536,dtype=torch.bfloat16);out=[];start=time.perf_counter()
 for step,batch in enumerate(batches):
  assert len({r['image'] for r in batch})==2 and all(r['split']=='train' for r in batch)
  scores=[]
  for r in batch:
   check();x=model.process({'query':{'text':r['query']},'documents':[{'image':str(args.root/t['image'])} for t in batch]});assert len(x)==2 and all(math.isfinite(z) for z in x);scores.append(x)
  out.append({'batch':batch,'scores':scores});print('step',step,'scores',scores,flush=True)
 if hashes != image_hashes(args.root,batches):raise ValueError('Training images changed during teacher scoring')
 with args.out.open('x',encoding='utf8') as f: f.write(json.dumps({'teacher':'Qwen3-VL-Reranker-2B','license':'Apache-2.0','seed':20261107,'steps':args.steps,'batches':out,'image_sha256':hashes,'seconds':time.perf_counter()-start,'peak_allocated':torch.cuda.max_memory_allocated()},indent=2))
if __name__=='__main__':main()
