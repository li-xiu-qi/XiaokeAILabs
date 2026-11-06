"""Synthetic-only ColSmol baseline/LoRA training, with guarded memory."""
import argparse,json,os,random,threading,time,math
from pathlib import Path

def check():
 for line in open('/proc/meminfo'):
  if line.startswith('MemAvailable:') and int(line.split()[1])<50*1024*1024:raise RuntimeError('MemAvailable < 50 GiB')
def watch():
 while True:
  try:check()
  except RuntimeError as e:print(e,flush=True);os._exit(3)
  time.sleep(1)
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--base',required=True);p.add_argument('--adapter',required=True);p.add_argument('--out',required=True,type=Path);p.add_argument('--steps',type=int,default=8);p.add_argument('--smoke',action='store_true');args=p.parse_args()
 args.out.mkdir(parents=True,exist_ok=False)
 check();threading.Thread(target=watch,daemon=True).start()
 import torch
 from PIL import Image
 from peft import PeftModel
 from colpali_engine.models import ColIdefics3,ColIdefics3Processor
 from colpali_engine.loss.late_interaction_losses import ColbertLoss
 torch.manual_seed(20261107);torch.cuda.set_per_process_memory_fraction(.15);rng=random.Random(20261107)
 records=json.loads((args.root/'records.json').read_text());train=[x for x in records if x['split']=='train'];test=[x for x in records if x['split']==('validation' if args.smoke else 'test')]
 assert not {x['entity'] for x in train}&{x['entity'] for x in test}
 model=ColIdefics3.from_pretrained(args.base,torch_dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='eager',local_files_only=True)
 model=PeftModel.from_pretrained(model,args.adapter,is_trainable=True,local_files_only=True)
 proc=ColIdefics3Processor.from_pretrained(args.adapter,local_files_only=True)
 def eval_model():
  model.eval();images=sorted({r['image'] for r in test}); rows=test[:4] if args.smoke else test
  imgs=images[:4] if args.smoke else images
  if args.smoke: rows=[r for r in test if r['image'] in imgs][:4]
  embs=[];queries=[]
  with torch.inference_mode():
   for image in imgs:
    check()
    with Image.open(args.root/image) as im: inputs=proc.process_images([im.convert('RGB')]).to(model.device)
    embs.append(model(**inputs)[0].cpu())
   for r in rows:
    inputs=proc.process_queries([r['query']]).to(model.device);queries.append(model(**inputs)[0].cpu())
   scores=proc.score_multi_vector(queries,embs,batch_size=1)
  ranks=[]
  for i,r in enumerate(rows):
   order=torch.argsort(scores[i],descending=True).tolist(); ranks.append(order.index(imgs.index(r['image']))+1)
  return {'n':len(rows),'candidates':len(imgs),'r1':sum(x==1 for x in ranks)/len(ranks),'r5':sum(x<=5 for x in ranks)/len(ranks),'ranks':ranks}
 baseline=eval_model();print('baseline',baseline['r1'],baseline['r5'],flush=True)
 model.train(); opt=torch.optim.AdamW([x for x in model.parameters() if x.requires_grad],lr=1e-5);loss_fn=ColbertLoss(temperature=.02)
 by_image={im:[x for x in train if x['image']==im] for im in {x['image'] for x in train}}
 train_images=sorted(by_image); history=[]
 for step in range(2 if args.smoke else args.steps):
  check(); batch=[rng.choice(by_image[i]) for i in rng.sample(train_images,2)]
  ims=[]
  for r in batch:
   with Image.open(args.root/r['image']) as im:ims.append(im.convert('RGB').copy())
  iq=proc.process_queries([r['query'] for r in batch]).to(model.device);idoc=proc.process_images(ims).to(model.device)
  opt.zero_grad(set_to_none=True);q=model(**iq);doc=model(**idoc);loss=loss_fn(q,doc)
  if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
  loss.backward();opt.step();history.append(float(loss.detach()));print('step',step,'loss',history[-1],flush=True)
 after=eval_model()
 (args.out/'results.json').write_text(json.dumps({'baseline':baseline,'trained':after,'loss':history,'steps':len(history),'batch_queries':2,'learning_rate':1e-5,'seed':20261107,'peak_allocated':torch.cuda.max_memory_allocated()},indent=2))
 model.save_pretrained(str(args.out/'adapter'));print('done',after['r1'],after['r5'],flush=True)
if __name__=='__main__':main()
