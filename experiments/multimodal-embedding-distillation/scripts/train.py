import argparse, json, time, random, math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, AutoProcessor

def check():
    for s in open('/proc/meminfo'):
        if s.startswith('MemAvailable:') and int(s.split()[1]) < 50*1024*1024:
            raise RuntimeError('MemAvailable below 50 GiB')

parser=argparse.ArgumentParser()
parser.add_argument('--workdir', type=Path, required=True)
parser.add_argument('--images', type=Path, required=True)
parser.add_argument('--student', required=True)
args=parser.parse_args()
root=args.workdir
imgroot=args.images
data=json.loads((root/'samples-private.json').read_text(encoding='utf8'))
teacher=np.load(root/'teacher-private.npz')
torch.cuda.set_per_process_memory_fraction(.20)
checkpoint=args.student
processor=AutoProcessor.from_pretrained(checkpoint,local_files_only=True)

def input_batch(rows):
    images=[]
    for row in rows:
        with Image.open(imgroot/row['filename']) as im: images.append(im.convert('RGB'))
    return processor(text=[r['caption'] for r in rows],images=images,padding=True,truncation=True,max_length=77,return_tensors='pt').to('cuda')

def vectors(model, rows):
    txt=[]; img=[]
    model.eval()
    with torch.inference_mode():
        for i in range(0,len(rows),8):
            check(); batch=input_batch(rows[i:i+8])
            with torch.autocast('cuda',dtype=torch.bfloat16): out=model(**batch)
            txt.append(out.text_embeds.float().cpu()); img.append(out.image_embeds.float().cpu())
    return F.normalize(torch.cat(txt),dim=-1),F.normalize(torch.cat(img),dim=-1)

def evaluate(model,split):
    text,images=vectors(model,data[split]); sim=text@images.T
    # Stable sort is unnecessary for continuous float32 similarities unless ties occur.
    order=torch.argsort(sim,dim=1,descending=True,stable=True)
    ranks=(order==torch.arange(len(order))[:,None]).nonzero()[:,1]+1
    return {'r1':float((ranks<=1).float().mean()),'r5':float((ranks<=5).float().mean()),
            'mrr':float((1/ranks.float()).mean()),'ranks':ranks.tolist()}

def get_model():
    check()
    return CLIPModel.from_pretrained(checkpoint,local_files_only=True,dtype=torch.float32).to('cuda')

def contrast(sim):
    labels=torch.arange(sim.shape[0],device=sim.device)
    return .5*(F.cross_entropy(sim/.07,labels)+F.cross_entropy(sim.T/.07,labels))

def kd(student,soft):
    temp=.05
    return .5*(F.kl_div(F.log_softmax(student/temp,dim=1),F.softmax(soft/temp,dim=1),reduction='batchmean')+
               F.kl_div(F.log_softmax(student.T/temp,dim=1),F.softmax(soft.T/temp,dim=1),reduction='batchmean'))

torch.manual_seed(20261107); np.random.seed(20261107); random.seed(20261107)
start=time.perf_counter(); model=get_model()
baseline={'val':evaluate(model,'val'),'test':evaluate(model,'test')}
del model; torch.cuda.empty_cache()
results={'config':{'seed':20261107,'batch_size':16,'epochs':2,'lr':2e-6,'weight_decay':0.01,
                   'contrastive_temperature':.07,'distillation_temperature':.05,'distillation_weight':.5,
                   'teacher':'jinaai/jina-embeddings-v5-omni-small','student':'BAAI/BGE-VL-base',
                   'student_parameters':149620737,'teacher_license':'CC BY-NC 4.0',
                   'dtype':'FP32 weights with BF16 autocast',
                   'note':'same batches; full student training; one seed; fixed two epochs; no validation tuning'},
         'baseline':baseline}
nptext=teacher['train_text']; npimg=teacher['train_image']
teacher_text=F.normalize(torch.from_numpy(nptext).to('cuda'),dim=-1)
teacher_img=F.normalize(torch.from_numpy(npimg).to('cuda'),dim=-1)
for condition in ('supervised','distilled'):
    torch.manual_seed(20261107); random.seed(20261107)
    model=get_model(); model.train(); opt=torch.optim.AdamW(model.parameters(),lr=2e-6,weight_decay=.01)
    order_rng=random.Random(20261107); losses=[]; training_start=time.perf_counter()
    for epoch in range(2):
        order=list(range(len(data['train']))); order_rng.shuffle(order)
        for k in range(0,len(order),16):
            check(); ids=order[k:k+16]; batch=input_batch([data['train'][i] for i in ids])
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                output=model(**batch)
                sim=output.text_embeds.float()@output.image_embeds.float().T
                base=contrast(sim)
                target=teacher_text[ids]@teacher_img[ids].T
                d=kd(sim,target)
                loss=base if condition=='supervised' else .5*base+.5*d
            if not torch.isfinite(loss): raise RuntimeError('nonfinite loss')
            loss.backward(); opt.step(); losses.append(float(loss.detach()))
            if len(losses)%16==0: print(condition,epoch+1,len(losses),losses[-1],flush=True)
    torch.cuda.synchronize(); duration=time.perf_counter()-training_start
    results[condition]={'val':evaluate(model,'val'),'test':evaluate(model,'test'),
                        'training_seconds':duration,'losses':losses}
    model.save_pretrained(root/('model-'+condition),safe_serialization=True)
    (root/'student-results.json').write_text(json.dumps(results,indent=2),encoding='utf8')
    del opt,model; torch.cuda.empty_cache()
results['total_seconds']=time.perf_counter()-start
results['cuda_peak_allocated']=torch.cuda.max_memory_allocated()
(root/'student-results.json').write_text(json.dumps(results,indent=2),encoding='utf8')
print(json.dumps({k:{s:{m:vals[m] for m in ('r1','r5','mrr')} for s,vals in v.items() if s in ('val','test')}
                  for k,v in results.items() if k in ('baseline','supervised','distilled')},indent=2))
