import argparse, json, time, random, math, hashlib
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
parser.add_argument('--min-teacher-gap', type=float, required=True,
                    help='Predeclared minimum absolute R@1 advantage on validation, at least 0.05')
args=parser.parse_args()
if not math.isfinite(args.min_teacher_gap) or args.min_teacher_gap < 0.05:
    raise SystemExit('Specify an absolute R@1 advantage of at least 0.05 before evaluation')
root=args.workdir
if any((root/name).exists() for name in ('training-complete.json','model-supervised','model-distilled')):
    raise SystemExit('Output already contains a training run; use a fresh work directory')
imgroot=args.images
data=json.loads((root/'samples-private.json').read_text(encoding='utf8'))
for field in ('filename','sha256'):
    splits=[set(row[field] for row in data[name]) for name in ('train','val','test')]
    if splits[0]&splits[1] or splits[0]&splits[2] or splits[1]&splits[2]:
        raise SystemExit('Train, validation and test images overlap by '+field)
manifest=json.loads((root/'teacher-manifest.json').read_text(encoding='utf8'))
plan=json.loads((root/'teacher-plan.json').read_text(encoding='utf8'))
if plan['teacher'] != manifest['teacher'] or plan['student'] != args.student or plan['minimum_gap'] != args.min_teacher_gap:
    raise SystemExit('Student, teacher or threshold differ from the predeclared plan')
if plan['samples_sha256'] != manifest['samples_sha256']:
    raise SystemExit('The sample data differ from the predeclared plan')
if manifest['samples_sha256'] != hashlib.sha256((root/'samples-private.json').read_bytes()).hexdigest():
    raise SystemExit('Teacher vectors were generated from different samples; regenerate them')
if manifest['vectors_sha256'] != hashlib.sha256((root/'teacher-private.npz').read_bytes()).hexdigest():
    raise SystemExit('Teacher vectors changed since the development run; regenerate them')
teacher=np.load(root/'teacher-private.npz')
for split in ('train','val'):
    for side in ('text','image'):
        if len(teacher[split+'_'+side]) != len(data[split]):
            raise SystemExit('Teacher vectors do not match the selected samples')
        if not np.isfinite(teacher[split+'_'+side]).all():
            raise SystemExit('Teacher vectors contain non-finite values')
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

def teacher_ranks(text, images):
    text=F.normalize(torch.from_numpy(text).float(),dim=-1)
    images=F.normalize(torch.from_numpy(images).float(),dim=-1)
    order=torch.argsort(text@images.T,dim=1,descending=True,stable=True)
    return ((order==torch.arange(len(order))[:,None]).nonzero()[:,1]+1).tolist()

def teacher_gate(teacher_ranks, student_ranks, minimum):
    if len(teacher_ranks) != len(student_ranks) or len(student_ranks)<2:
        raise ValueError('Teacher and student must use the same nontrivial validation pool')
    differences=np.asarray(teacher_ranks)==1
    differences=differences.astype(float)-(np.asarray(student_ranks)==1).astype(float)
    rng=np.random.default_rng(20261107)
    bootstrap=rng.choice(differences,(10000,len(differences)),replace=True).mean(axis=1)
    gap=float(differences.mean()); lower,upper=map(float,np.quantile(bootstrap,[.025,.975]))
    return {'metric':'text-to-image R@1','teacher':float(np.mean(np.asarray(teacher_ranks)==1)),
            'student':float(np.mean(np.asarray(student_ranks)==1)),'gap':gap,
            'minimum_gap':minimum,'paired_bootstrap_95ci':[lower,upper],
            'passed':gap>=minimum and lower>0,'validation_queries':len(differences)}

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
baseline={'val':evaluate(model,'val')}
gate=teacher_gate(teacher_ranks(teacher['val_text'],teacher['val_image']),baseline['val']['ranks'],args.min_teacher_gap)
(root/'teacher-gate.json').write_text(json.dumps(gate,indent=2),encoding='utf8')
print(json.dumps(gate,indent=2),flush=True)
if not gate['passed']:
    raise SystemExit('Teacher advantage on validation did not pass the predeclared gate; no training or test evaluation performed')
del model; torch.cuda.empty_cache()
results={'config':{'seed':20261107,'batch_size':16,'epochs':2,'lr':2e-6,'weight_decay':0.01,
                   'contrastive_temperature':.07,'distillation_temperature':.05,'distillation_weight':.5,
                   'teacher':manifest['teacher'],'student':checkpoint,
                   'dtype':'FP32 weights with BF16 autocast',
                   'note':'same batches; full student training; one seed; fixed two epochs; no validation tuning'},
         'teacher_gate':gate,'baseline':baseline}
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
    results[condition]={'val':evaluate(model,'val'),
                        'training_seconds':duration,'losses':losses}
    model.save_pretrained(root/('model-'+condition),safe_serialization=True)
    (root/'student-results.json').write_text(json.dumps(results,indent=2),encoding='utf8')
    del opt,model; torch.cuda.empty_cache()
# All training decisions are fixed before the held-out test pool is scored.
model=get_model(); results['baseline']['test']=evaluate(model,'test'); del model; torch.cuda.empty_cache()
for condition in ('supervised','distilled'):
    model=CLIPModel.from_pretrained(root/('model-'+condition),local_files_only=True,dtype=torch.float32).to('cuda')
    results[condition]['test']=evaluate(model,'test'); del model; torch.cuda.empty_cache()
results['total_seconds']=time.perf_counter()-start
results['cuda_peak_allocated']=torch.cuda.max_memory_allocated()
(root/'student-results.json').write_text(json.dumps(results,indent=2),encoding='utf8')
(root/'training-complete.json').write_text(json.dumps({'samples_sha256':manifest['samples_sha256'],'gate':gate}),encoding='utf8')
print(json.dumps({k:{s:{m:vals[m] for m in ('r1','r5','mrr')} for s,vals in v.items() if s in ('val','test')}
                  for k,v in results.items() if k in ('baseline','supervised','distilled')},indent=2))
