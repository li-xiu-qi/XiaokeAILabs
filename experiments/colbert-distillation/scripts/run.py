import os, json, random, time, threading, argparse, math, gc, hashlib
from pathlib import Path
import numpy as np
import torch
from pylate import models, losses, rank

os.environ['HF_HUB_OFFLINE']='1'
ROOT=SOURCE=DATA=MODEL=None
SEED=20261107

def check():
    for line in open('/proc/meminfo'):
        if line.startswith('MemAvailable:') and int(line.split()[1]) < 50*1024*1024:
            raise RuntimeError('available memory below 50 GiB')

def watchdog():
    while True:
        try: check()
        except RuntimeError as e: print(str(e),flush=True); os._exit(3)
        time.sleep(1)

def read_jsonl(path):
    return [json.loads(x) for x in path.read_text(encoding='utf8').splitlines()]

def save(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8');tmp.replace(path)

def setup(smoke):
    docs={x['_id']:(x.get('title','')+' '+x['text']).strip() for x in read_jsonl(DATA/'corpus.jsonl')}
    queries={x['_id']:x['text'] for x in read_jsonl(DATA/'queries.jsonl')}
    manifest=json.loads((SOURCE/'manifest.json').read_text());teacher=json.loads((SOURCE/'teacher.json').read_text())
    train=manifest['train'];dev=manifest['dev'];scores=teacher['scores']
    assert len(train)==128 and len(dev)>=2
    assert all(len(row['candidates'])==4 and len(scores[row['id']])==4 for row in train)
    assert all(len(row['candidates'])==30 and len(scores[row['id']])==30 for row in dev)
    return docs,queries,train[:2] if smoke else train,dev[:2] if smoke else dev,scores,manifest,teacher

def teacher_dev_metrics(rows,scores):
    result=[]
    for row in rows:
        relevant=set(map(str,row['relevant']))
        ranked=sorted(zip(row['candidates'],scores[row['id']]),key=lambda x:(-x[1],str(x[0])))
        labels=[int(str(d) in relevant) for d,_ in ranked[:10]]
        ideal=sum(1/math.log2(i+2) for i in range(min(10,len(relevant))))
        result.append({'query_id':row['id'],'ndcg10':sum(v/math.log2(i+2) for i,v in enumerate(labels))/ideal if ideal else 0.})
    return result

def evaluate(model,docs,queries,test):
    model.eval(); records=[];started=time.perf_counter()
    with torch.inference_mode():
        for n,row in enumerate(test):
            check();ids=row['candidates']
            q=model.encode([queries[row['id']]],is_query=True,show_progress_bar=False,convert_to_numpy=True)
            d=model.encode([docs[i] for i in ids],is_query=False,batch_size=16,show_progress_bar=False,convert_to_numpy=True)
            result=rank.rerank(documents_ids=[ids],queries_embeddings=q,documents_embeddings=[d],device='cuda')[0]
            ordered=[str(x['id']) for x in result];rel=set(row['relevant'])
            labels=[int(x in rel) for x in ordered[:10]]
            ideal=sum(1/math.log2(i+2) for i in range(min(10,len(rel))))
            records.append({'query_id':row['id'],'order':ordered,'scores':[float(x['score']) for x in result],
                            'ndcg10':sum(v/math.log2(i+2) for i,v in enumerate(labels))/ideal if ideal else 0.,
                            'mrr10':next((1/(i+1) for i,v in enumerate(labels) if v),0.)})
            if n%50==0: print('evaluated',n,'/',len(test),flush=True)
    torch.cuda.synchronize()
    return records,time.perf_counter()-started

def train(model,docs,queries,rows,teacher,mode,smoke):
    model.train();loss_fn=losses.Distillation(model,normalize_scores=False)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-6);rng=random.Random(SEED);permutation=[]
    started=time.perf_counter();steps=2 if smoke else 128
    for step in range(steps):
        check()
        while len(permutation)<4:
            order=list(range(len(rows)));rng.shuffle(order);permutation.extend(order)
        batch=[rows[permutation.pop(0)] for _ in range(4)]
        q=model.tokenize([queries[r['id']] for r in batch],is_query=True)
        d=model.tokenize([docs[did] for r in batch for did in r['candidates']],is_query=False)
        features=[{k:v.to('cuda') if isinstance(v,torch.Tensor) else v for k,v in x.items()} for x in (q,d)]
        labels=torch.tensor(([teacher[r['id']] for r in batch] if mode=='distilled' else [[50.,0.,0.,0.]]*4),device='cuda')
        if mode=='distilled':labels=labels/0.05
        loss=loss_fn(features,labels)
        if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        if step%20==0: print(mode,step,'loss',float(loss.detach().cpu()),flush=True)
    torch.cuda.synchronize();return {'steps':steps,'batch_queries':4,'lr':3e-6,'seconds':time.perf_counter()-started}

def main():
    global ROOT,SOURCE,DATA,MODEL
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--output',required=True,type=Path);p.add_argument('--source',required=True,type=Path);p.add_argument('--data',required=True,type=Path);p.add_argument('--model',required=True,type=Path);p.add_argument('--pools',required=True,type=Path);p.add_argument('--min-teacher-gap',required=True,type=float,help='Prespecified absolute dev nDCG@10 margin, at least 0.05 for this experiment');args=p.parse_args()
    if not math.isfinite(args.min_teacher_gap) or args.min_teacher_gap < .05: p.error('--min-teacher-gap must be at least 0.05')
    ROOT,SOURCE,DATA,MODEL=args.output,args.source,args.data,args.model
    root=ROOT/('smoke' if args.smoke else 'formal')
    if root.exists() and any(root.iterdir()): raise SystemExit('Use a fresh output directory to avoid overwriting results')
    root.mkdir(parents=True,exist_ok=True)
    docs,queries,train_rows,dev,teacher,manifest,teacher_meta=setup(args.smoke)
    if manifest.get('min_teacher_gap')!=args.min_teacher_gap: raise SystemExit('Gap differs from the threshold declared before teacher scoring')
    digest=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    for key,path in [('pools_sha256',args.pools),('corpus_sha256',DATA/'corpus.jsonl'),('queries_sha256',DATA/'queries.jsonl')]:
        if manifest.get(key)!=digest(path): raise SystemExit('Source data changed or manifest predates the development gate: '+key)
    if args.smoke:
        print('Smoke: development input and manifest checked; no model evaluation, training, or test access.',flush=True);return
    if teacher_meta.get('synthetic_only'): raise SystemExit('Synthetic teacher is for smoke tests only')
    if any(not math.isfinite(x) for vals in teacher.values() for x in vals): raise SystemExit('Nonfinite teacher scores')
    # The threshold and input fingerprints are recorded before evaluating either model.
    gate_file=root/'teacher-gate.json'
    if gate_file.exists(): raise SystemExit('Output already contains a gate; use a fresh output directory')
    save(root/'plan.json',{'min_teacher_gap':args.min_teacher_gap,'seed':SEED,'bootstrap_samples':10000,'bootstrap_lower_bound_gt':0,'input_manifest_sha256':digest(SOURCE/'manifest.json'),'teacher_scores_sha256':digest(SOURCE/'teacher.json'),'pools_sha256':digest(args.pools)})
    check();threading.Thread(target=watchdog,daemon=True).start();torch.cuda.set_per_process_memory_fraction(.15)
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    model=models.ColBERT(model_name_or_path=str(MODEL),device='cuda',query_length=32,document_length=180,local_files_only=True)
    student_dev,dev_seconds=evaluate(model,docs,queries,dev)
    del model;gc.collect();torch.cuda.empty_cache()
    teachers=teacher_dev_metrics(dev,teacher); student_by_id={r['query_id']:r['ndcg10'] for r in student_dev}
    if set(student_by_id)!={r['query_id'] for r in teachers}: raise SystemExit('Development query mismatch')
    diff=np.array([r['ndcg10']-student_by_id[r['query_id']] for r in teachers])
    means=np.random.default_rng(SEED).choice(diff,(10000,len(diff)),replace=True).mean(axis=1)
    low,high=map(float,np.quantile(means,[.025,.975])); gap=float(diff.mean())
    passed=gap>=args.min_teacher_gap and low>0
    save(gate_file,{'teacher_ndcg10':float(np.mean([x['ndcg10'] for x in teachers])),'student_ndcg10':float(np.mean([x['ndcg10'] for x in student_dev])),'gap':gap,'paired_bootstrap_95ci':[low,high],'passed':passed,'teacher_per_query':teachers,'student_per_query':student_dev,'student_eval_seconds':dev_seconds})
    if not passed: raise SystemExit('Teacher dev advantage gate failed; no training or test evaluation')
    for mode in ('supervised','distilled'):
        random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
        model=models.ColBERT(model_name_or_path=str(MODEL),device='cuda',query_length=32,document_length=180,local_files_only=True)
        check();parameters=sum(x.numel() for x in model.parameters())
        timing=train(model,docs,queries,train_rows,teacher,mode,False)
        model.save_pretrained(str(root/(mode+'-model')))
        save(root/(mode+'-training.json'),{'model':str(MODEL),'parameter_count':parameters,'training':timing})
        del model;gc.collect();torch.cuda.empty_cache()
    # Test candidates and labels are read only after both training runs finish.
    if digest(args.pools)!=manifest['pools_sha256']: raise SystemExit('Candidate pool changed during training')
    pools=json.loads(args.pools.read_text(encoding='utf8'))
    if set(pools['train']) & (set(pools['dev'])|set(pools['test'])) or set(pools['dev']) & set(pools['test']): raise SystemExit('Query partitions overlap')
    test=[{'id':q,'candidates':[str(x[0]) for x in row['pool']],'relevant':[str(x) for x in row['relevant']]} for q,row in pools['test'].items()]
    for mode in ('baseline','supervised','distilled'):
        model_path=MODEL if mode=='baseline' else root/(mode+'-model')
        model=models.ColBERT(model_name_or_path=str(model_path),device='cuda',query_length=32,document_length=180,local_files_only=True)
        records,eval_seconds=evaluate(model,docs,queries,test)
        save(root/(mode+'.json'),{'model':str(model_path),'eval_seconds':eval_seconds,
                                 'peak_allocated':torch.cuda.max_memory_allocated(),'per_query':records})
        del model;gc.collect();torch.cuda.empty_cache()
    print('finished',flush=True)

if __name__=='__main__':main()
