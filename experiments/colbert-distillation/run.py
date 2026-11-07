import os, json, random, time, threading, argparse, math, gc
from pathlib import Path
import numpy as np
import torch
from pylate import models, losses, rank
from verify_binding import verify_binding

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
    verify_binding(SOURCE,DATA,manifest,teacher,smoke)
    train=manifest['train'];test=manifest['test'];scores=teacher['scores']
    assert len(train)==128 and len(test)==300
    assert all(len(row['candidates'])==4 and len(scores[row['id']])==4 for row in train)
    assert all(len(row['candidates'])==30 and len(scores[row['id']])==30 for row in test)
    return docs,queries,train[:2] if smoke else train,test[:2] if smoke else test,scores

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
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--output',required=True,type=Path);p.add_argument('--source',required=True,type=Path);p.add_argument('--data',required=True,type=Path);p.add_argument('--model',required=True,type=Path);args=p.parse_args()
    ROOT,SOURCE,DATA,MODEL=args.output,args.source,args.data,args.model
    root=ROOT/('smoke' if args.smoke else 'formal');root.mkdir(parents=True,exist_ok=False)
    check();threading.Thread(target=watchdog,daemon=True).start();torch.cuda.set_per_process_memory_fraction(.15)
    docs,queries,train_rows,test,teacher=setup(args.smoke)
    for mode in ('baseline','supervised','distilled'):
        random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
        model=models.ColBERT(model_name_or_path=str(MODEL),device='cuda',query_length=32,document_length=180,local_files_only=True)
        check();parameters=sum(x.numel() for x in model.parameters())
        timing={} if mode=='baseline' else train(model,docs,queries,train_rows,teacher,mode,args.smoke)
        records,eval_seconds=evaluate(model,docs,queries,test)
        save(root/(mode+'.json'),{'model':str(MODEL),'parameter_count':parameters,'training':timing,'eval_seconds':eval_seconds,
                                 'peak_allocated':torch.cuda.max_memory_allocated(),'per_query':records})
        if mode!='baseline': model.save_pretrained(str(root/(mode+'-model')))
        del model;gc.collect();torch.cuda.empty_cache()
    print('finished',flush=True)

if __name__=='__main__':main()
