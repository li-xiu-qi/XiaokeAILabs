"""SciFact pairwise reranker distillation; run only after GPU slot is granted."""
import argparse, hashlib, json, math, os, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

ROOT=Path('.')
DATA=Path('.')
TEACHER=Path('.')
STUDENT=Path('.')
SEED=20261107; MAXLEN=192; STEPS=180; BATCH=8; TEMP=2.0; WEIGHT=0.5

def read_jsonl(path):
    with path.open() as f: return [json.loads(x) for x in f if x.strip()]
def write(name, obj): (ROOT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def tok(t, pairs):
    q,d=zip(*pairs)
    return t(list(q),list(d),padding=True,truncation='only_second',max_length=MAXLEN,return_tensors='pt').to('cuda')
def model(path):
    t=AutoTokenizer.from_pretrained(path,local_files_only=True)
    m=AutoModelForSequenceClassification.from_pretrained(path,local_files_only=True).cuda().eval()
    return t,m

def score(model, tokenizer, entries):
    output={}; model.eval()
    torch.cuda.synchronize(); start=time.perf_counter()
    with torch.inference_mode():
        for i in range(0,len(entries),32):
            batch=entries[i:i+32]
            logits=model(**tok(tokenizer, [(queries[q],docs[d]) for q,d in batch])).logits.flatten().float().cpu().tolist()
            assert all(math.isfinite(v) for v in logits)
            output.update((f'{q}:{d}',v) for (q,d),v in zip(batch,logits))
    torch.cuda.synchronize()
    return output,time.perf_counter()-start

def metrics(part,scores):
    rows=[]
    for qid,row in part.items():
        ranking=sorted(((str(did),scores[f'{qid}:{did}']) for did,_ in row['pool']),key=lambda x:(-x[1],x[0]))
        rel=set(map(str,row['relevant']))
        hit=[int(d in rel) for d,_ in ranking]
        dcg=sum(x/math.log2(i+2) for i,x in enumerate(hit[:10]))
        ideal=sum(1/math.log2(i+2) for i in range(min(10,len(rel))))
        rr=next((1/(i+1) for i,x in enumerate(hit) if x),0.)
        rows.append({'qid':qid,'ndcg10':dcg/ideal if ideal else 0,'mrr10':rr if rr>=.1 else 0,'relevant_in_pool':sum(hit),'relevant_total':len(rel)})
    return {'n':len(rows),'ndcg10':sum(x['ndcg10'] for x in rows)/len(rows),'mrr10':sum(x['mrr10'] for x in rows)/len(rows),'recall_pool':sum(x['relevant_in_pool'] for x in rows)/sum(x['relevant_total'] for x in rows),'per_query':rows}

def reset():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

def train(mode,teacher_scores):
    reset(); t,m=model(STUDENT); m.train()
    opt=torch.optim.AdamW(m.parameters(),lr=2e-5,weight_decay=.01)
    sched=get_linear_schedule_with_warmup(opt,num_warmup_steps=18,num_training_steps=STEPS)
    rng=random.Random(SEED); ids=list(pools['train']); log=[]; n=0
    while n<STEPS:
        rng.shuffle(ids)
        for start in range(0,len(ids),BATCH):
            batch=[]
            for q in ids[start:start+BATCH]:
                row=pools['train'][q]; batch.append((q,rng.choice(row['relevant']),rng.choice(row['unjudged'])))
            pairs=[(q,d) for q,p,neg in batch for d in (p,neg)]
            s=m(**tok(t,[(queries[q],docs[d]) for q,d in pairs])).logits.reshape(-1,2)
            supervised=F.cross_entropy(s,torch.zeros(len(batch),device='cuda',dtype=torch.long))
            if mode=='distilled':
                teacher=torch.tensor([[teacher_scores[f'{q}:{p}'],teacher_scores[f'{q}:{neg}']] for q,p,neg in batch],device='cuda')
                soft=F.kl_div(F.log_softmax(s/TEMP,dim=-1), F.softmax(teacher/TEMP,dim=-1),reduction='batchmean')*TEMP**2
                loss=(1-WEIGHT)*supervised+WEIGHT*soft
            else: loss=supervised
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.); opt.step(); sched.step(); n+=1
            log.append({'step':n,'loss':float(loss.detach().cpu())})
            if n>=STEPS: break
    out=ROOT/f'{mode}-model'; out.mkdir(exist_ok=True); m.save_pretrained(out); t.save_pretrained(out); write(f'{mode}-training.json',log)
    return t,m

def main():
    global queries,docs,pools,ROOT,DATA,TEACHER,STUDENT
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True, help='Directory containing scifact/ and prepared/pools.json')
    ap.add_argument('--teacher', required=True, help='Local MiniLM-L12 teacher snapshot')
    ap.add_argument('--student', required=True, help='Local TinyBERT student snapshot')
    ap.add_argument('--out', required=True)
    args=ap.parse_args()
    ROOT=Path(args.out); ROOT.mkdir(parents=True,exist_ok=True)
    DATA=Path(args.data_root); TEACHER=Path(args.teacher); STUDENT=Path(args.student)
    teacher_id='cross-encoder/ms-marco-MiniLM-L12-v2@7b0235231ca2674cb8ca8f022859a6eba2b1c968'
    if not torch.cuda.is_available(): raise SystemExit('CUDA unavailable')
    torch.set_num_threads(2); reset()
    queries={x['_id']:x['text'] for x in read_jsonl(DATA/'scifact/queries.jsonl')}
    docs={x['_id']:(x.get('title','')+' '+x['text']).strip() for x in read_jsonl(DATA/'scifact/corpus.jsonl')}
    pools=json.loads((DATA/'prepared/pools.json').read_text())
    assert not (set(pools['train']) & (set(pools['dev'])|set(pools['test'])))
    trainpairs=list(dict.fromkeys((q,str(d)) for q,row in pools['train'].items() for d in row['relevant']+row['unjudged']))
    evalpairs=list(dict.fromkeys((q,str(d)) for split in ('dev','test') for q,row in pools[split].items() for d,_ in row['pool']))
    allpairs=list(dict.fromkeys(trainpairs+evalpairs))
    write('manifest.json',{'seed':SEED,'max_length':MAXLEN,'steps':STEPS,'batch_queries':BATCH,'temperature':TEMP,'distill_weight':WEIGHT,'teacher':teacher_id,'student':'cross-encoder/ms-marco-TinyBERT-L2-v2@81d1926f67cb8eee2c2be17ca9f793c7c3bd20cc','pools_sha256':sha(DATA/'prepared/pools.json'),'counts':{s:len(pools[s]) for s in pools},'pairs':{'teacher_train':len(trainpairs),'eval':len(evalpairs)},'caveats':'student trained on MS MARCO; unjudged != guaranteed negative; query-held-out, documents can overlap'})
    t,m=model(TEACHER); eval_ts,eval_seconds=score(m,t,evalpairs); train_ts,train_seconds=score(m,t,trainpairs); ts={**eval_ts,**train_ts}; teacher_eval={'seconds':eval_seconds,'train_scoring_seconds':train_seconds,**{s:metrics(pools[s],eval_ts) for s in ('dev','test')}}; write('teacher-scores.json',ts); write('teacher-eval.json',teacher_eval); del m,t; torch.cuda.empty_cache()
    t,m=model(STUDENT); ss,seconds=score(m,t,evalpairs); student_eval={'seconds':seconds,**{s:metrics(pools[s],ss) for s in ('dev','test')}}; write('student-before-scores.json',ss); write('student-before.json',student_eval); del m,t; torch.cuda.empty_cache()
    if teacher_eval['dev']['ndcg10'] <= student_eval['dev']['ndcg10']:
        write('stopped.json',{'reason':'teacher dev NDCG@10 not greater than student; training skipped','teacher':teacher_eval['dev']['ndcg10'],'student':student_eval['dev']['ndcg10']}); return
    for mode in ('supervised','distilled'):
        t,m=train(mode,ts); ss,seconds=score(m,t,evalpairs); write(f'{mode}-scores.json',ss); write(f'{mode}-eval.json',{'seconds':seconds,**{s:metrics(pools[s],ss) for s in ('dev','test')}}); del m,t; torch.cuda.empty_cache()
if __name__=='__main__': main()
