"""SciFact pairwise reranker distillation; run only after GPU slot is granted."""
import argparse, hashlib, json, math, os, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, get_linear_schedule_with_warmup

ROOT=Path('.')
DATA=Path('.')
TEACHER=Path('.')
STUDENT=Path('.')
DEVICE='cuda'
SEED=20261107; MAXLEN=192; STEPS=180; BATCH=8; TEMP=2.0; WEIGHT=0.5
QWEN_TASK='Given a scientific claim, retrieve scientific evidence relevant to the claim'
QWEN_PREFIX='<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
QWEN_SUFFIX='<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'

def read_jsonl(path):
    with path.open() as f: return [json.loads(x) for x in f if x.strip()]
def write(name, obj): (ROOT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def tok(t, pairs):
    q,d=zip(*pairs)
    return t(list(q),list(d),padding=True,truncation='only_second',max_length=MAXLEN,return_tensors='pt').to(DEVICE)
def model(path):
    t=AutoTokenizer.from_pretrained(path,local_files_only=True)
    m=AutoModelForSequenceClassification.from_pretrained(path,local_files_only=True).to(DEVICE).eval()
    return t,m

def teacher_model(path, kind):
    if kind == 'cross-encoder': return model(path)
    t=AutoTokenizer.from_pretrained(path,local_files_only=True,padding_side='left')
    options={'local_files_only':True}
    if DEVICE=='cuda': options.update(torch_dtype=torch.bfloat16,device_map={'':'cuda:0'})
    m=AutoModelForCausalLM.from_pretrained(path,**options)
    if DEVICE=='cpu': m=m.to('cpu')
    return t,m.eval()

def qwen_score(model, tokenizer, entries, max_length=512, batch_size=4):
    prefix=tokenizer.encode(QWEN_PREFIX,add_special_tokens=False)
    suffix=tokenizer.encode(QWEN_SUFFIX,add_special_tokens=False)
    yes=tokenizer.encode('yes',add_special_tokens=False)
    no=tokenizer.encode('no',add_special_tokens=False)
    if len(yes)!=1 or len(no)!=1 or max_length <= len(prefix)+len(suffix):
        raise ValueError('Invalid Qwen yes/no tokenization or maximum length')
    output={}; model.eval()
    if DEVICE=='cuda': torch.cuda.synchronize()
    start=time.perf_counter()
    with torch.inference_mode():
        for i in range(0,len(entries),batch_size):
            batch=entries[i:i+batch_size]
            texts=[f'<Instruct>: {QWEN_TASK}\n<Query>: {queries[q]}\n<Document>: {docs[d]}' for q,d in batch]
            token_ids=tokenizer(texts,padding=False,truncation=True,max_length=max_length-len(prefix)-len(suffix),add_special_tokens=False)['input_ids']
            inputs=tokenizer.pad({'input_ids':[prefix+x+suffix for x in token_ids]},padding=True,return_tensors='pt').to(DEVICE)
            logits=model(**inputs,use_cache=False,logits_to_keep=1).logits[:,-1,:].float()
            values=(logits[:,yes[0]]-logits[:,no[0]]).cpu().tolist()
            if not all(math.isfinite(x) for x in values): raise ValueError('Nonfinite Qwen teacher score')
            output.update((f'{q}:{d}',v) for (q,d),v in zip(batch,values))
    if DEVICE=='cuda': torch.cuda.synchronize()
    return output,time.perf_counter()-start

def sampled_training_pairs():
    rng=random.Random(SEED); ids=list(pools['train']); pairs=[]; n=0
    while n<STEPS:
        rng.shuffle(ids)
        for start in range(0,len(ids),BATCH):
            for q in ids[start:start+BATCH]:
                row=pools['train'][q]
                pairs.extend(((q,str(rng.choice(row['relevant']))),(q,str(rng.choice(row['unjudged'])))))
            n+=1
            if n>=STEPS: break
    return list(dict.fromkeys(pairs))

def score(model, tokenizer, entries):
    output={}; model.eval()
    if DEVICE=='cuda': torch.cuda.synchronize()
    start=time.perf_counter()
    with torch.inference_mode():
        for i in range(0,len(entries),32):
            batch=entries[i:i+32]
            logits=model(**tok(tokenizer, [(queries[q],docs[d]) for q,d in batch])).logits.flatten().float().cpu().tolist()
            assert all(math.isfinite(v) for v in logits)
            output.update((f'{q}:{d}',v) for (q,d),v in zip(batch,logits))
    if DEVICE=='cuda': torch.cuda.synchronize()
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
            supervised=F.cross_entropy(s,torch.zeros(len(batch),device=DEVICE,dtype=torch.long))
            if mode=='distilled':
                teacher=torch.tensor([[teacher_scores[f'{q}:{p}'],teacher_scores[f'{q}:{neg}']] for q,p,neg in batch],device=DEVICE)
                soft=F.kl_div(F.log_softmax(s/TEMP,dim=-1), F.softmax(teacher/TEMP,dim=-1),reduction='batchmean')*TEMP**2
                loss=(1-WEIGHT)*supervised+WEIGHT*soft
            else: loss=supervised
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.); opt.step(); sched.step(); n+=1
            log.append({'step':n,'loss':float(loss.detach().cpu())})
            if n>=STEPS: break
    out=ROOT/f'{mode}-model'; out.mkdir(exist_ok=True); m.save_pretrained(out); t.save_pretrained(out); write(f'{mode}-training.json',log)
    return t,m

def main():
    global queries,docs,pools,ROOT,DATA,TEACHER,STUDENT,DEVICE,STEPS
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True, help='Directory containing scifact/ and prepared/pools.json')
    ap.add_argument('--teacher', required=True, help='Local teacher model snapshot')
    ap.add_argument('--teacher-type', choices=['cross-encoder','qwen3'], default='cross-encoder')
    ap.add_argument('--qwen-max-length', type=int, default=512)
    ap.add_argument('--qwen-batch-size', type=int, default=4)
    ap.add_argument('--student', required=True, help='Local TinyBERT student snapshot')
    ap.add_argument('--out', required=True)
    ap.add_argument('--stage', choices=['teacher','student','all'], default='all')
    ap.add_argument('--device', choices=['cuda','cpu'], default='cuda')
    ap.add_argument('--smoke', action='store_true', help='Two queries per split and two training steps; no performance conclusions')
    ap.add_argument('--min-teacher-gap', type=float, help='Required positive, prespecified absolute NDCG@10 teacher advantage on dev')
    args=ap.parse_args()
    if args.qwen_batch_size < 1 or args.qwen_max_length < 1: raise SystemExit('Qwen batch size and length must be positive')
    if not args.smoke and (args.min_teacher_gap is None or not math.isfinite(args.min_teacher_gap) or args.min_teacher_gap<=0):
        raise SystemExit('Specify a positive, prespecified --min-teacher-gap before teacher scoring')
    ROOT=Path(args.out); ROOT.mkdir(parents=True,exist_ok=True)
    DATA=Path(args.data_root); TEACHER=Path(args.teacher); STUDENT=Path(args.student)
    DEVICE=args.device
    if DEVICE=='cuda' and not torch.cuda.is_available(): raise SystemExit('CUDA unavailable')
    if args.smoke: STEPS=2
    torch.set_num_threads(2); reset()
    queries={x['_id']:x['text'] for x in read_jsonl(DATA/'scifact/queries.jsonl')}
    docs={x['_id']:(x.get('title','')+' '+x['text']).strip() for x in read_jsonl(DATA/'scifact/corpus.jsonl')}
    pools=json.loads((DATA/'prepared/pools.json').read_text())
    assert not (set(pools['train']) & (set(pools['dev'])|set(pools['test'])))
    if args.smoke: pools={s:dict(list(rows.items())[:2]) for s,rows in pools.items()}
    trainpairs=sampled_training_pairs()
    evalpairs=list(dict.fromkeys((q,str(d)) for q,row in pools['dev'].items() for d,_ in row['pool']))
    allpairs=list(dict.fromkeys(trainpairs+evalpairs))
    manifest={'seed':SEED,'max_length':MAXLEN,'steps':STEPS,'batch_queries':BATCH,'temperature':TEMP,'distill_weight':WEIGHT,'teacher_path':str(TEACHER.resolve()),'teacher_type':args.teacher_type,'qwen_max_length':args.qwen_max_length,'qwen_batch_size':args.qwen_batch_size,'qwen_task':QWEN_TASK if args.teacher_type=='qwen3' else None,'min_teacher_gap':args.min_teacher_gap,'student_path':str(STUDENT.resolve()),'pools_sha256':sha(DATA/'prepared/pools.json'),'corpus_sha256':sha(DATA/'scifact/corpus.jsonl'),'queries_sha256':sha(DATA/'scifact/queries.jsonl'),'counts':{s:len(pools[s]) for s in pools},'pairs':{'teacher_train':len(trainpairs),'eval':len(evalpairs)},'smoke':args.smoke,'caveats':'unjudged != guaranteed negative; query-held-out, documents can overlap'}
    def score_teacher(m,t,pairs):
        return qwen_score(m,t,pairs,args.qwen_max_length,args.qwen_batch_size) if args.teacher_type=='qwen3' else score(m,t,pairs)
    if args.stage in ('teacher','all'):
        write('manifest.json',manifest)
        write('teacher-status.json',{'state':'running'})
        t,m=teacher_model(TEACHER,args.teacher_type); eval_ts,eval_seconds=score_teacher(m,t,evalpairs); train_ts,train_seconds=score_teacher(m,t,trainpairs); ts={**eval_ts,**train_ts}; teacher_eval={'seconds':eval_seconds,'train_scoring_seconds':train_seconds,'dev':metrics(pools['dev'],eval_ts)}; write('teacher-scores.json',ts); write('teacher-eval.json',teacher_eval); del m,t; torch.cuda.empty_cache()
        write('teacher-status.json',{'state':'complete','scores_sha256':sha(ROOT/'teacher-scores.json'),'eval_sha256':sha(ROOT/'teacher-eval.json')})
        if args.stage=='teacher': return
    else:
        if json.loads((ROOT/'manifest.json').read_text()) != manifest: raise SystemExit('Data, pools, models or run settings differ from teacher stage')
        status=json.loads((ROOT/'teacher-status.json').read_text())
        if status != {'state':'complete','scores_sha256':sha(ROOT/'teacher-scores.json'),'eval_sha256':sha(ROOT/'teacher-eval.json')}: raise SystemExit('Teacher stage incomplete or outputs changed')
        ts=json.loads((ROOT/'teacher-scores.json').read_text()); teacher_eval=json.loads((ROOT/'teacher-eval.json').read_text())
        if set(ts)!={f'{q}:{d}' for q,d in allpairs}: raise SystemExit('Teacher scores do not match candidate pairs')
    t,m=model(STUDENT); ss,seconds=score(m,t,evalpairs); student_eval={'seconds':seconds,'dev':metrics(pools['dev'],ss)}; write('student-dev-scores.json',ss); write('student-dev.json',student_eval); del m,t; torch.cuda.empty_cache()
    if args.smoke or args.min_teacher_gap is None or not math.isfinite(args.min_teacher_gap) or args.min_teacher_gap <= 0:
        write('stopped.json', {'reason':'training requires a positive prespecified dev teacher advantage; smoke data cannot establish the gate'})
        return
    teacher_rows={row['qid']:row['ndcg10'] for row in teacher_eval['dev']['per_query']}
    student_rows={row['qid']:row['ndcg10'] for row in student_eval['dev']['per_query']}
    if len(teacher_rows) < 2 or teacher_rows.keys() != student_rows.keys():
        raise SystemExit('Teacher and student development queries must match and contain at least two queries')
    diffs=np.array([teacher_rows[q]-student_rows[q] for q in sorted(teacher_rows)], dtype=float)
    rng=np.random.default_rng(SEED)
    bootstrap=np.array([rng.choice(diffs, size=len(diffs), replace=True).mean() for _ in range(10000)])
    gap=float(diffs.mean()); lower=float(np.quantile(bootstrap, .025))
    gate={'teacher':teacher_eval['dev']['ndcg10'],'student':student_eval['dev']['ndcg10'],
          'gap':gap,'required_gap':args.min_teacher_gap,'paired_bootstrap_lower_95':lower,
          'bootstrap_samples':10000,'bootstrap_seed':SEED}
    write('teacher-gate.json', gate)
    if gap < args.min_teacher_gap or lower <= 0:
        write('stopped.json',{'reason':'teacher advantage gate failed; training skipped',**gate}); return
    for mode in ('supervised','distilled'):
        t,m=train(mode,ts); del m,t; torch.cuda.empty_cache()
    # Only after both training runs are frozen do we open the held-out test set.
    evalpairs=list(dict.fromkeys((q,str(d)) for split in ('dev','test') for q,row in pools[split].items() for d,_ in row['pool']))
    for name,path in (('teacher-final',TEACHER),('student-before',STUDENT),('supervised',ROOT/'supervised-model'),('distilled',ROOT/'distilled-model')):
        t,m=teacher_model(path,args.teacher_type) if name=='teacher-final' else model(path)
        ss,seconds=score_teacher(m,t,evalpairs) if name=='teacher-final' else score(m,t,evalpairs)
        write(f'{name}-scores.json',ss); write(f'{name}-eval.json' if name != 'student-before' else 'student-before.json',{'seconds':seconds,**{s:metrics(pools[s],ss) for s in ('dev','test')}})
        del m,t; torch.cuda.empty_cache()
if __name__=='__main__': main()
