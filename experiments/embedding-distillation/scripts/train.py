import argparse, json, math, os, random, time, gc, threading, hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from validate_candidates import validate_candidate_rows

ROOT = DATA = POOLS = TEACHER = STUDENT = None
DEVICE = 'cuda'
SEED = 20261107
PROMPT = 'Instruct: Given a scientific claim, retrieve documents that support or refute the claim\nQuery: '

def check():
    for line in open('/proc/meminfo'):
        if line.startswith('MemAvailable:') and int(line.split()[1]) < 50 * 1024 * 1024:
            raise RuntimeError('available system memory below 50 GiB')

def watch_memory():
    while True:
        try:
            check()
        except RuntimeError as exc:
            print(str(exc), flush=True)
            os._exit(3)
        time.sleep(1)

def lines(path):
    return [json.loads(x) for x in path.read_text(encoding='utf8').splitlines()]

def atomic(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf8')
    tmp.replace(path)

def data(smoke):
    docs = {x['_id']: (x.get('title','') + ' ' + x['text']).strip() for x in lines(DATA / 'corpus.jsonl')}
    queries = {x['_id']: x['text'] for x in lines(DATA / 'queries.jsonl')}
    pools = json.loads(POOLS.read_text(encoding='utf8'))
    if set(pools['train']) & set(pools['test']):
        raise ValueError('training and test pools contain overlapping query IDs')
    ids = random.Random(SEED).sample(sorted(pools['train']), 128)
    train = [{'id': q, 'candidates': [pools['train'][q]['relevant'][0]] + pools['train'][q]['unjudged'][:3]} for q in ids]
    test = [{'id': q, 'candidates': [p[0] for p in row['pool']], 'relevant': row['relevant']} for q, row in pools['test'].items()]
    validate_candidate_rows(train, test, queries, docs)
    return docs, queries, train[:2] if smoke else train, test[:2] if smoke else test

def embed(model, texts, batch, prompt=''):
    outputs = []; start = time.perf_counter()
    for offset in range(0, len(texts), batch):
        check()
        a = model.encode([prompt+x for x in texts[offset:offset+batch]], batch_size=batch,
                         normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        assert np.isfinite(a).all()
        outputs.append(a)
        if offset % (batch*50) == 0: print('encoded', offset, '/', len(texts), flush=True)
    if DEVICE == 'cuda': torch.cuda.synchronize()
    return np.concatenate(outputs), time.perf_counter()-start

def calc(model, docs, queries, train, test, batch, teacher=False):
    rows = train + test
    dids = sorted({d for row in rows for d in row['candidates']})
    qids = [row['id'] for row in rows]
    qv, qt = embed(model, [queries[x] for x in qids], batch, PROMPT if teacher else '')
    dv, dt = embed(model, [docs[x] for x in dids], batch)
    idx = {k:i for i,k in enumerate(dids)}
    scores = {row['id']: [float(x) for x in (dv[[idx[d] for d in row['candidates']]] @ qv[i])] for i,row in enumerate(rows)}
    return scores, {'documents':len(dids), 'queries':len(qids), 'doc_encode_s':dt, 'query_encode_s':qt}

def metrics(rows, scores):
    result=[]
    for row in rows:
        order = sorted(range(len(row['candidates'])), key=lambda i: -scores[row['id']][i])
        relevant = set(row['relevant']); found = [int(row['candidates'][i] in relevant) for i in order[:10]]
        ideal = sum(1/math.log2(i+2) for i in range(min(10,len(relevant))))
        result.append({'query_id':row['id'], 'ndcg10':sum(v/math.log2(i+2) for i,v in enumerate(found))/ideal if ideal else 0,
                       'mrr10':next((1/(i+1) for i,v in enumerate(found) if v), 0.)})
    return result

def load_student():
    model = SentenceTransformer(STUDENT, device=DEVICE, local_files_only=True)
    model.max_seq_length=192
    return model

def train_student(model, mode, rows, docs, queries, teacher_scores, smoke):
    model.train(); opt=torch.optim.AdamW(model.parameters(), lr=2e-5)
    rng=random.Random(SEED); steps=2 if smoke else 128; permutation=[]; t0=time.perf_counter()
    loss_last=None
    for step in range(steps):
        check()
        while len(permutation)<4:
            order=list(range(len(rows))); rng.shuffle(order); permutation.extend(order)
        selected=[rows[permutation.pop(0)] for _ in range(4)]
        texts=[queries[row['id']] for row in selected]+[docs[d] for row in selected for d in row['candidates']]
        features={k:v.to(DEVICE) if isinstance(v,torch.Tensor) else v for k,v in model.tokenize(texts).items()}
        embeddings=F.normalize(model(features)['sentence_embedding'], p=2, dim=-1)
        q=embeddings[:4]; d=embeddings[4:].reshape(4,4,-1)
        logits=torch.einsum('bd,bkd->bk',q,d)/0.05
        if mode=='supervised': loss=F.cross_entropy(logits,torch.zeros(4,dtype=torch.long,device=DEVICE))
        else:
            teacher=torch.tensor([teacher_scores[row['id']] for row in selected],device=DEVICE)/0.05
            loss=-(F.softmax(teacher,dim=-1)*F.log_softmax(logits,dim=-1)).sum(dim=-1).mean()
        if not torch.isfinite(loss): raise RuntimeError('nonfinite loss')
        opt.zero_grad(set_to_none=True);loss.backward();opt.step();loss_last=float(loss.detach().cpu())
        if step%20==0: print(mode,step,'loss',loss_last,flush=True)
    if DEVICE == 'cuda': torch.cuda.synchronize()
    return {'steps':steps,'batch_queries':4,'lr':2e-5,'last_loss':loss_last,'training_s':time.perf_counter()-t0}

def main():
    global ROOT, DATA, POOLS, TEACHER, STUDENT, DEVICE
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['teacher','student'],required=True);parser.add_argument('--smoke',action='store_true');parser.add_argument('--device',choices=['cuda','cpu'],default='cuda');parser.add_argument('--data',type=Path,required=True);parser.add_argument('--pools',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--teacher',required=True);parser.add_argument('--student',required=True);args=parser.parse_args()
    DEVICE=args.device
    ROOT, DATA, POOLS, TEACHER, STUDENT = args.out, args.data, args.pools, args.teacher, args.student
    root=ROOT/('smoke' if args.smoke else 'formal');root.mkdir(parents=True,exist_ok=True)
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    check();threading.Thread(target=watch_memory,daemon=True).start()
    if DEVICE == 'cuda': torch.cuda.set_per_process_memory_fraction(.20)
    docs,queries,train,test=data(args.smoke)
    manifest={'seed':SEED,'train':train,'test':test,'teacher':TEACHER,'student':STUDENT,'max_length':192,'temperature':0.05,'pools_sha256':hashlib.sha256(POOLS.read_bytes()).hexdigest(),'corpus_sha256':hashlib.sha256((DATA/'corpus.jsonl').read_bytes()).hexdigest(),'queries_sha256':hashlib.sha256((DATA/'queries.jsonl').read_bytes()).hexdigest()}
    if args.stage=='teacher':
        atomic(root/'manifest.json', manifest); atomic(root/'teacher-status.json', {'state':'running'})
    else:
        if json.loads((root/'manifest.json').read_text(encoding='utf8')) != manifest: raise SystemExit('Data, pools, or model paths differ from teacher stage')
        status=json.loads((root/'teacher-status.json').read_text(encoding='utf8'))
        if status != {'state':'complete','sha256':hashlib.sha256((root/'teacher.json').read_bytes()).hexdigest()}: raise SystemExit('Teacher stage incomplete or output changed')
    if args.stage=='teacher':
        t=time.perf_counter();model=SentenceTransformer(TEACHER,device=DEVICE,local_files_only=True,model_kwargs={'dtype':torch.bfloat16} if DEVICE=='cuda' else {});model.max_seq_length=192
        check();print('teacher loaded seconds',time.perf_counter()-t,flush=True)
        scores,timing=calc(model,docs,queries,train,test,2,teacher=True)
        atomic(root/'teacher.json',{'scores':scores,'timing':timing,'test_metrics':metrics(test,scores),'peak_allocated':torch.cuda.max_memory_allocated()})
        atomic(root/'teacher-status.json',{'state':'complete','sha256':hashlib.sha256((root/'teacher.json').read_bytes()).hexdigest()})
    else:
        teacher=json.loads((root/'teacher.json').read_text(encoding='utf8'))['scores']
        for mode in ['baseline','supervised','distilled']:
            torch.manual_seed(SEED);torch.cuda.empty_cache()
            if DEVICE == 'cuda': torch.cuda.reset_peak_memory_stats()
            model=load_student();check()
            info={} if mode=='baseline' else train_student(model,mode,train,docs,queries,teacher,args.smoke)
            model.eval();scores,timing=calc(model,docs,queries,[],test,16)
            atomic(root/(mode+'.json'),{'scores':scores,'timing':timing,'test_metrics':metrics(test,scores),'training':info,'peak_allocated':torch.cuda.max_memory_allocated(),'parameter_count':sum(x.numel() for x in model.parameters())})
            if mode!='baseline':model.save(str(root/(mode+'-model')))
            del model;gc.collect();torch.cuda.empty_cache()
    print('finished',args.stage,flush=True)

if __name__=='__main__':main()
