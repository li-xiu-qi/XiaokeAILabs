"""Prepare the frozen candidate manifest and teacher score inputs.

--synthetic-encoder is a CPU-only interface smoke test, not a model evaluation.
"""
import argparse, hashlib, json, math, random, os, threading, time
from pathlib import Path

PROMPT = 'Instruct: Given a scientific claim, retrieve documents that support or refute the claim\nQuery: '

def load_lines(path):
    return {r['_id']: r for r in (json.loads(s) for s in path.read_text(encoding='utf8').splitlines())}

def synthetic(text):
    raw = hashlib.sha256(text.encode()).digest()
    v = [(b - 127.5) / 127.5 for b in raw]
    norm = math.sqrt(sum(x*x for x in v))
    return [x/norm for x in v]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True);p.add_argument('--pools',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--model',type=Path)
    p.add_argument('--device',default='cpu');p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--train-count',type=int,default=128);p.add_argument('--synthetic-encoder',action='store_true')
    p.add_argument('--min-teacher-gap',type=float,required=True,help='Predeclared absolute dev nDCG@10 margin, at least 0.05')
    args=p.parse_args()
    if not math.isfinite(args.min_teacher_gap) or args.min_teacher_gap<.05: p.error('--min-teacher-gap must be at least 0.05')
    if not args.synthetic_encoder and args.model is None: p.error('--model required unless --synthetic-encoder')
    if (args.out/'manifest.json').exists() or (args.out/'teacher.json').exists(): p.error('Use a fresh --out directory')
    if args.device.startswith('cuda') and not args.synthetic_encoder:
        def watch_memory():
            while True:
                with open('/proc/meminfo') as f:
                    available=next(int(line.split()[1]) for line in f if line.startswith('MemAvailable:'))
                if available<50*1024*1024:
                    print('Memory below 50 GiB; stopping teacher',flush=True);os._exit(3)
                time.sleep(1)
        threading.Thread(target=watch_memory,daemon=True).start()
    docs=load_lines(args.data/'corpus.jsonl');queries=load_lines(args.data/'queries.jsonl')
    pools=json.loads(args.pools.read_text(encoding='utf8'))
    if not all(k in pools for k in ('train','dev','test')): raise ValueError('Independent train, dev and test splits required')
    if any(set(pools[a]) & set(pools[b]) for a,b in (('train','dev'),('train','test'),('dev','test'))): raise ValueError('Query splits overlap')
    ids=random.Random(20261107).sample(sorted(pools['train']),args.train_count)
    train=[{'id':q,'candidates':[pools['train'][q]['relevant'][0]]+pools['train'][q]['unjudged'][:3]} for q in ids]
    dev=[{'id':q,'candidates':[str(x[0]) for x in row['pool']],'relevant':[str(x) for x in row['relevant']]} for q,row in pools['dev'].items()]
    assert all(len(set(r['candidates']))==4 for r in train)
    assert len(dev)>=2 and all(len(set(r['candidates']))==30 for r in dev)
    rows=train+dev; dids=sorted({d for r in rows for d in r['candidates']})
    assert all(r['id'] in queries and all(d in docs for d in r['candidates']) for r in rows)
    if args.synthetic_encoder:
        encode=lambda texts: [synthetic(t) for t in texts]
    else:
        import torch
        from sentence_transformers import SentenceTransformer
        kwargs={'dtype':torch.bfloat16} if args.device.startswith('cuda') else {}
        model=SentenceTransformer(str(args.model),device=args.device,local_files_only=True,model_kwargs=kwargs)
        model.max_seq_length=192
        encode=lambda texts: model.encode(texts,batch_size=args.batch_size,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=True)
    qv=encode([PROMPT+queries[r['id']]['text'] for r in rows])
    dv=encode([(docs[d].get('title','')+' '+docs[d]['text']).strip() for d in dids]); idx={d:i for i,d in enumerate(dids)}
    scores={r['id']:[float(sum(float(a)*float(b) for a,b in zip(qv[i],dv[idx[d]]))) for d in r['candidates']] for i,r in enumerate(rows)}
    assert all(math.isfinite(s) for vals in scores.values() for s in vals)
    args.out.mkdir(parents=True,exist_ok=True)
    digest=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    manifest={'seed':20261107,'train':train,'dev':dev,'teacher':str(args.model) if args.model else 'SYNTHETIC_ONLY','max_length':192,'temperature':0.05,'min_teacher_gap':args.min_teacher_gap,'pools_sha256':digest(args.pools),'corpus_sha256':digest(args.data/'corpus.jsonl'),'queries_sha256':digest(args.data/'queries.jsonl')}
    (args.out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    (args.out/'teacher.json').write_text(json.dumps({'scores':scores,'synthetic_only':args.synthetic_encoder},indent=2),encoding='utf8')
    print(f"wrote {len(train)} train and {len(dev)} dev queries; test not scored; synthetic_only={args.synthetic_encoder}")
if __name__=='__main__':main()
