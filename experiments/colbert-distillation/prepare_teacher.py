"""Prepare the frozen candidate manifest and teacher score inputs.

--synthetic-encoder is a CPU-only interface smoke test, not a model evaluation.
"""
import argparse, hashlib, json, math, random
from pathlib import Path
from dataset_io import read_records

PROMPT = 'Instruct: Given a scientific claim, retrieve documents that support or refute the claim\nQuery: '

def load_lines(path):
    return {r['_id']: r for r in read_records(path)}

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
    args=p.parse_args()
    if not args.synthetic_encoder and args.model is None: p.error('--model required unless --synthetic-encoder')
    docs=load_lines(args.data/'corpus.jsonl');queries=load_lines(args.data/'queries.jsonl')
    pools=json.loads(args.pools.read_text(encoding='utf8')); assert set(pools['train']).isdisjoint(pools['test'])
    ids=random.Random(20261107).sample(sorted(pools['train']),args.train_count)
    train=[{'id':q,'candidates':[pools['train'][q]['relevant'][0]]+pools['train'][q]['unjudged'][:3]} for q in ids]
    test=[{'id':q,'candidates':[x[0] for x in row['pool']],'relevant':row['relevant']} for q,row in pools['test'].items()]
    assert all(len(set(r['candidates']))==4 for r in train)
    assert all(len(set(r['candidates']))==30 for r in test)
    rows=train+test; dids=sorted({d for r in rows for d in r['candidates']})
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
    args.out.mkdir(parents=True,exist_ok=False)
    manifest={'seed':20261107,'train':train,'test':test,'teacher':str(args.model) if args.model else 'SYNTHETIC_ONLY','max_length':192,'temperature':0.05}
    manifest_path=args.out/'manifest.json'
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    (args.out/'teacher.json').write_text(json.dumps({'scores':scores,'synthetic_only':args.synthetic_encoder,
        'manifest_sha256':sha(manifest_path),'corpus_sha256':sha(args.data/'corpus.jsonl'),
        'queries_sha256':sha(args.data/'queries.jsonl')},indent=2),encoding='utf8')
    print(f"wrote {len(train)} train and {len(test)} test queries; synthetic_only={args.synthetic_encoder}")
if __name__=='__main__':main()
