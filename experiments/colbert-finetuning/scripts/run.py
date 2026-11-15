import argparse, json, math, random, os, sys
from pathlib import Path
os.environ["HF_HUB_OFFLINE"] = "1"
import numpy as np
import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from pylate import models, losses, utils, rank
parser=argparse.ArgumentParser()
parser.add_argument('--data', type=Path, required=True)
parser.add_argument('--prepared', type=Path, required=True)
parser.add_argument('--pairs', type=Path, required=True)
parser.add_argument('--model', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--smoke', action='store_true')
args=parser.parse_args()
root=args.out
root.mkdir(parents=True,exist_ok=True)
source=args.data
random.seed(20261107); np.random.seed(20261107); torch.manual_seed(20261107); torch.cuda.manual_seed_all(20261107)
def rows(p):
    return [json.loads(x) for x in open(p, encoding='utf8')]
docs={x['_id']: (x.get('title','')+' '+x['text']).strip() for x in rows(source/'corpus.jsonl')}
queries={x['_id']:x['text'] for x in rows(source/'queries.jsonl')}
pools=json.load(open(args.prepared/'pools.json'))
triples=rows(args.pairs)
assert set(pools['train']).isdisjoint(pools['test']) and set(pools['train']).isdisjoint(pools['dev']) and set(pools['dev']).isdisjoint(pools['test'])
assert all(t['query_id'] in pools['train'] and t['positive_doc_id'] in pools['train'][t['query_id']]['relevant'] and t['unjudged_doc_id'] in pools['train'][t['query_id']]['unjudged'] for t in triples)
train=Dataset.from_list([{'query':queries[t['query_id']], 'positive':docs[t['positive_doc_id']], 'negative':docs[t['unjudged_doc_id']]} for t in triples])
model=models.ColBERT(model_name_or_path=str(args.model),device='cuda',query_length=32,document_length=180)
print('model loaded training triples',len(triples),flush=True)
def metric(order,relevant):
    y=[int(x in relevant) for x in order[:10]]
    dcg=sum(v/math.log2(i+2) for i,v in enumerate(y))
    ideal=sum(1/math.log2(i+2) for i in range(min(len(relevant),10)))
    return dcg/ideal if ideal else 0., next((1/(i+1) for i,v in enumerate(y) if v),0.)
def evaluate(label,limit=None):
    records=[]; model.eval()
    for n,(qid,item) in enumerate(pools['test'].items()):
        if limit and n>=limit: break
        ids=[x[0] for x in item['pool']]; relevant=set(item['relevant'])
        qe=model.encode([queries[qid]],is_query=True,show_progress_bar=False,convert_to_numpy=True)
        de=model.encode([docs[x] for x in ids],is_query=False,batch_size=16,show_progress_bar=False,convert_to_numpy=True)
        result=rank.rerank(documents_ids=[ids],queries_embeddings=qe,documents_embeddings=[de],device='cuda')[0]
        order=[str(x['id']) for x in result]
        a,b=metric(order,relevant); c,d=metric(ids,relevant)
        records.append({'query_id':qid,'relevant':sorted(relevant),'bm25_order':ids,'colbert_order':order,'scores':[float(x['score']) for x in result], 'ndcg10':a,'mrr10':b,'bm25_ndcg10':c,'bm25_mrr10':d})
        if n%25==0: print(label,n,flush=True)
    (root/(label+'-per-query.json')).write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf8')
    return {'n':len(records), **{k:sum(x[k] for x in records)/len(records) for k in ['ndcg10','mrr10','bm25_ndcg10','bm25_mrr10']}}
smoke=args.smoke
before=evaluate('smoke-before' if smoke else 'before',2 if smoke else None)
print('before',before,flush=True)
loss=losses.Contrastive(model=model,temperature=1.0)
args=SentenceTransformerTrainingArguments(output_dir=str(root/'outputs'),max_steps=2 if smoke else 180,per_device_train_batch_size=4,learning_rate=3e-6,warmup_steps=0 if smoke else 18,logging_steps=1 if smoke else 10,save_strategy='no',report_to='none',seed=20261107,dataloader_num_workers=0, fp16=False,bf16=False)
trainer=SentenceTransformerTrainer(model=model,args=args,train_dataset=train,loss=loss,data_collator=utils.ColBERTCollator(model.tokenize))
trainer.train()
if not smoke:
    model.save_pretrained(str(root/'fine-tuned'))
after=evaluate('smoke-after' if smoke else 'after',2 if smoke else None)
print('after',after,flush=True)
summary={'before':before,'after':after,'train_pairs':len(triples),'train_queries':len(pools['train']),'test_queries':len(pools['test']),'seed':20261107,'steps':2 if smoke else 180,'batch_size':4,'lr':3e-6,'temperature':1.0,'model':'lightonai/colbertv2.0','data':'BEIR SciFact qrels/test.tsv','candidate_pool':'fixed BM25 top 30','warning':'unjudged negatives may be relevant; test questions held out but documents can overlap; this measures reranking only'}
(root/('smoke-summary.json' if smoke else 'summary.json')).write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
