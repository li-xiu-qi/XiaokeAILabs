"""Prepare and evaluate a small, deterministic T2Retrieval development subset."""
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd


def prepare(source, output):
    output.mkdir(parents=True, exist_ok=False)
    corpus = pd.read_parquet(source / 'corpus/dev-00000-of-00001.parquet').set_index('_id')['text'].to_dict()
    queries = pd.read_parquet(source / 'queries/dev-00000-of-00001.parquet').set_index('_id')['text'].to_dict()
    labels = pd.read_parquet(source / 'data/dev-00000-of-00001.parquet')
    labels = labels[labels['score'] > 0]
    groups = {k: sorted(set(x for x in v if x in corpus and corpus[x]))
              for k, v in labels.groupby('query-id')['corpus-id']}
    keys = sorted((k for k, v in groups.items() if k in queries and queries[k] and 1 <= len(v) <= 3), key=int)
    rng = random.Random(42)
    rng.shuffle(keys)
    train, test, seen_queries, train_docs = [], [], set(), set()
    for key in keys:
        if len(train) >= 256:
            break
        if queries[key] in seen_queries or groups[key][0] in train_docs:
            continue
        train.append(key)
        seen_queries.add(queries[key])
        train_docs.update(groups[key])
    for key in keys:
        if len(test) >= 100:
            break
        if key in train or queries[key] in seen_queries or train_docs.intersection(groups[key]):
            continue
        test.append(key)
        seen_queries.add(queries[key])
    if len(train) != 256 or len(test) != 100:
        raise ValueError('Dataset does not contain enough suitable queries')
    positives = {doc for key in test for doc in groups[key]}
    pool = sorted(positives | set(rng.sample(sorted(set(corpus) - positives - train_docs), 200)))
    rows = [{'query': queries[key], 'pos_doc': corpus[groups[key][0]]} for key in train]
    evaluation = {'queries': [queries[key] for key in test], 'qids': test,
                  'documents': [corpus[doc] for doc in pool], 'docids': pool,
                  'relevant': {key: groups[key] for key in test}}
    (output / 'train.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
    (output / 'eval.json').write_text(json.dumps(evaluation, ensure_ascii=False), encoding='utf-8')
    for path in [output / 'train.jsonl', output / 'eval.json']:
        print(path.name, hashlib.sha256(path.read_bytes()).hexdigest())
    print('train queries:', len(train), 'test queries:', len(test), 'candidate documents:', len(pool))


def evaluate(data, model, output):
    from sentence_transformers import SentenceTransformer
    if output.exists():
        raise ValueError(f'Output already exists: {output}')
    dataset = json.loads(data.read_text(encoding='utf-8'))
    encoder = SentenceTransformer(model, device='cuda')
    encoder.max_seq_length = 256
    queries = encoder.encode(dataset['queries'], batch_size=32, normalize_embeddings=True, show_progress_bar=True)
    documents = encoder.encode(dataset['documents'], batch_size=32, normalize_embeddings=True, show_progress_bar=True)
    order = np.argsort(-(queries @ documents.T), axis=1)
    hits, reciprocal_ranks = [], []
    for i, key in enumerate(dataset['qids']):
        relevant = set(dataset['relevant'][key])
        ranks = [j + 1 for j, x in enumerate(order[i]) if dataset['docids'][int(x)] in relevant]
        hits.append(int(min(ranks) <= 10))
        reciprocal_ranks.append(1 / min(ranks))
    result = {'queries': len(hits), 'candidate_documents': len(dataset['docids']),
              'hit_at_10': float(np.mean(hits)), 'mrr': float(np.mean(reciprocal_ranks)),
              'hit_per_query': hits, 'rr_per_query': reciprocal_ranks}
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print({key: value for key, value in result.items() if not isinstance(value, list)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--source', type=Path, required=True)
    prep.add_argument('--output', type=Path, required=True)
    ev = sub.add_parser('evaluate')
    ev.add_argument('--data', type=Path, required=True)
    ev.add_argument('--model', required=True)
    ev.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.source, args.output)
    else:
        evaluate(args.data, args.model, args.output)


if __name__ == '__main__':
    main()
