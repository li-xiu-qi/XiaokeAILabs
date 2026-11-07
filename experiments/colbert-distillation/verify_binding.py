"""Check that teacher scores correspond to the candidate and text data."""
import hashlib
import math


def verify_binding(source, data, manifest, teacher, smoke=False):
    files={'manifest_sha256':source/'manifest.json','corpus_sha256':data/'corpus.jsonl',
           'queries_sha256':data/'queries.jsonl'}
    for key,path in files.items():
        if teacher.get(key) != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError(f'{key} mismatch or missing; regenerate teacher scores')
    if teacher.get('synthetic_only') is not False and not (smoke and teacher.get('synthetic_only') is True):
        raise ValueError('synthetic teacher scores are permitted only for smoke tests')
    rows=manifest['train']+manifest['test']
    ids=[r['id'] for r in rows]
    if len(ids)!=len(set(ids)) or set(ids)!=set(teacher['scores']):
        raise ValueError('teacher query IDs do not match manifest')
    for row in rows:
        scores=teacher['scores'][row['id']]
        if (len(scores)!=len(row['candidates']) or len(set(row['candidates']))!=len(row['candidates'])
                or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in scores)):
            raise ValueError(f"invalid teacher scores or candidates for {row['id']}")
