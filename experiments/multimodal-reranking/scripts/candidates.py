"""Build BGE top-5 candidates from a 100-image pool; data remains local."""
import argparse
import csv
import json
import random
from pathlib import Path
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from validate_inputs import validate_sample, validate_norms, first_caption

p = argparse.ArgumentParser()
p.add_argument('--annotations', type=Path, required=True)
p.add_argument('--images', type=Path, required=True)
p.add_argument('--model', required=True, help='local BGE-VL-base snapshot')
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
with a.annotations.open(encoding='utf-8', newline='') as f:
    rows = random.Random(20261107).sample([r for r in csv.DictReader(f) if r['split'] == 'test'], 100)
validate_sample(rows)
queries = [first_caption(row['raw']) for row in rows]
model = SentenceTransformer(a.model, trust_remote_code=True)
first = model[0]
mods = list(getattr(first, 'modalities', ['text']))
if 'image' not in mods:
    first.modalities = mods + ['image']
images = []
for row in rows:
    with Image.open(a.images / row['filename']) as im:
        images.append(im.convert('RGB'))
te = np.asarray(model.encode(queries, task='retrieval', normalize_embeddings=False, batch_size=8, show_progress_bar=False), dtype=np.float64)
ie = np.asarray(model.encode(images, task='retrieval', normalize_embeddings=False, batch_size=8, show_progress_bar=False), dtype=np.float64)
tn = np.linalg.norm(te, axis=1, keepdims=True)
inorm = np.linalg.norm(ie, axis=1, keepdims=True)
validate_norms(tn.ravel())
validate_norms(inorm.ravel())
te /= tn
ie /= inorm
scores = te @ ie.T
result = []
for i, row in enumerate(rows):
    order = np.argsort(-scores[i], kind='stable')[:5]
    result.append({'query': queries[i], 'target_filename': row['filename'], 'candidates': [
        {'filename': rows[int(j)]['filename'], 'embedding_score': float(scores[i, j])} for j in order]})
if a.output.exists():
    raise FileExistsError(a.output)
a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
