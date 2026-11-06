"""Score each fixed BGE candidate with Qwen3-VL-Reranker; all inputs stay local."""
import argparse
import json
import math
from pathlib import Path
import torch
from qwen3_vl_reranker import Qwen3VLReranker

p = argparse.ArgumentParser()
p.add_argument('--candidates', type=Path, required=True)
p.add_argument('--images', type=Path, required=True)
p.add_argument('--model', required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
source = json.loads(a.candidates.read_text(encoding='utf-8'))
done = [json.loads(x) for x in a.output.read_text(encoding='utf-8').splitlines()] if a.output.exists() else []
assert len(done) <= len(source) and all(done[i]['target_filename'] == source[i]['target_filename'] for i in range(len(done)))
torch.cuda.set_per_process_memory_fraction(.20)
model = Qwen3VLReranker(a.model, min_pixels=4096, max_pixels=65536, dtype=torch.bfloat16)
for row in source[len(done):]:
    scores = []
    for start in range(0, 5, 2):
        batch = row['candidates'][start:start + 2]
        images = [a.images / c['filename'] for c in batch]
        assert all(p.is_file() for p in images)
        values = model.process({'query': {'text': row['query']}, 'documents': [{'image': str(p)} for p in images]})
        assert len(values) == len(batch) and all(math.isfinite(s) for s in values)
        scores.extend({**c, 'reranker_score': float(s)} for c, s in zip(batch, values))
    with a.output.open('a', encoding='utf-8') as f:
        f.write(json.dumps({'target_filename': row['target_filename'], 'candidates': scores}, ensure_ascii=False) + '\n')
    print('completed query', flush=True)