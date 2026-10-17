import os, time
os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['OMP_NUM_THREADS']='4'
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np

t0=time.perf_counter()
m = SentenceTransformer('models/all-MiniLM-L6-v2')
print('model load:', round(time.perf_counter()-t0,1), 's', flush=True)

train = Path('data/20news/20news-bydate-train')
texts = []
for cat in sorted(train.iterdir()):
    for fp in sorted(cat.iterdir()):
        t = fp.read_text(encoding='utf-8', errors='ignore').strip()
        if len(t) >= 50:
            texts.append(t[:2000])
        if len(texts) >= 200: break
    if len(texts) >= 200: break
print('sampled', len(texts), 'texts, avg len', int(np.mean([len(t) for t in texts])), flush=True)

for bs in (32, 64):
    t0=time.perf_counter()
    e = m.encode(texts, batch_size=bs, show_progress_bar=False)
    dt=time.perf_counter()-t0
    print(f'batch={bs}: {len(texts)/dt:.0f} docs/s -> full 18846 in {18846/(len(texts)/dt)/60:.1f} min', flush=True)
