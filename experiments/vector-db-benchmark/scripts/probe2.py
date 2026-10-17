import os, time
os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np

m = SentenceTransformer('models/all-MiniLM-L6-v2')
train = Path('data/20news/20news-bydate-train')
texts = []
for cat in sorted(train.iterdir()):
    for fp in sorted(cat.iterdir()):
        t = fp.read_text(encoding='utf-8', errors='ignore').strip()
        if len(t) >= 50: texts.append(t[:2000])
        if len(texts) >= 300: break
    if len(texts) >= 300: break

# 关键实验: 更长的 max_seq_length 分桶 + 关闭 padding 到 512
print('default max_seq_length:', m.max_seq_length, 'tokenizer max_len:', m.tokenizer.model_max_length, flush=True)
import torch
torch.set_num_threads(16)

# 1) 默认 512 padding
for bs in (64, 128):
    t0=time.perf_counter(); m.encode(texts, batch_size=bs, show_progress_bar=False); dt=time.perf_counter()-t0
    print(f'default bs={bs}: {len(texts)/dt:.0f} docs/s', flush=True)

# 2) truncate_dim / convert 优化
t0=time.perf_counter(); m.encode(texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True); dt=time.perf_counter()-t0
print(f'default bs=64 numpy: {len(texts)/dt:.0f} docs/s', flush=True)

# 3) 检查 token 长度分布
enc = m.tokenizer(texts[:50], padding=False, truncation=False, return_length=True)
lens = [len(e) for e in enc['input_ids']]
print('token len: p50', int(np.percentile(lens,50)), 'p90', int(np.percentile(lens,90)), 'max', max(lens), flush=True)
