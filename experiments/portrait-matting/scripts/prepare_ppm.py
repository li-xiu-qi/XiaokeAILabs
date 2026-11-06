#!/usr/bin/env python3
"""Create the fixed exploratory PPM subset from the PaddleSeg tutorial archive."""
import argparse
import hashlib
import json
import random
import zipfile
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--archive', type=Path, required=True)
p.add_argument('--output', type=Path, required=True, help='new directory, never overwritten')
a = p.parse_args()
expected = '5a3d9029437954594c26ea773e927c4b7c4caaee7e94f6bbd8e96196c868254a'
h = hashlib.sha256()
with a.archive.open('rb') as f:
    for block in iter(lambda: f.read(1048576), b''):
        h.update(block)
if h.hexdigest() != expected:
    raise ValueError('archive SHA-256 mismatch')
rng = random.Random(20261107)
rows = []
a.output.mkdir(parents=True, exist_ok=False)
with zipfile.ZipFile(a.archive) as z:
    for split, count in [('val', 10), ('train', 20)]:
        names = sorted(z.read(f'PPM-100/{split}.txt').decode().splitlines())
        for fg in rng.sample(names, count):
            alpha = fg.replace('/fg/', '/alpha/')
            if not fg.startswith(f'{split}/fg/') or not alpha.startswith(f'{split}/alpha/'):
                raise ValueError(f'unexpected archive path {fg}')
            if len(Path(fg).parts) != 3 or any(part in ('.', '..') for part in Path(fg).parts):
                raise ValueError(f'unsafe archive path {fg}')
            rec = {'sample_id': Path(fg).stem, 'source_split': split}
            for key, rel in [('image', fg), ('ground_truth_alpha', alpha)]:
                dest = a.output / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                data = z.read('PPM-100/' + rel)
                dest.write_bytes(data)
                rec[key] = rel
                rec[key + '_sha256'] = hashlib.sha256(data).hexdigest()
            rows.append(rec)
(a.output / 'manifest.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in rows), encoding='utf-8')
print('sample pairs:', len(rows), 'seed: 20261107')
