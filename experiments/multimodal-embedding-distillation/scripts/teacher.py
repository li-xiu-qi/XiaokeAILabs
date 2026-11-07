"""Encode private image/text pairs with Jina v5; files remain local."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer


def check_memory():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:') and int(line.split()[1]) < 50 * 1024 * 1024:
            raise RuntimeError('MemAvailable below 50 GiB')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workdir', type=Path, required=True)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--teacher', required=True)
    args = parser.parse_args()
    sample_bytes = (args.workdir / 'samples-private.json').read_bytes()
    data = json.loads(sample_bytes)
    if (args.workdir / 'teacher-private.npz').exists() or (args.workdir / 'teacher-summary.json').exists():
        raise FileExistsError('teacher outputs already exist; use a fresh workdir')
    start = time.perf_counter()
    check_memory()
    torch.cuda.set_per_process_memory_fraction(.20)
    model = SentenceTransformer(args.teacher, trust_remote_code=True, device='cuda')
    # Sentence Transformers 6 may omit image from this custom model's modalities.
    first = model[0]
    modalities = list(getattr(first, 'modalities', ['text']))
    if 'image' not in modalities:
        first.modalities = modalities + ['image']
    vectors = {'samples_sha256': np.array(hashlib.sha256(sample_bytes).hexdigest())}
    for split, rows in data.items():
        all_text, all_image = [], []
        for offset in range(0, len(rows), 4):
            check_memory()
            batch = rows[offset:offset + 4]
            text = np.asarray(model.encode_query([r['caption'] for r in batch], task='retrieval',
                batch_size=4, normalize_embeddings=False, show_progress_bar=False))
            images = []
            for row in batch:
                with Image.open(args.images / row['filename']) as img:
                    images.append(img.convert('RGB'))
            image = np.asarray(model.encode(images, task='retrieval', batch_size=4,
                normalize_embeddings=False, show_progress_bar=False))
            if text.shape != image.shape or not np.isfinite(text).all() or not np.isfinite(image).all():
                raise ValueError('Teacher returned mismatched or non-finite vectors')
            all_text.append(text.astype('float32'))
            all_image.append(image.astype('float32'))
        vectors[split + '_text'] = np.concatenate(all_text)
        vectors[split + '_image'] = np.concatenate(all_image)
    np.savez_compressed(args.workdir / 'teacher-private.npz', **vectors)
    text, image = vectors['test_text'].copy(), vectors['test_image'].copy()
    text /= np.linalg.norm(text, axis=1, keepdims=True)
    image /= np.linalg.norm(image, axis=1, keepdims=True)
    order = np.argsort(-(text @ image.T), axis=1, kind='stable')
    ranks = np.argmax(order == np.arange(len(order))[:, None], axis=1) + 1
    summary = {'test_n': len(ranks), 'r1': float(np.mean(ranks <= 1)),
               'r5': float(np.mean(ranks <= 5)), 'mrr': float(np.mean(1 / ranks)),
               'duration_including_load_s': time.perf_counter() - start,
               'cuda_peak_allocated': torch.cuda.max_memory_allocated()}
    (args.workdir / 'teacher-summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
