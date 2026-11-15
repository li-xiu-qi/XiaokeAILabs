"""Encode private image/text pairs with Jina v5; files remain local."""
import argparse
import json
import hashlib
import math
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
    parser.add_argument('--student', help='Student checkpoint, declared before development scoring')
    parser.add_argument('--min-teacher-gap', type=float, help='Predeclared absolute validation R@1 gap, at least 0.05')
    parser.add_argument('--stage', choices=['development', 'test'], default='development',
                        help='Test scoring is permitted only after both student runs complete')
    args = parser.parse_args()
    if args.stage == 'development':
        if not args.student or args.min_teacher_gap is None or not math.isfinite(args.min_teacher_gap) or args.min_teacher_gap < 0.05:
            raise SystemExit('Declare --student and --min-teacher-gap >= 0.05 before development scoring')
        plan_path = args.workdir / 'teacher-plan.json'
        plan = {'teacher':args.teacher, 'student':args.student, 'minimum_gap':args.min_teacher_gap,
                'metric':'text-to-image R@1', 'paired_bootstrap_95ci_lower_gt_zero':True,
                'samples_sha256':hashlib.sha256((args.workdir/'samples-private.json').read_bytes()).hexdigest()}
        if plan_path.exists():
            if json.loads(plan_path.read_text(encoding='utf8')) != plan or (args.workdir/'teacher-val-summary.json').exists():
                raise SystemExit('The plan already exists or the development result was observed; use a fresh work directory')
        else:
            plan_path.write_text(json.dumps(plan,indent=2),encoding='utf8')
    if args.stage == 'test' and not (args.workdir / 'training-complete.json').is_file():
        raise SystemExit('Finish both student training runs before evaluating the test set')
    if args.stage == 'test':
        completed = json.loads((args.workdir / 'training-complete.json').read_text(encoding='utf8'))
        manifest = json.loads((args.workdir / 'teacher-manifest.json').read_text(encoding='utf8'))
        current = hashlib.sha256((args.workdir / 'samples-private.json').read_bytes()).hexdigest()
        if completed['samples_sha256'] != current:
            raise SystemExit('The test samples differ from the completed student run')
        if manifest['teacher'] != args.teacher:
            raise SystemExit('The test teacher differs from the one used during development')
    data = json.loads((args.workdir / 'samples-private.json').read_text(encoding='utf8'))
    start = time.perf_counter()
    check_memory()
    torch.cuda.set_per_process_memory_fraction(.20)
    model = SentenceTransformer(args.teacher, trust_remote_code=True, device='cuda')
    # Sentence Transformers 6 may omit image from this custom model's modalities.
    first = model[0]
    modalities = list(getattr(first, 'modalities', ['text']))
    if 'image' not in modalities:
        first.modalities = modalities + ['image']
    vectors = {}
    splits = ('train', 'val') if args.stage == 'development' else ('test',)
    for split in splits:
        rows = data[split]
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
    dest = 'teacher-private.npz' if args.stage == 'development' else 'teacher-test-private.npz'
    np.savez_compressed(args.workdir / dest, **vectors)
    if args.stage == 'development':
        samples_path = args.workdir / 'samples-private.json'
        manifest = {'samples_sha256': hashlib.sha256(samples_path.read_bytes()).hexdigest(),
                    'vectors_sha256': hashlib.sha256((args.workdir / dest).read_bytes()).hexdigest(),
                    'teacher': args.teacher, 'train_n': len(data['train']), 'val_n': len(data['val'])}
        (args.workdir / 'teacher-manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf8')
    split = 'val' if args.stage == 'development' else 'test'
    text, image = vectors[split + '_text'].copy(), vectors[split + '_image'].copy()
    text /= np.linalg.norm(text, axis=1, keepdims=True)
    image /= np.linalg.norm(image, axis=1, keepdims=True)
    order = np.argsort(-(text @ image.T), axis=1, kind='stable')
    ranks = np.argmax(order == np.arange(len(order))[:, None], axis=1) + 1
    summary = {'split': split, 'n': len(ranks), 'r1': float(np.mean(ranks <= 1)),
               'r5': float(np.mean(ranks <= 5)), 'mrr': float(np.mean(1 / ranks)),
               'duration_including_load_s': time.perf_counter() - start,
               'cuda_peak_allocated': torch.cuda.max_memory_allocated()}
    (args.workdir / ('teacher-' + split + '-summary.json')).write_text(json.dumps(summary, indent=2), encoding='utf8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
