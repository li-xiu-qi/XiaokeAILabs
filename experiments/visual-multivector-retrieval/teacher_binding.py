"""Bind teacher scores to the image bytes used to produce them."""
import hashlib
import math


def image_hashes(root, batches):
    paths = {row['image'] for batch in batches for row in batch}
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sorted(paths)}


def verify_teacher(root, teacher):
    batches = teacher['batches']
    if teacher.get('steps') != len(batches) or not batches:
        raise ValueError('Teacher step count does not match its batches')
    for entry in batches:
        batch, scores = entry['batch'], entry['scores']
        if len(batch) != 2 or len({r['image'] for r in batch}) != 2 or any(r['split'] != 'train' for r in batch):
            raise ValueError('Teacher batch must contain two different training images')
        if len(scores) != 2 or any(len(row) != 2 or any(not math.isfinite(x) or not 0 <= x <= 1 for x in row) for row in scores):
            raise ValueError('Teacher scores must be a finite 2x2 probability matrix')
    expected = teacher.get('image_sha256')
    if not expected or expected != image_hashes(root, [entry['batch'] for entry in batches]):
        raise ValueError('Teacher image hashes missing or do not match; regenerate teacher scores')
