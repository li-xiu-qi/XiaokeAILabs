import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def alpha(path):
    with Image.open(path) as im:
        if im.mode not in ('L', 'RGB', 'RGBA'):
            raise ValueError(f'{path}: expected 8-bit grayscale or replicated RGB; got {im.mode}')
        a = np.asarray(im)
        if a.ndim == 3:
            if not (np.array_equal(a[..., 0], a[..., 1]) and np.array_equal(a[..., 0], a[..., 2])):
                raise ValueError(f'{path}: RGB channels differ; not a grayscale matte')
            if a.shape[2] == 4 and not np.all(a[..., 3] == 255):
                raise ValueError(f'{path}: non-opaque alpha channel; specify a grayscale matte')
            a = a[..., 0]
        return a.astype(np.float64) / 255.0


def dilate(mask, radius):
    out = mask.copy()
    for _ in range(radius):
        p = np.pad(out, 1, constant_values=False)
        h, w = out.shape
        out = np.logical_or.reduce([p[y:y+h, x:x+w] for y in range(3) for x in range(3)])
    return out


def edge(mask):
    p = np.pad(mask, 1, constant_values=False)
    h, w = mask.shape
    eroded = np.logical_and.reduce([p[y:y+h, x:x+w] for y in range(3) for x in range(3)])
    return mask & ~eroded


def metrics(gt, pred, threshold=0.5, tolerance=3):
    if gt.shape != pred.shape or gt.ndim != 2:
        raise ValueError('ground truth and prediction must have identical 2D shapes')
    a, b = gt >= threshold, pred >= threshold
    intersection = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    total = int(a.sum() + b.sum())
    ea, eb = edge(a), edge(b)
    na, nb = int(ea.sum()), int(eb.sum())
    if na == 0 and nb == 0:
        bf = 1.0
    elif na == 0 or nb == 0:
        bf = 0.0
    else:
        precision = np.count_nonzero(eb & dilate(ea, tolerance)) / nb
        recall = np.count_nonzero(ea & dilate(eb, tolerance)) / na
        bf = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    delta = np.abs(gt - pred)
    unknown = (gt > 0) & (gt < 1)
    return {'iou': intersection / union if union else 1.0,
            'dice': 2 * intersection / total if total else 1.0,
            'boundary_f1': bf, 'mae': float(delta.mean()),
            'mse': float((delta ** 2).mean()),
            'sad_1000': float(delta.sum() / 1000.0),
            'unknown_mae': float(delta[unknown].mean()) if unknown.any() else None,
            'unknown_pixels': int(unknown.sum()), 'height': gt.shape[0], 'width': gt.shape[1]}


def main():
    ap = argparse.ArgumentParser(description='Score paired 8-bit alpha mattes; paths relative to manifest.')
    ap.add_argument('manifest', type=Path, help='JSONL: sample_id, ground_truth_alpha, prediction_alpha')
    ap.add_argument('output', type=Path)
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--tolerance', type=int, default=3, help='boundary distance in pixels, Chebyshev metric')
    args = ap.parse_args()
    if not 0 <= args.threshold <= 1 or args.tolerance < 0:
        ap.error('threshold must be in [0, 1] and tolerance must be non-negative')
    records, ids = [], set()
    for line in args.manifest.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = row['sample_id']
        if key in ids:
            raise ValueError(f'duplicate sample ID: {key}')
        ids.add(key)
        base = args.manifest.parent
        result = metrics(alpha(base / row['ground_truth_alpha']), alpha(base / row['prediction_alpha']), args.threshold, args.tolerance)
        records.append({'sample_id': key, **result})
    if not records:
        raise ValueError('manifest is empty')
    keys = ['iou', 'dice', 'boundary_f1', 'mae', 'mse', 'sad_1000', 'unknown_mae']
    summary = {k: (float(np.mean(vals)) if vals else None) for k in keys
               for vals in [[r[k] for r in records if r[k] is not None]]}
    args.output.write_text(json.dumps({'parameters': {'threshold': args.threshold, 'boundary_tolerance_pixels_chebyshev': args.tolerance},
                                      'n': len(records), 'macro_mean': summary, 'samples': records}, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
