#!/usr/bin/env python3
"""Private PPM RVM single-frame benchmark; model execution requires scheduled device access."""
import argparse
import hashlib
import json
import math
import platform
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, __version__ as pillow_version


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_dataset(manifest):
    rows = [json.loads(line) for line in manifest.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    seen = set()
    if not rows:
        raise ValueError('empty manifest')
    for row in rows:
        sid = row['sample_id']
        if not re.fullmatch(r'[A-Za-z0-9_-]+', sid) or sid in seen:
            raise ValueError(f'invalid or duplicate sample_id {sid!r}')
        seen.add(sid)
        for key in ('image', 'ground_truth_alpha'):
            path = manifest.parent / row[key]
            if not path.is_file() or sha256(path) != row[key + '_sha256']:
                raise ValueError(f'missing or changed source: {path}')
    return rows


def prepare_pair(image_path, alpha_path, max_side):
    from score_mattes import alpha
    with Image.open(image_path) as im:
        # Do not auto-rotate independently: original annotations use source raster coordinates.
        image = im.convert('RGB')
    # Enforce the same 8-bit, grayscale-only input contract as the scorer.
    matte = Image.fromarray(np.rint(alpha(alpha_path) * 255).astype(np.uint8), 'L')
    if image.size != matte.size:
        raise ValueError(f'input/label dimensions differ: {image_path}, {alpha_path}')
    original = image.size
    factor = min(1.0, max_side / max(original))
    size = tuple(max(1, int(round(d * factor))) for d in original)
    if size != original:
        image = image.resize(size, Image.Resampling.LANCZOS)
        matte = matte.resize(size, Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.uint8), np.asarray(matte, dtype=np.uint8), original


def run_dataset(manifest, output, predictor, warmup, repeats, max_side):
    """predictor.predict(rgb, ratio, warmup, repeats) -> (float alpha, times_s, ratio)."""
    from score_mattes import metrics
    rows = read_dataset(manifest)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'alpha').mkdir()
    (output / 'target').mkdir()
    records = []
    for row in rows:
        sid = row['sample_id']
        rgb, target, original = prepare_pair(manifest.parent / row['image'], manifest.parent / row['ground_truth_alpha'], max_side)
        # Match RVM's recommendation of ~512 pixels on the longest downsampled side.
        ratio = min(1.0, 512.0 / max(rgb.shape[:2]))
        prediction, times = predictor.predict(rgb, ratio, warmup, repeats)
        if prediction.shape != target.shape or not np.isfinite(prediction).all():
            raise ValueError(f'{sid}: invalid alpha shape or values')
        # Preserve the exact 8-bit matte used for metrics; conversion is explicit and reproducible.
        pred8 = np.rint(np.clip(prediction, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(pred8, 'L').save(output / 'alpha' / f'{sid}.png')
        Image.fromarray(target, 'L').save(output / 'target' / f'{sid}.png')
        m = metrics(target.astype(np.float64) / 255, pred8.astype(np.float64) / 255)
        records.append({'sample_id': sid, 'source_split': row.get('source_split'),
                        'image_sha256': row['image_sha256'], 'ground_truth_sha256': row['ground_truth_alpha_sha256'],
                        'original_size_wh': list(original), 'inference_size_wh': [rgb.shape[1], rgb.shape[0]],
                        'downsample_ratio': ratio, 'raw_forward_seconds': times,
                        'mean_forward_seconds': statistics.mean(times), 'metrics': m,
                        'alpha': f'alpha/{sid}.png', 'resized_target': f'target/{sid}.png'})
        print(f'{len(records)}/{len(rows)} {sid}', flush=True)
    keys = ('iou', 'dice', 'boundary_f1', 'mae', 'mse', 'sad_1000', 'unknown_mae')
    summary = {}
    for key in keys:
        vals = [r['metrics'][key] for r in records if r['metrics'][key] is not None]
        summary[key] = statistics.mean(vals) if vals else None
    return records, summary


class RVMPredictor:
    def __init__(self, repo, weights, device):
        import torch
        self.torch = torch
        self.device = torch.device(device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        sys.path.insert(0, str(repo.resolve()))
        from model import MattingNetwork
        self.model = MattingNetwork('mobilenetv3').eval().to(self.device)
        self.model.load_state_dict(torch.load(weights, map_location=self.device, weights_only=True))

    def predict(self, rgb, ratio, warmup, repeats):
        torch = self.torch
        src = torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0).to(self.device, dtype=torch.float32).div_(255)
        def sync():
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
        times = []
        with torch.inference_mode():
            for i in range(warmup + repeats):
                sync()
                start = time.perf_counter()
                _, pha, *_ = self.model(src, None, None, None, None, ratio)
                sync()
                elapsed = time.perf_counter() - start
                if i >= warmup:
                    times.append(elapsed)
            alpha = pha[0, 0].detach().cpu().numpy()
        return alpha, times


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path, help='new private output directory; never overwritten')
    p.add_argument('--rvm-repo', required=True, type=Path)
    p.add_argument('--weights', required=True, type=Path)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    p.add_argument('--max-side', type=int, default=1024)
    p.add_argument('--warmup', type=int, default=1)
    p.add_argument('--repeats', type=int, default=3)
    args = p.parse_args()
    if args.output.exists() or args.max_side < 32 or args.warmup < 0 or args.repeats < 1:
        p.error('output must not exist; max-side >= 32, warmup >= 0 and repeats >= 1')
    # Refuse a wrong checkpoint rather than attributing its results to the known smoke-test weights.
    expected = '3c7c1d92033f7c38d6577c481d13a195d7d80a159b960f4f3119ac7b534cf4f8'
    actual = sha256(args.weights)
    if actual != expected:
        raise ValueError(f'weights checksum differs from smoke checkpoint: {actual}')
    revision = subprocess.run(["git", "-C", str(args.rvm_repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    if revision != "53d74c6826735f01f4406b5ca9075eee27bec094":
        raise ValueError(f"unexpected RVM repository revision: {revision}")
    predictor = RVMPredictor(args.rvm_repo, args.weights, args.device)
    records, summary = run_dataset(args.manifest, args.output, predictor, args.warmup, args.repeats, args.max_side)
    metadata = {'created_utc': datetime.now(timezone.utc).isoformat(), 'model': 'RVM MobileNetV3',
        'repo_commit': revision,
        'weights_sha256': actual, 'manifest_sha256': sha256(args.manifest), 'device': str(predictor.device),
        'torch': predictor.torch.__version__, 'cuda_runtime': predictor.torch.version.cuda,
        'device_name': predictor.torch.cuda.get_device_name(predictor.device) if args.device == 'cuda' else platform.processor(),
        'python': platform.python_version(), 'numpy': np.__version__, 'pillow': pillow_version,
        'preprocess': {'rgb': '8-bit RGB / 255; no EXIF rotation; fit longest side to max_side with LANCZOS',
                       'target': '8-bit grayscale; same resize with BILINEAR', 'max_side': args.max_side,
                       'downsample_ratio': 'min(1, 512 / resized longest side)', 'state': 'four None values per forward'},
        'warmups_per_sample': args.warmup, 'repetitions_per_sample': args.repeats,
        'timing': 'synchronous model forward only; excludes image decode, host-to-device, alpha copy and save',
        'metric': 'alpha rounded to 8-bit PNG; threshold 0.5; boundary Chebyshev tolerance 3 resized pixels',
        'scope': 'single-frame exploratory RVM run; data provenance and license depend on the supplied manifest',
        'n': len(records), 'macro_mean': summary, 'samples': records}
    (args.output / 'results.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
