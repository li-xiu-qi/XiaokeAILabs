"""Report hashes and frame-level differences for two matched output folders."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import av
import numpy as np
from pair_contract import validate_pair

parser = argparse.ArgumentParser()
parser.add_argument('--baseline', required=True)
parser.add_argument('--lora', required=True)
args = parser.parse_args()
report = []
baseline = Path(args.baseline)
baseline_files = {p.name: p for p in baseline.glob('*.mp4')}
lora_files = {p.name: p for p in Path(args.lora).glob('*.mp4')}
if not baseline_files or baseline_files.keys() != lora_files.keys():
    raise SystemExit(f'Unequal or empty video sets: baseline-only={sorted(baseline_files.keys() - lora_files.keys())}; '
                     f'lora-only={sorted(lora_files.keys() - baseline_files.keys())}')
metadata_paths = [baseline / 'generation-manifest.json', Path(args.lora) / 'generation-manifest.json']
if all(path.is_file() for path in metadata_paths):
    metadata = [json.loads(path.read_text(encoding='utf-8')) for path in metadata_paths]
    validate_pair(*metadata, baseline_files.keys())
    for manifest, files in zip(metadata, (baseline_files, lora_files)):
        for name, path in files.items():
            if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['video_sha256'].get(path.stem):
                raise ValueError(f'video differs from generation manifest: {path}')
elif any(path.exists() for path in metadata_paths):
    raise ValueError('generation metadata must be present for both conditions or neither')
else:
    print('Warning: no generation metadata; filenames alone do not verify matching prompt and seed.', file=sys.stderr)
for name, a in sorted(baseline_files.items()):
    b = lora_files[name]
    sequences, item = [], {'id': a.stem}
    for label, path in [('baseline', a), ('lora', b)]:
        with av.open(str(path)) as video:
            frames = [f.to_ndarray(format='rgb24') for f in video.decode(video.streams.video[0])]
        if len(frames) != 49:
            raise SystemExit(f'{path}: expected 49 frames, got {len(frames)}')
        if any(frame.shape != frames[0].shape for frame in frames):
            raise SystemExit(f'{path}: inconsistent frame dimensions')
        step_differences = [float(np.abs(frames[i].astype(np.int16)-frames[i-1].astype(np.int16)).mean())
                            for i in range(1, len(frames))]
        item[label] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                       'unique_rgb_frames': len({hashlib.sha256(f.tobytes()).digest() for f in frames}),
                       'adjacent_mae_range': [min(step_differences), max(step_differences)]}
        sequences.append(frames)
    if sequences[0][0].shape != sequences[1][0].shape:
        raise SystemExit(f'{a.name}: frame dimensions differ')
    diffs = [float(np.abs(x.astype(np.int16)-y.astype(np.int16)).mean()) for x, y in zip(*sequences)]
    item['paired_mae_mean'] = float(np.mean(diffs))
    report.append(item)
print(json.dumps(report, indent=2))
