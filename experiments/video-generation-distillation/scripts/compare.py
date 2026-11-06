"""Verify hashes and report frame-level differences for two output folders."""
import argparse
import hashlib
import json
from pathlib import Path
import av
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--baseline', required=True)
parser.add_argument('--lora', required=True)
args = parser.parse_args()
report = []
baseline = Path(args.baseline)
for a in sorted(baseline.glob('*.mp4')):
    b = Path(args.lora) / a.name
    if not b.is_file():
        raise SystemExit(f'Missing comparison for {a.name}')
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
if not report:
    raise SystemExit('No baseline MP4 files found')
print(json.dumps(report, indent=2))
