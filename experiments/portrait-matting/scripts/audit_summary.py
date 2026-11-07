"""Recompute aggregate metrics from the published per-image summary, without images."""
import json
import math
import statistics
from pathlib import Path


def audit(data):
    rows = data['samples']
    if len(rows) != data['n'] or not rows or len({r['sample'] for r in rows}) != len(rows):
        raise ValueError('Sample count or identifiers are inconsistent')
    result = {}
    for name, stored in data['macro_mean'].items():
        values = [r['metrics'][name] for r in rows if r['metrics'][name] is not None]
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in values) or (stored is not None and (type(stored) not in (int, float) or not math.isfinite(stored))):
            raise ValueError(f'Invalid metric: {name}')
        mean = statistics.mean(values) if values else None
        if (mean is None) != (stored is None) or (mean is not None and not math.isclose(mean, stored, rel_tol=1e-12, abs_tol=1e-12)):
            raise ValueError(f'Aggregate differs from per-image values: {name}')
        result[name] = mean
    if any(len(r['raw_forward_seconds']) != data['repetitions_per_sample'] or not r['raw_forward_seconds'] for r in rows):
        raise ValueError('Unexpected number of forward times')
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for r in rows for v in r['raw_forward_seconds']):
        raise ValueError('Invalid forward time')
    times = [statistics.mean(r['raw_forward_seconds']) for r in rows]
    return {'n': len(rows), 'macro_mean': result,
            'mean_forward_ms': statistics.mean(times) * 1000,
            'min_per_image_mean_ms': min(times) * 1000,
            'max_per_image_mean_ms': max(times) * 1000}


if __name__ == '__main__':
    path = Path(__file__).resolve().parents[1] / 'docs/evidence/summary.json'
    print(json.dumps(audit(json.loads(path.read_text(encoding='utf-8'))), indent=2))
