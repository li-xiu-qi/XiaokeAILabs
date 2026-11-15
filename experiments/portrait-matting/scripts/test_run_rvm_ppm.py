import hashlib
import importlib.util
import json
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image

here = Path(__file__).parent
spec = importlib.util.spec_from_file_location('runner', here / 'run_rvm_ppm.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
class Fake:
    def predict(self, rgb, ratio, warmup, repeats):
        assert ratio == 1.0 and warmup == 1 and repeats == 2
        assert rgb.shape == (16, 32, 3)
        return np.ones((16, 32), dtype=float), [0.01, 0.02]
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    Image.fromarray(np.full((32, 64, 3), 128, dtype=np.uint8)).save(root / 'image.png')
    Image.fromarray(np.full((32, 64), 255, dtype=np.uint8)).save(root / 'target.png')
    h = lambda name: hashlib.sha256((root / name).read_bytes()).hexdigest()
    row = {'sample_id': 'synthetic', 'image': 'image.png', 'image_sha256': h('image.png'),
           'ground_truth_alpha': 'target.png', 'ground_truth_alpha_sha256': h('target.png')}
    manifest = root / 'manifest.jsonl'
    manifest.write_text(json.dumps(row) + '\n', encoding='utf-8')
    records, summary = runner.run_dataset(manifest, root / 'run', Fake(), 1, 2, 32)
    assert records[0]['original_size_wh'] == [64, 32]
    assert records[0]['inference_size_wh'] == [32, 16]
    assert records[0]['raw_forward_seconds'] == [0.01, 0.02]
    assert summary['iou'] == 1 and summary['mae'] == 0
    assert Image.open(root / 'run/alpha/synthetic.png').size == (32, 16)
    try:
        runner.run_dataset(manifest, root / 'run', Fake(), 1, 2, 32)
        raise AssertionError('overwriting output succeeded')
    except FileExistsError:
        pass
    (root / 'target.png').write_bytes(b'changed')
    try:
        runner.read_dataset(manifest)
        raise AssertionError('changed source accepted')
    except ValueError:
        pass
print('synthetic pipeline, metrics, overwrite and integrity checks passed')
