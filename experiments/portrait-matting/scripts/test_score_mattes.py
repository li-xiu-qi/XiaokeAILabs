import importlib.util
from pathlib import Path
import numpy as np
p = Path(__file__).parent / 'score_mattes.py'
spec = importlib.util.spec_from_file_location('score_mattes', p)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
z = np.zeros((9, 9)); a = z.copy(); a[2:7, 2:7] = 1
assert m.metrics(z, z)['iou'] == 1
assert m.metrics(z, a)['boundary_f1'] == 0
perfect = m.metrics(a, a)
assert perfect['iou'] == perfect['dice'] == perfect['boundary_f1'] == 1
assert perfect['mae'] == perfect['mse'] == 0 and perfect['unknown_mae'] is None
b = np.roll(a, 1, axis=1)
assert m.metrics(a, b, tolerance=0)['boundary_f1'] < 1
assert m.metrics(a, b, tolerance=1)['boundary_f1'] == 1
x = z.copy(); x[1, 1] = .25
assert abs(m.metrics(x, z)['unknown_mae'] - .25) < 1e-10
try:
    m.metrics(a, a[:3])
    raise AssertionError('shape mismatch passed')
except ValueError:
    pass
print('7 synthetic metric invariants passed')
