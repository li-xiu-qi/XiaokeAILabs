"""Check model scripts do not override the Hugging Face Hub endpoint."""
import os
import runpy
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'


class EndpointTest(unittest.TestCase):
    def test_import_leaves_endpoint_unchanged(self):
        torch = types.ModuleType('torch')
        transformers = types.ModuleType('transformers')
        for name in ('AutoImageProcessor', 'AutoModel', 'AutoTokenizer'):
            setattr(transformers, name, object())
        modules = {'torch': torch, 'transformers': transformers,
                   'openvino': types.ModuleType('openvino')}
        for filename in ('run_openvino_smoke.py', 'benchmark_native_baseline.py',
                         'compare_lora_modes.py'):
            for configured in (None, 'https://example.invalid/custom'):
                with self.subTest(script=filename, endpoint=configured):
                    env = dict(os.environ)
                    env.pop('HF_ENDPOINT', None)
                    if configured is not None:
                        env['HF_ENDPOINT'] = configured
                    with patch.dict(os.environ, env, clear=True), \
                         patch.dict(sys.modules, modules), \
                         patch.object(sys, 'argv', [filename]):
                        runpy.run_path(str(SCRIPTS / filename), run_name='endpoint_test')
                        self.assertEqual(os.environ.get('HF_ENDPOINT'), configured)


if __name__ == '__main__':
    unittest.main()
