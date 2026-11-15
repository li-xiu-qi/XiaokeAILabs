"""Check converter safety gates with mocked dependencies, without model weights."""
import contextlib
import json
import runpy
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'convert_document_text_ir.py'
REFERENCE = np.array([[1., 2.], [2., 1.], [1., 1.]], dtype=np.float32)


def fakes(valid):
    class Value:
        def cpu(self): return self
        def numpy(self): return REFERENCE
    class TextModel:
        _default_loraid = 1
        @property
        def default_loraid(self): return self._default_loraid
        def __call__(self, _inputs):
            if self._default_loraid is not None:
                raise AssertionError('LoRA was not disabled')
            return Value()
    class Model:
        def __init__(self): self.text_model = TextModel()
        def eval(self): return self
    class AutoModel:
        @staticmethod
        def from_pretrained(*_args, **_kwargs): return Model()
    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return lambda *_a, **_k: {'input_ids': np.zeros((3, 512), dtype=np.int64)}
    torch = types.ModuleType('torch')
    torch.float32 = 'float32'
    torch.no_grad = contextlib.nullcontext
    transformers = types.ModuleType('transformers')
    transformers.AutoModel, transformers.AutoTokenizer = AutoModel, AutoTokenizer
    ov = types.ModuleType('openvino')
    def convert(model, example_input):
        if model.default_loraid is not None:
            raise AssertionError('LoRA was not disabled for conversion')
        return object()
    def save(_model, path, compress_to_fp16):
        if not compress_to_fp16: raise AssertionError('expected FP16 IR')
        Path(path).write_text('mock', encoding='utf-8')
    class Core:
        def compile_model(self, _path, _device):
            return lambda _inputs: {'output': REFERENCE.copy()}
    ov.convert_model, ov.save_model, ov.Core = convert, save, Core
    helper = types.ModuleType('run_openvino_smoke')
    helper.MODEL_ID, helper.MODEL_REVISION = 'mock', 'revision'
    helper.CODE_REVISIONS = {'jinaai/jina-clip-implementation': 'code'}
    helper.TEXTS = ['a', 'b', 'c']
    helper.set_static_input_shape = lambda *_args: None
    helper.cosine_diff = lambda *_args: {'max_abs': 0., 'min_row_cosine': 1.}
    helper.native_matches = lambda _diff: valid
    return {'torch': torch, 'transformers': transformers, 'openvino': ov,
            'run_openvino_smoke': helper}


class DocumentConversionTest(unittest.TestCase):
    def run_converter(self, out, valid=True):
        argv = [str(SCRIPT), '--preprocessor', 'mock', '--cache-dir', 'mock',
                '--out', str(out), '--offline']
        with patch.dict(sys.modules, fakes(valid)), patch.object(sys, 'argv', argv):
            runpy.run_path(str(SCRIPT), run_name='__main__')

    def test_success_and_existing_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'ir'
            self.run_converter(out)
            self.assertTrue((out / 'jina-clip-text_v2_fp16.xml').is_file())
            self.assertTrue(json.loads((out / 'validation.json').read_text())['ok'])
            with self.assertRaises(SystemExit) as error:
                self.run_converter(out)
            self.assertEqual(error.exception.code, 2)

    def test_failed_validation_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'ir'
            with self.assertRaises(SystemExit) as error:
                self.run_converter(out, valid=False)
            self.assertTrue(error.exception.code)
            self.assertFalse(json.loads((out / 'validation.json').read_text())['ok'])


if __name__ == '__main__':
    unittest.main()
