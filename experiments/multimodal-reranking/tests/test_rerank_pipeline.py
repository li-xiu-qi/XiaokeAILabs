"""Exercise the public ranking pipeline without downloading a model or data."""
import contextlib
import io
import json
from pathlib import Path
import runpy
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'


class PipelineTest(unittest.TestCase):
    def test_pipeline_and_resume_guard(self):
        class FakeReranker:
            def __init__(self, *args, **kwargs):
                pass

            def process(self, inputs):
                return [float(Path(doc['image']).stem) for doc in inputs['documents']]

        fake_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(set_per_process_memory_fraction=lambda fraction: None),
            bfloat16=object(),
        )
        fake_qwen = types.SimpleNamespace(Qwen3VLReranker=FakeReranker)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            images = root / 'images'
            images.mkdir()
            for n in range(5):
                (images / f'{n}.jpg').write_bytes(b'fake image')
            candidates = root / 'candidates.json'
            rows = [{'query': f'query {i}', 'target_filename': '0.jpg', 'candidates': [
                {'filename': f'{n}.jpg', 'embedding_score': float(5 - n)} for n in range(5)
            ]} for i in range(100)]
            candidates.write_text(json.dumps(rows), encoding='utf-8')
            reranked = root / 'reranked.jsonl'
            anonymous = root / 'anonymous.json'
            args = ['rerank.py', '--candidates', str(candidates), '--images', str(images),
                    '--model', 'fake-model', '--output', str(reranked)]
            with patch.dict(sys.modules, {'torch': fake_torch, 'qwen3_vl_reranker': fake_qwen}), \
                    patch.object(sys, 'argv', args), contextlib.redirect_stdout(io.StringIO()):
                runpy.run_path(str(SCRIPTS / 'rerank.py'), run_name='__main__')
                first = reranked.read_bytes()
                self.assertEqual(len(first.splitlines()), 100)
                runpy.run_path(str(SCRIPTS / 'rerank.py'), run_name='__main__')
                self.assertEqual(reranked.read_bytes(), first)
                # Legacy rows lack the query/model identity required for safe resume.
                legacy = [json.loads(line) for line in first.splitlines()]
                legacy[0].pop('query')
                legacy[0].pop('model')
                reranked.write_text(''.join(json.dumps(row) + chr(10) for row in legacy), encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'does not match'):
                    runpy.run_path(str(SCRIPTS / 'rerank.py'), run_name='__main__')
                reranked.write_bytes(first)
                rows[0]['query'] = 'changed query'
                candidates.write_text(json.dumps(rows), encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'does not match'):
                    runpy.run_path(str(SCRIPTS / 'rerank.py'), run_name='__main__')
                self.assertEqual(reranked.read_bytes(), first)
            rows[0]['query'] = 'query 0'
            candidates.write_text(json.dumps(rows), encoding='utf-8')
            with patch.object(sys, 'argv', ['anonymize.py', '--candidates', str(candidates),
                    '--reranked', str(reranked), '--output', str(anonymous)]):
                runpy.run_path(str(SCRIPTS / 'anonymize.py'), run_name='__main__')
            self.assertEqual(len(json.loads(anonymous.read_text(encoding='utf-8'))), 100)
            output = io.StringIO()
            with patch.object(sys, 'argv', ['summarize.py', str(anonymous)]), \
                    contextlib.redirect_stdout(output):
                runpy.run_path(str(SCRIPTS / 'summarize.py'), run_name='__main__')
            stats = json.loads(output.getvalue())
            self.assertEqual((stats['n'], stats['before_hit1'], stats['after_hit1']), (100, 100, 0))
            # The optional small-run count must reach both public downstream CLIs.
            candidates.write_text(json.dumps(rows[:5]), encoding='utf-8')
            reranked.write_bytes(b''.join(first.splitlines(keepends=True)[:5]))
            small = root / 'small.json'
            with patch.object(sys, 'argv', ['anonymize.py', '--candidates', str(candidates),
                    '--reranked', str(reranked), '--output', str(small), '--expected-count', '5']):
                runpy.run_path(str(SCRIPTS / 'anonymize.py'), run_name='__main__')
            output = io.StringIO()
            with patch.object(sys, 'argv', ['summarize.py', str(small), '--expected-count', '5']), \
                    contextlib.redirect_stdout(output):
                runpy.run_path(str(SCRIPTS / 'summarize.py'), run_name='__main__')
            self.assertEqual(json.loads(output.getvalue())['n'], 5)


    def test_vision_processing_failure_stops_reranking(self):
        def fail_vision(*args, **kwargs):
            raise ValueError('invalid image')

        fake_torch = types.SimpleNamespace(
            no_grad=lambda: (lambda f: f),
            nn=types.SimpleNamespace(Linear=object),
        )
        fake_scipy = types.ModuleType('scipy')
        fake_scipy.special = types.ModuleType('scipy.special')
        fake_vision = types.SimpleNamespace(process_vision_info=fail_vision)
        fake_transformers = types.SimpleNamespace(
            Qwen3VLForConditionalGeneration=object, AutoProcessor=object)
        modules = {'torch': fake_torch, 'numpy': types.ModuleType('numpy'),
                   'scipy': fake_scipy, 'scipy.special': fake_scipy.special,
                   'qwen_vl_utils': fake_vision, 'transformers': fake_transformers}
        with patch.dict(sys.modules, modules):
            namespace = runpy.run_path(str(SCRIPTS / 'qwen3_vl_reranker.py'))
            model = namespace['Qwen3VLReranker'].__new__(namespace['Qwen3VLReranker'])
            model.max_length = 32
            model.processor = types.SimpleNamespace(apply_chat_template=lambda *a, **kw: 'input')
            with self.assertRaisesRegex(RuntimeError, 'failed to process image or video') as error:
                model.tokenize([[]])
            self.assertIsInstance(error.exception.__cause__, ValueError)


if __name__ == '__main__':
    unittest.main()