"""Exercise API error handling without loading a model or browser."""
import base64
import io
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
rm = types.ModuleType('rm_render')
rm.CATEGORIES = ('text', 'formula', 'table')
class Candidate:
    def __init__(self, **kwargs): self.__dict__.update(kwargs)
rm.Candidate = Candidate
rm.render_candidates = lambda *a, **kw: []
js = types.ModuleType('jev_score')
js.JevScorer = object
js.build_prompt = lambda *a: None
js._assert_prompt_unchanged = lambda: 'stub'
sys.modules['rm_render'] = rm
sys.modules['jev_score'] = js
import jev_server
from fastapi.testclient import TestClient

class ApiTests(unittest.TestCase):
    def setUp(self):
        buf = io.BytesIO()
        Image.new('RGB', (2, 2), 'white').save(buf, format='PNG')
        self.image = base64.b64encode(buf.getvalue()).decode()
        self.client = TestClient(jev_server.app)

    def test_all_failed_returns_422_for_both_endpoints(self):
        failed = [types.SimpleNamespace(ok=False, render_image_path=None)]
        with patch.object(jev_server, 'render_candidates', return_value=failed):
            for route in ('/ui/score', '/v1/rm/score_best'):
                r = self.client.post(route, json={'image': self.image, 'candidates':['candidate']})
                self.assertEqual(r.status_code, 422)
                self.assertEqual(r.json()['detail'], 'No candidate could be scored')
            r = self.client.post('/v1/rm/score_best_batch', json={'items':[{'image':self.image,'candidates':['candidate']}]})
            self.assertEqual(r.status_code, 422)

    def test_nonfinite_score_is_excluded(self):
        class Scorer:
            def _forward_one(self, *args): return (float('nan'), None)
        with patch.object(jev_server, '_scorer', Scorer()):
            r = self.client.post('/ui/score', json={'image': self.image, 'candidates':['a'],
                                                    'render_images':[self.image]})
        self.assertEqual(r.status_code, 422)

    def test_partial_failure_selects_only_scored_candidate(self):
        class Scorer:
            def _forward_one(self, *args): return (0.2, None)
        with patch.object(jev_server, '_scorer', Scorer()):
            # First render fails, second is supplied by the client.
            failed = [types.SimpleNamespace(ok=False, render_image_path=None)]
            with patch.object(jev_server, 'render_candidates', return_value=failed):
                r = self.client.post('/ui/score', json={'image': self.image, 'candidates':['a','b'],
                                                        'render_images':['',self.image]})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()['best_index'], 1)
        self.assertEqual(r.json()['verdicts'], ['render_error','scored'])

if __name__ == '__main__': unittest.main()
