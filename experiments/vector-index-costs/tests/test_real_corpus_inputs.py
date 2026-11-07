"""Input integrity checks do not require Faiss or any network access."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_recall_real_corpus.py"
spec = importlib.util.spec_from_file_location("real_corpus", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class InputChecks(unittest.TestCase):
    def test_metadata_conflicts_and_missing_fields(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base, query = root / "base", root / "query"
            base.mkdir()
            query.mkdir()
            first = {"model": "a", "dim": 2, "replicate": 1, "dtype": "float32"}
            (base / "meta.json").write_text(json.dumps(first), encoding="utf8")
            for key, invalid in (("model", "b"), ("replicate", 2), ("dtype", "float16")):
                second = dict(first, **{key: invalid})
                (query / "meta.json").write_text(json.dumps(second), encoding="utf8")
                with self.subTest(field=key), self.assertRaisesRegex(ValueError, key):
                    module.load_metadata(str(base / "base.bin"), str(query / "query.bin"))
            (query / "meta.json").write_text(json.dumps({"n_query": 1}), encoding="utf8")
            metadata, _ = module.load_metadata(str(base / "base.bin"), str(query / "query.bin"))
            self.assertEqual(metadata, first)  # missing fields do not prove consistency

    def test_finite_values_and_claimed_normalization(self):
        module.validate_vectors(np.array([[1., 0.], [0., 1.]], dtype=np.float32), "query", True)
        for invalid in (np.nan, np.inf):
            with self.assertRaisesRegex(ValueError, "NaN or infinity"):
                module.validate_vectors(np.array([[invalid, 0.]], dtype=np.float32), "query")
        with self.assertRaisesRegex(ValueError, "normalized"):
            module.validate_vectors(np.array([[2., 0.]], dtype=np.float32), "query", True)
        module.validate_vectors(np.array([[2., 0.]], dtype=np.float32), "query", False)


if __name__ == "__main__":
    unittest.main()
