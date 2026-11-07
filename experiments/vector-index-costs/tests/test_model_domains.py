"""Reject invalid budgets instead of returning negative or non-finite estimates."""
import importlib.util
import sys
import unittest
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "scripts" / "vector_index_model.py"
spec = importlib.util.spec_from_file_location("vector_index_model", path)
model = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = model
spec.loader.exec_module(model)

class ModelDomainTests(unittest.TestCase):
    def test_storage_rejects_invalid_graph_and_centroid_sizes(self):
        for values in ({"hnsw_m": -100}, {"ivf_nlist": -1}, {"hnsw_m": 0}, {"ivf_nlist": 1.5}, {"n": True}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                model.compute_storage(model.VectorIndexSpec(**values))

    def test_latency_rejects_nonfinite_and_fractional_parameters(self):
        for values in ({"compute_q": float("nan")}, {"compute_q": float("inf")},
                       {"compute_q": True}, {"hnsw_ef": 1.5}, {"compute_q_dim": 0}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                model.compute_latency(model.VectorIndexSpec(**values))

    def test_valid_estimates_match_known_totals(self):
        s = model.VectorIndexSpec(n=20000, d=128, pq_m=8, pq_nbits=8, ivf_nlist=64)
        result = model.compute_storage(s)
        self.assertEqual(result["flat"], 10240000)
        self.assertEqual(result["pq"], 291072)
        self.assertGreater(model.compute_latency(s)["flat_ms"], 0)

if __name__ == "__main__":
    unittest.main()
