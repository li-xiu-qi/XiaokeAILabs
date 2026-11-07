import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('safe_sqlite_bench', ROOT / 'scripts/safe_sqlite_bench.py')
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)

class SafetyTests(unittest.TestCase):
    def test_search_uses_only_memory_database(self):
        real_connect = bench.sqlite3.connect
        with patch.object(bench.sqlite3, 'connect', wraps=real_connect) as connect:
            result = bench.run(vectors=100, queries=5, dimensions=8)
        connect.assert_called_once_with(':memory:')
        self.assertEqual(result['recall_at_10_mean'], 1.0)

    def test_recall_credits_alternative_equal_distance_neighbors(self):
        distances = bench.np.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 9., 10.], dtype=bench.np.float32)
        self.assertEqual(bench.recall_with_boundary_ties(distances, list(range(1, 10)) + [11]), 1.0)
        self.assertEqual(bench.recall_with_boundary_ties(distances, list(range(1, 10)) + [12]), 0.9)
        self.assertEqual(bench.recall_with_boundary_ties(distances, [1] * 10), 0.1)
        self.assertEqual(bench.recall_with_boundary_ties(distances, list(range(2, 12))), 0.9)
        with self.assertRaises(ValueError):
            bench.recall_with_boundary_ties(distances, [0])

    def test_recall_rejects_invalid_inputs(self):
        np = bench.np
        distances = np.arange(12, dtype=np.float32)
        for bad in (0, -1, 13, 1.5, True):
            with self.subTest(k=bad), self.assertRaises(ValueError):
                bench.recall_with_boundary_ties(distances, [], bad)
        for bad in (np.nan, np.inf, -1.0):
            invalid = distances.copy()
            invalid[0] = bad
            with self.subTest(distance=bad), self.assertRaises(ValueError):
                bench.recall_with_boundary_ties(invalid, [], 10)
        for bad in (1.0, True, '1', 13):
            with self.subTest(rowid=bad), self.assertRaises(ValueError):
                bench.recall_with_boundary_ties(distances, [bad], 10)
        with self.assertRaises(ValueError):
            bench.recall_with_boundary_ties(distances.reshape(3, 4), [], 10)

    def test_invalid_sizes_fail_before_connection(self):
        with patch.object(bench.sqlite3, 'connect') as connect:
            with self.assertRaises(ValueError):
                bench.run(vectors=9)
        connect.assert_not_called()

    def test_historical_entries_stop_before_imports_or_io(self):
        names = ['bench.py','final_bench.py','hybrid_search_bench.py','index_bench.py',
                 'memory_bench3.py','multi_bench.py','probe_bm25.py',
                 'probe_chinese_tokenizer.py','probe_index.py','probe_lance.py',
                 'probe_vikingdb.py','smoke_test.py']
        for name in names:
            with self.subTest(name=name):
                result = subprocess.run([sys.executable, str(ROOT / 'scripts' / name)],
                                        capture_output=True, text=True, timeout=10)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('Historical benchmark disabled', result.stderr)

if __name__ == '__main__':
    unittest.main()
