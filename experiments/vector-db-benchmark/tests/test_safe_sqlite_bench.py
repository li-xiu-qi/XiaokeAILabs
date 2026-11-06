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
