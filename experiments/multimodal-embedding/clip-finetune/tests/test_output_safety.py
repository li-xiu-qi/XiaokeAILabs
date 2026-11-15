import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from output_safety import require_empty_output_dir


class OutputSafetyTests(unittest.TestCase):
    def test_new_and_empty_directory_are_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            require_empty_output_dir(str(out))
            self.assertTrue(out.is_dir())
            with self.assertRaises(FileExistsError):
                require_empty_output_dir(str(out))
            empty = Path(tmp) / "empty"
            empty.mkdir()
            require_empty_output_dir(str(empty))

    def test_existing_results_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            out.mkdir()
            result = out / "best.pt"
            result.write_bytes(b"existing")
            with self.assertRaises(FileExistsError):
                require_empty_output_dir(str(out))
            self.assertEqual(result.read_bytes(), b"existing")


if __name__ == "__main__":
    unittest.main()
