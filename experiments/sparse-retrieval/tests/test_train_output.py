import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from train import reserve_output_dir


class OutputProtectionTests(unittest.TestCase):
    def test_new_and_empty_directories_are_claimed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reserve_output_dir(root / "new")
            self.assertTrue((root / "new" / ".training-run").is_file())
            empty = root / "empty"
            empty.mkdir()
            reserve_output_dir(empty)
            self.assertTrue((empty / ".training-run").is_file())
            with self.assertRaises(FileExistsError):
                reserve_output_dir(empty)

    def test_existing_data_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            weights = root / "splade_weights.pt"
            weights.write_bytes(b"original")
            with self.assertRaises(FileExistsError):
                reserve_output_dir(root)
            self.assertEqual(weights.read_bytes(), b"original")


if __name__ == "__main__":
    unittest.main()
