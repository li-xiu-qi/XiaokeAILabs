"""Regression tests for dataset checks that must not silently pass."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import check_dataset


class DatasetPreflightTests(unittest.TestCase):
    def test_split_with_no_detectable_face_stops(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "photo.jpg").touch()
            with patch.object(check_dataset, "identity_dir", return_value=path), \
                    patch.object(check_dataset, "analyse", return_value=None):
                with self.assertRaisesRegex(SystemExit, "no detectable faces"):
                    check_dataset.load_split(object(), "train")

    def test_missing_negative_control_stops(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(sys, "argv", ["check_dataset.py"]), \
                    patch.object(check_dataset, "_require", return_value=object()), \
                    patch.object(check_dataset, "FaceEncoder", return_value=object()), \
                    patch.object(check_dataset, "prepare_output_dir", return_value=Path(directory)), \
                    patch.object(check_dataset, "load_split", return_value=[{}]), \
                    patch.object(check_dataset, "identity_dir", return_value=Path(directory)):
                with self.assertRaisesRegex(SystemExit, "negative-control faces"):
                    check_dataset.main()


if __name__ == "__main__":
    unittest.main()
