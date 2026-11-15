"""Offline checks for preserving data and earlier training results."""
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_embedding.py"
spec = importlib.util.spec_from_file_location("embedding_train_safety", SCRIPT)
train = importlib.util.module_from_spec(spec)
# The guard must be testable without downloading a model or installing its wrapper.
fake_sentence_transformers = types.ModuleType("sentence_transformers")
fake_sentence_transformers.SentenceTransformer = mock.Mock()
with mock.patch.dict(sys.modules, {"sentence_transformers": fake_sentence_transformers}):
    spec.loader.exec_module(train)


class OutputSafetyTests(unittest.TestCase):
    def test_new_and_empty_output_do_not_touch_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "train.jsonl"
            data.write_text('{"query":"q","pos_doc":"d"}\n', encoding="utf-8")
            content = data.read_bytes()
            new_dir = root / "new"
            empty_dir = root / "empty"
            empty_dir.mkdir()
            train.validate_output_dir(new_dir, data)
            train.validate_output_dir(empty_dir, data)
            self.assertFalse(new_dir.exists())
            self.assertEqual(list(empty_dir.iterdir()), [])
            self.assertEqual(data.read_bytes(), content)

    def test_file_and_nonempty_directory_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "train.jsonl"
            data.write_text("training data", encoding="utf-8")
            existing = root / "run"
            existing.mkdir()
            sentinel = existing / "checkpoint"
            sentinel.write_text("keep me", encoding="utf-8")
            for output in (data, existing):
                with self.subTest(output=output), self.assertRaises(ValueError):
                    train.validate_output_dir(output, data)
            self.assertEqual(data.read_text(encoding="utf-8"), "training data")
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep me")

    def test_input_inside_empty_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            with self.assertRaisesRegex(ValueError, "训练数据"):
                train.validate_output_dir(output, output / "demo_data.jsonl")
            self.assertEqual(list(output.iterdir()), [])

    def test_main_rejects_before_demo_creation_and_model_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "run"
            output.mkdir()
            sentinel = output / "previous.txt"
            sentinel.write_text("unchanged", encoding="utf-8")
            existing_file = root / "existing.txt"
            existing_file.write_text("keep file", encoding="utf-8")
            original_cwd = Path.cwd()
            try:
                os.chdir(root)
                for destination in (output, existing_file):
                    with self.subTest(destination=destination), \
                         mock.patch.object(sys, "argv", ["train_embedding.py", "--output_dir", str(destination)]), \
                         mock.patch.object(train.AutoTokenizer, "from_pretrained") as tokenizer, \
                         mock.patch.object(train, "EmbeddingModel") as model, \
                         contextlib.redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit) as result:
                            train.main()
                    self.assertEqual(result.exception.code, 2)
                    tokenizer.assert_not_called()
                    model.assert_not_called()
                    self.assertFalse((root / "demo_data.jsonl").exists())
                    self.assertEqual(sentinel.read_text(encoding="utf-8"), "unchanged")
                    self.assertEqual(existing_file.read_text(encoding="utf-8"), "keep file")
            finally:
                os.chdir(original_cwd)


if __name__ == "__main__":
    unittest.main()
