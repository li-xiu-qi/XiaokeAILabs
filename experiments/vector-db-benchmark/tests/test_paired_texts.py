"""Offline checks for vector/text pairing; no model or dataset download."""

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))


class FakeEncoder:
    def __init__(self, _model):
        pass

    def encode(self, texts, **_kwargs):
        return np.array([[len(text), i] for i, text in enumerate(texts)], dtype=np.float32)


with patch.dict(sys.modules, {"sentence_transformers": types.SimpleNamespace(SentenceTransformer=FakeEncoder)}):
    import gen_emb

from paired_texts import load_aligned_texts, text_digest


class PairedTextsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.data = self.base / "corpus"
        for split in ("train", "test"):
            for category, filename in (("b", "2"), ("a", "1")):
                folder = self.data / f"20news-bydate-{split}" / category
                folder.mkdir(parents=True)
                (folder / filename).write_text(
                    f"Header: x\n\n{split} {category} " + "sample text " * 5, encoding="utf-8"
                )
        self.emb = self.base / "emb.npz"
        self.texts = self.base / "texts.npz"

    def generate(self):
        args = ["gen_emb.py", "--data-dir", str(self.data), "--model", "unused",
                "--output", str(self.emb), "--texts-output", str(self.texts)]
        with patch.object(sys, "argv", args):
            gen_emb.main()

    def test_generated_pair_has_same_contents_and_rejects_changed_text(self):
        self.generate()
        train, test = load_aligned_texts(self.emb, self.texts)
        self.assertEqual(len(train), 2)
        self.assertEqual(len(test), 2)
        self.assertTrue(train[0].startswith("train a"))
        self.assertTrue(train[1].startswith("train b"))
        with np.load(self.texts, allow_pickle=False) as f:
            values = {key: f[key] for key in f.files}
        values["train_texts"] = values["train_texts"][::-1]
        changed = self.base / "changed.npz"
        np.savez_compressed(changed, **values)
        with self.assertRaisesRegex(ValueError, "content or ordering"):
            load_aligned_texts(self.emb, changed)

    def test_historical_archives_without_hashes_are_rejected(self):
        np.savez_compressed(self.emb, train_emb=np.zeros((1, 2)), test_emb=np.zeros((1, 2)))
        np.savez_compressed(self.texts, train_texts=["a"], test_texts=["b"])
        with self.assertRaisesRegex(ValueError, "no paired text hashes"):
            load_aligned_texts(self.emb, self.texts)

    def test_row_count_mismatch_is_rejected(self):
        self.generate()
        with np.load(self.emb, allow_pickle=False) as f:
            values = {key: f[key] for key in f.files}
        values["train_emb"] = values["train_emb"][:1]
        other = self.base / "other.npz"
        np.savez_compressed(other, **values)
        with self.assertRaisesRegex(ValueError, "row counts differ"):
            load_aligned_texts(other, self.texts)

    def test_existing_output_is_never_overwritten(self):
        self.texts.write_bytes(b"preserve")
        with self.assertRaises(FileExistsError):
            self.generate()
        self.assertEqual(self.texts.read_bytes(), b"preserve")
        self.assertFalse(self.emb.exists())

    def test_hash_frames_each_text(self):
        self.assertNotEqual(text_digest(["ab", "c"]), text_digest(["a", "bc"]))


if __name__ == "__main__":
    unittest.main()
