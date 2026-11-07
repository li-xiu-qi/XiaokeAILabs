"""Exercise caption validation without loading ML libraries or model weights."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from split_contract import clean_captions


class CaptionContractTest(unittest.TestCase):
    def test_blank_captions_are_removed(self):
        self.assertEqual(clean_captions([" first ", "  ", "second"]), ["first", "second"])

    def test_missing_or_malformed_caption_is_rejected(self):
        for captions in (None, "single string", [], ["  "], ["valid", None]):
            with self.subTest(captions=captions), self.assertRaises(ValueError):
                clean_captions(captions)


if __name__ == "__main__":
    unittest.main()
