import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from score_selection import best_scored_index


class SelectionTests(unittest.TestCase):
    def test_failed_render_does_not_win(self):
        self.assertEqual(best_scored_index([0.99, 0.2], ["render_error", "scored"]), 1)

    def test_all_failed_or_nonfinite_returns_none(self):
        self.assertIsNone(best_scored_index([0.0], ["render_error"]))
        self.assertIsNone(best_scored_index([math.nan], ["scored"]))

    def test_highest_score_and_tie(self):
        self.assertEqual(best_scored_index([0.3, 0.8, 0.8], ["scored"] * 3), 1)

    def test_mismatched_arrays_rejected(self):
        with self.assertRaises(ValueError):
            best_scored_index([0.3], [])


if __name__ == "__main__":
    unittest.main()
