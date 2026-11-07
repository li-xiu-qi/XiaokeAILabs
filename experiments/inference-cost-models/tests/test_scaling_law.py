"""Check the paper's joint-fit limits and training-compute arithmetic."""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from scaling_law_model import (
    ALPHA_D, ALPHA_N, D_C, N_C, chinchilla_optimal_flops,
    flops_to_params_for_budget, kaplan_loss,
)


class ScalingLawTests(unittest.TestCase):
    def test_paper_joint_fit_and_large_data_limit(self):
        # When each dimension equals its own scale, Eq. 1.5 reduces to 2**alpha_D.
        self.assertAlmostEqual(kaplan_loss(N_C / 1e9, D_C / 1e12), 2 ** ALPHA_D)
        # With arbitrarily much data, the data-limited term vanishes.
        self.assertAlmostEqual(kaplan_loss(7, 1e100), (N_C / 7e9) ** ALPHA_N)

    def test_training_compute_round_trip(self):
        self.assertAlmostEqual(flops_to_params_for_budget(chinchilla_optimal_flops(70)), 70)

    def test_nonpositive_and_nonfinite_inputs(self):
        for value in (0, -1, math.inf, math.nan):
            with self.subTest(value=value), self.assertRaises(ValueError):
                kaplan_loss(value, 1)


if __name__ == "__main__":
    unittest.main()
