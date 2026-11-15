"""Regression test for single application of calibration temperature."""
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate


class TemperatureCalibrationTest(unittest.TestCase):
    def test_logits_are_scaled_once(self):
        rows = [{"gold": [1.0, 0.0]}]
        model = Mock(return_value=SimpleNamespace(logits=torch.tensor([[2.0], [0.0]])))
        with patch.object(evaluate, "encode_batch", return_value=({}, [2])):
            logits = evaluate.run_logits(None, model, rows, "cpu")
        raw, calibrated = evaluate.split_metrics(logits, rows, temperature=2.0)

        self.assertAlmostEqual(raw["soft_ce"], -math.log(torch.softmax(torch.tensor([2.0, 0.0]), -1)[0].item()), places=6)
        expected = -math.log(torch.softmax(torch.tensor([1.0, 0.0]), -1)[0].item())
        double_scaled = -math.log(torch.softmax(torch.tensor([0.5, 0.0]), -1)[0].item())
        self.assertAlmostEqual(calibrated["soft_ce"], expected, places=6)
        self.assertNotAlmostEqual(calibrated["soft_ce"], double_scaled, places=3)
        model.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
