"""Numeric examples use independent explicit byte and FLOP calculations."""
import sys
import unittest
from unittest.mock import patch, Mock
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from hybrid_architecture_model import kv_cache_gb, estimate_memory_gb, qwen3_8_27b_analysis
from llm_spec import GB10, Qwen3_8_27B
from roofline_model import decode_tps
from embedding_throughput_model import predict_throughput
from compare_engines import test_engine as request_timing
from speculative_model import expected_output, speedup, solve_alpha_for_speedup


class CostTests(unittest.TestCase):
    def test_units_and_batch(self):
        self.assertAlmostEqual(kv_cache_gb(16, 4, 256, 65536), 4.294967296)
        self.assertAlmostEqual(kv_cache_gb(16, 4, 256, 65536, batch_size=8), 34.359738368)
        self.assertAlmostEqual(estimate_memory_gb(1, 2, 4.294967296, 0.5), 6.794967296)
        self.assertAlmostEqual(qwen3_8_27b_analysis(ctx=65536, batch_size=8)['kv_cache_gb'], 34.359738368)

    def test_kv_precision_independent_from_weights(self):
        bf16 = decode_tps(GB10, Qwen3_8_27B, 'q4_k_xl', ctx_len=65536, kv_dtype='bf16')
        fp8 = decode_tps(GB10, Qwen3_8_27B, 'q4_k_xl', ctx_len=65536, kv_dtype='fp8')
        expected_bf16 = GB10.bw_gbs * 1e9 / (27.3e9 * 0.60 + 4294967296)
        self.assertAlmostEqual(bf16, expected_bf16)
        self.assertGreater(fp8, bf16)

    def test_request_timing_uses_usage_and_full_elapsed(self):
        response = Mock()
        response.json.return_value = {"usage": {"prompt_tokens": 7, "completion_tokens": 10}}
        with patch("compare_engines.requests.post", return_value=response) as post, \
             patch("compare_engines.time.perf_counter", side_effect=[10.0, 12.0]):
            result = request_timing("http://example.test/", "model-a", "prompt", 20)
        self.assertEqual(result["prompt_tokens"], 7)
        self.assertEqual(result["completion_tokens"], 10)
        self.assertEqual(result["completion_tokens_per_request_second"], 5.0)
        self.assertEqual(post.call_args.kwargs["json"]["max_tokens"], 20)
        response.raise_for_status.assert_called_once()

    def test_speculative_conditional_expectation_and_inverse(self):
        self.assertEqual(expected_output(0.5, 2), 1.75)
        self.assertEqual(expected_output(0.0, 4), 1.0)
        self.assertEqual(expected_output(1.0, 4), 5.0)
        target = speedup(0.7, 4, 0.05)
        self.assertAlmostEqual(solve_alpha_for_speedup(target, 4, 0.05), 0.7)
        for invalid in (-0.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                expected_output(invalid, 4)
        with self.assertRaises(ValueError):
            solve_alpha_for_speedup(10, 4, 0.05)
        with self.assertRaises(ValueError):
            solve_alpha_for_speedup(1, 0, 0)

    def test_rejects_unsupported_inputs_and_unverified_fit(self):
        with self.assertRaises(ValueError):
            kv_cache_gb(16, 4, 256, 1, kv_quant='unknown')
        with self.assertRaises(NotImplementedError):
            predict_throughput(200, batch_size=1, seq_len=16)


if __name__ == '__main__':
    unittest.main()
