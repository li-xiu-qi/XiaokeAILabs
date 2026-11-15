"""Offline coverage for CLI defaults, overrides, and output protection."""
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from train_bert_rerank import BertRerankerModel, accumulated_loss_scale, parse_args, validate_paths


class PointwiseCliTest(unittest.TestCase):
    def test_defaults_match_existing_training_configuration(self):
        self.assertEqual(parse_args([]), {
            "model_name_or_path": "BAAI/bge-reranker-large",
            "train_dataset": "default_train_data.jsonl",
            "output_dir": "./output_bge_reranker_large_finetuned",
            "max_len": 512, "epochs": 30, "lr": 1e-5, "batch_size": 50,
            "seed": 42, "warmup_proportion": 0.1,
            "gradient_accumulation_steps": 8, "mixed_precision": "bf16",
        })

    def test_overrides_and_invalid_values(self):
        args = parse_args(["--model", "local-model", "--data", "data.jsonl",
                           "--output", "out", "--epochs", "1", "--batch", "2",
                           "--grad-accumulation", "3", "--precision", "no",
                           "--max-len", "128", "--lr", "0.001", "--seed", "7",
                           "--warmup-proportion", "0.2"])
        self.assertEqual((args["model_name_or_path"], args["train_dataset"], args["output_dir"]),
                         ("local-model", "data.jsonl", "out"))
        self.assertEqual((args["epochs"], args["batch_size"], args["gradient_accumulation_steps"],
                          args["mixed_precision"], args["max_len"]), (1, 2, 3, "no", 128))
        for options in (["--batch-size", "0"], ["--epochs", "-1"],
                        ["--gradient-accumulation-steps", "0"], ["--max-len", "0"],
                        ["--mixed-precision", "int8"], ["--lr", "inf"],
                        ["--lr", "nan"], ["--warmup-proportion", "nan"]):
            with self.subTest(options=options), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(options)

    def test_preprocess_uses_callable_tokenizer(self):
        class Tokenizer:
            def __call__(self, pairs, **options):
                self.pairs, self.options = pairs, options
                return {"input_ids": torch.tensor([[1, 2]])}

        tokenizer = Tokenizer()  # Deliberately has no batch_encode_plus method.
        model = BertRerankerModel(torch.nn.Identity(), tokenizer)
        result = model.preprocess([(" query ", " document ")], max_len=128)
        self.assertEqual(tokenizer.pairs, [["query", "document"]])
        self.assertEqual(tokenizer.options, {
            "add_special_tokens": True, "padding": "longest", "max_length": 128,
            "truncation": "only_second", "return_tensors": "pt",
        })
        self.assertEqual(result["input_ids"].tolist(), [[1, 2]])

    def test_accumulated_gradients_match_sample_mean(self):
        # 一组 100 + 1 个样本；最后样本与其他 100 个不能各占一半权重。
        for total, batch_size, accumulation in ((101, 100, 2), (205, 100, 2), (3, 2, 4)):
            inputs = torch.arange(1, total + 1, dtype=torch.float64)
            batch_count = (total + batch_size - 1) // batch_size
            for group_start in range(0, batch_count, accumulation):
                first = group_start * batch_size
                end = min(total, (group_start + accumulation) * batch_size)
                expected_param = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
                ((expected_param * inputs[first:end]) ** 2).mean().backward()
                actual_param = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
                for step in range(group_start, min(group_start + accumulation, batch_count)):
                    batch = inputs[step * batch_size:min(total, (step + 1) * batch_size)]
                    loss = ((actual_param * batch) ** 2).mean()
                    scale = accumulated_loss_scale(step, total, batch_size, accumulation, len(batch))
                    # Accelerator.backward 会把每个微批损失固定除以累积步数。
                    (loss * scale / accumulation).backward()
                torch.testing.assert_close(actual_param.grad, expected_param.grad)

    def test_input_and_existing_artifacts_are_protected(self):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            data = root / "input.jsonl"
            data.write_text('{}\n', encoding="utf-8")
            output = root / "output"
            args = {"train_dataset": str(data), "output_dir": str(output)}
            validate_paths(args)
            self.assertFalse(output.exists())
            output.mkdir()
            validate_paths(args)
            for name in ("final_model", "loss_curve.png"):
                artifact = output / name
                artifact.write_text("keep", encoding="utf-8")
                with self.assertRaises(FileExistsError):
                    validate_paths(args)
                self.assertEqual(artifact.read_text(encoding="utf-8"), "keep")
                artifact.unlink()
            with self.assertRaises(NotADirectoryError):
                validate_paths({"train_dataset": str(data), "output_dir": str(data)})
            with self.assertRaises(FileNotFoundError):
                validate_paths({"train_dataset": str(root / "missing"), "output_dir": str(output)})


if __name__ == "__main__":
    unittest.main()
