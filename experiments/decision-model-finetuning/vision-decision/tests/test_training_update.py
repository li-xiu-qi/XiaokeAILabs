"""Check that the actual training loop updates a parameter on CUDA.

The lightweight model and processor are test doubles; this is not a Qwen or
ScienceQA reproduction. Run in an isolated CUDA environment with torch and Pillow.
"""
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.score = torch.nn.Linear(1, 1, bias=False)
        torch.nn.init.constant_(self.score.weight, 0.5)
        text_config = types.SimpleNamespace(pad_token_id=None, eos_token_id=None)
        self.config = types.SimpleNamespace(get_text_config=lambda: text_config, use_cache=True)
        self.num_labels = 1

    def forward(self, input_ids, **kwargs):
        return types.SimpleNamespace(logits=self.score(input_ids))

    def gradient_checkpointing_enable(self, **kwargs):
        pass

    def print_trainable_parameters(self):
        pass

    def save_pretrained(self, path):
        pass


class TrainUpdateTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "requires CUDA BF16")
    def test_one_step_changes_score_weight(self):
        model = TinyModel()
        before = model.score.weight.detach().clone()
        tokenizer = types.SimpleNamespace(pad_token_id=0, eos_token_id=1)
        processor = types.SimpleNamespace(save_pretrained=lambda path: None)
        row = {"options": ["a", "b"], "gold": [1.0, 0.0]}
        transformers = types.ModuleType("transformers")
        transformers.AutoProcessor = object
        transformers.Qwen3_5ForSequenceClassification = types.SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: model
        )
        peft = types.ModuleType("peft")
        peft.LoraConfig = lambda **kwargs: None
        peft.get_peft_model = lambda model, config: model
        with patch.dict(sys.modules, {"transformers": transformers, "peft": peft}):
            spec = importlib.util.spec_from_file_location("vision_train_test", SCRIPTS / "train.py")
            train = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(train)
            with tempfile.TemporaryDirectory() as tmp:
                argv = ["train.py", "--data", tmp, "--out", str(Path(tmp) / "run"),
                        "--model", "test-double", "--steps", "1", "--bs", "1"]
                encoded = ({"input_ids": torch.tensor([[1.0], [-1.0]])}, [2])
                with patch.object(sys, "argv", argv), \
                     patch.object(train, "load_rows", return_value=[row]), \
                     patch.object(train, "get_proc", return_value=(processor, tokenizer)), \
                     patch.object(train, "encode_batch", return_value=encoded):
                    train.main()
        self.assertFalse(torch.equal(before, model.score.weight.detach().cpu()))


if __name__ == "__main__":
    unittest.main()