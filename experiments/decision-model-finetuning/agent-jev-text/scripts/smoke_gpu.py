"""Exercise the training entry on a small input.

By default this uses synthetic data and a random miniature backbone. Pass
--backbone and --prepared to check local model and data files. This checks the
pipeline, not model quality.
The output directory must not exist, to protect previous runs.
"""
import argparse
import json
import runpy
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3Model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--backbone", help="Optional local Qwen3 backbone; otherwise use a tiny random one")
    parser.add_argument("--prepared", help="Optional prepared JSONL splits; otherwise use synthetic data")
    args = parser.parse_args()
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A CUDA device with bfloat16 support is required")

    if args.backbone:
        model_dir = Path(args.backbone).resolve(strict=True)
    else:
        vocab = {token: i for i, token in enumerate(
            ["[UNK]", "[PAD]", "[EOS]", "[", "]", "STATE", "QUESTION", "CANDIDATE",
             "state", "pick", "first", "second", "third"]
        )}
        tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]",
                                           pad_token="[PAD]", eos_token="[EOS]")
        model_dir = out / "random-backbone"
        tokenizer.save_pretrained(model_dir)
        config = Qwen3Config(vocab_size=len(vocab), hidden_size=32, intermediate_size=64,
                             num_hidden_layers=1, num_attention_heads=2,
                             num_key_value_heads=1, head_dim=16, max_position_embeddings=128)
        Qwen3Model(config).save_pretrained(model_dir)

    if args.prepared:
        prepared = Path(args.prepared).resolve(strict=True)
    else:
        prepared = out / "prepared"
        prepared.mkdir()
        for split in ("train", "dev", "calibration", "test"):
            row = {"id": split, "case_id": split, "workflow": "smoke", "state": "state",
                   "question": {"id": "q", "type": "choice", "text": "pick",
                                "candidates": ["first", "second", "third"]},
                   "target": [0.7, 0.2, 0.1]}
            (prepared / f"{split}_questions.jsonl").write_text(json.dumps(row) + "\n")
    protocol = {"seed": 42, "max_steps": 1, "warmup_steps": 1,
                "microbatch_questions": 1, "gradient_accumulation": 1,
                "backbone_lr": 1e-3, "head_lr": 1e-3}
    protocol_path = out / "protocol.json"
    protocol_path.write_text(json.dumps(protocol))

    original_step = torch.optim.AdamW.step
    updated = []

    def checked_step(optimizer, *args, **kwargs):
        parameters = [p for group in optimizer.param_groups for p in group["params"]
                      if p.grad is not None and torch.count_nonzero(p.grad).item()]
        if not parameters or not all(p.is_cuda for p in parameters):
            raise AssertionError("Expected nonzero gradients on CUDA parameters")
        parameter = parameters[0]
        before = parameter.detach().clone()
        result = original_step(optimizer, *args, **kwargs)
        updated.append(not torch.equal(before, parameter.detach()))
        return result

    torch.optim.AdamW.step = checked_step
    script = Path(__file__).with_name("train.py")
    sys.argv = [str(script), "--backbone", str(model_dir), "--prepared", str(prepared),
                "--protocol", str(protocol_path), "--run-dir", str(out / "run")]
    try:
        runpy.run_path(str(script), run_name="__main__")
    finally:
        torch.optim.AdamW.step = original_step
    assert updated == [True], "The optimizer did not change a CUDA parameter"
    selection = json.loads((out / "run" / "selection.json").read_text())
    assert selection["best_step"] == 0 and Path(selection["checkpoint"]).is_file()
    report = json.loads((out / "run" / "report.json").read_text())
    assert "selected" in report and "selected_calibrated" in report
    print("PASS: one CUDA optimizer step changed weights; initial checkpoint, calibration and report completed")


if __name__ == "__main__":
    main()
