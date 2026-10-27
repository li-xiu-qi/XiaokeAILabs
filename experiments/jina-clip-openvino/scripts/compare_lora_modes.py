#!/usr/bin/env python3
"""Measure whether disabling Jina's task-LoRA path changes text embeddings."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_ID = "jinaai/jina-clip-v2"
MODEL_REVISION = "e10d47f5691d0454a0fb5d13f46f2199b74cb436"
CODE_REVISION = "39e6a55ae971b59bea6e44675d237c99762e7ee2"
TEXTS = ["a photo of a beach", "a photo of a dog", "a photo of a seal"]


def normalize(values):
    return values / (np.linalg.norm(values, axis=1, keepdims=True) + 1e-12)


def main():
    root = Path(__file__).resolve().parents[1]
    model = AutoModel.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        code_revision=CODE_REVISION,
        trust_remote_code=True,
        cache_dir=str(root / "models/huggingface"),
        local_files_only=True,
        dtype=torch.float32,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        root / "artifacts/fp16/preprocessor", trust_remote_code=True
    )
    inputs = tokenizer(
        TEXTS,
        return_tensors="pt",
        padding="max_length",
        max_length=512,
        truncation=True,
    )
    text_model = model.text_model
    original_loraid = text_model.default_loraid
    result = {
        "default_loraid": original_loraid,
        "lora_adaptation_map": text_model.lora_adaptation_map,
    }
    with torch.no_grad():
        original = text_model(inputs["input_ids"]).cpu().numpy()
        text_model._default_loraid = None
        disabled = text_model(inputs["input_ids"]).cpu().numpy()
    original_norm = normalize(original)
    disabled_norm = normalize(disabled)
    result.update(
        {
            "max_abs": float(np.max(np.abs(original_norm - disabled_norm))),
            "row_cosine": np.sum(original_norm * disabled_norm, axis=1).tolist(),
            "original_norm_mean": float(np.mean(np.linalg.norm(original, axis=1))),
            "disabled_norm_mean": float(np.mean(np.linalg.norm(disabled, axis=1))),
        }
    )
    out = root / "artifacts/lora-disable-probe.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
