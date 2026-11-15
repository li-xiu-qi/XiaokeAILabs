#!/usr/bin/env python3
"""Load a small causal language model and generate one CUDA response."""
import json
import os
import sys
import time
from importlib.metadata import version
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = os.environ.get("QWEN_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
model_revision = os.environ.get("QWEN_MODEL_REVISION")
model_dir = Path(snapshot_download(repo_id=model_id, revision=model_revision))
load_started = time.perf_counter()
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_dir, dtype=torch.bfloat16, device_map="cuda"
)
model.eval()
load_seconds = time.perf_counter() - load_started
messages = [{"role": "user", "content": "Reply with one word: billing"}]
encoded = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt", tokenize=True
)
if hasattr(encoded, "keys") and "input_ids" in encoded.keys():
    encoded = encoded["input_ids"]
encoded = encoded.to("cuda")
with torch.inference_mode():
    generated = model.generate(encoded, max_new_tokens=12, do_sample=False)
new_tokens = int(generated.shape[1] - encoded.shape[1])
generated_text = tokenizer.decode(
    generated[0, encoded.shape[1]:], skip_special_tokens=True
)
device = next(model.parameters()).device
if device.type != "cuda" or new_tokens <= 0 or not generated_text.strip():
    raise SystemExit("GPU smoke failed: expected CUDA execution and non-empty generated text")

summary = {
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "transformers": version("transformers"),
    "accelerate": version("accelerate"),
    "gpu": torch.cuda.get_device_name(0),
    "cuda_available": torch.cuda.is_available(),
    "bf16_supported": torch.cuda.is_bf16_supported(),
    "model_id": model_id,
    "model_revision": model_dir.name,
    "model_path": str(model_dir),
    "load_seconds": load_seconds,
    "device": str(device),
    "prompt_tokens": int(encoded.shape[1]),
    "generated_tokens": new_tokens,
    "encoding": "chat_template",
    "output": generated_text,
    "cuda_allocated_gib": torch.cuda.memory_allocated() / 1024**3,
    "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
    "result": "passed",
}
out_dir = Path(os.environ.get("QWEN_OUTPUT_DIR", os.environ.get("XL_OUT_DIR", ".work")))
out_dir.mkdir(parents=True, exist_ok=True)
out = out_dir / "summary.json"
out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
print(f"Evidence written to {out}")
