#!/usr/bin/env python3
"""Benchmark native PyTorch Jina CLIP on CPU or CUDA with a shared reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import torch
import transformers
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

MODEL_ID = "jinaai/jina-clip-v2"
MODEL_REVISION = "e10d47f5691d0454a0fb5d13f46f2199b74cb436"
CODE_REVISION = "39e6a55ae971b59bea6e44675d237c99762e7ee2"
TEXTS = ["a photo of a beach", "a photo of a dog", "a photo of a seal"]
MIN_NATIVE_COSINE = 0.999
MAX_NATIVE_ABS = 0.02


def normalize(values: np.ndarray) -> np.ndarray:
    return values / (np.linalg.norm(values, axis=1, keepdims=True) + 1e-12)


def cosine_diff(reference: np.ndarray, actual: np.ndarray) -> dict:
    reference_norm = normalize(reference)
    actual_norm = normalize(actual)
    per_row = np.sum(reference_norm * actual_norm, axis=1)
    return {
        "max_abs": float(np.max(np.abs(reference_norm - actual_norm))),
        "min_row_cosine": float(np.min(per_row)),
        "mean_row_cosine": float(np.mean(per_row)),
    }


def native_matches(diff: dict) -> bool:
    return diff["max_abs"] <= MAX_NATIVE_ABS and diff["min_row_cosine"] >= MIN_NATIVE_COSINE


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", choices=["text", "vision"], required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--dtype", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--preprocessor", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("models/huggingface"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 1:
        raise ValueError("warmup must be >= 0 and iterations must be >= 1")
    if args.device == "cpu" and args.dtype == "fp16":
        raise ValueError("The native baseline uses FP32 on CPU")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    dtype = torch.float32 if args.dtype == "fp32" else torch.float16
    device = torch.device(args.device)
    with np.load(args.reference) as saved:
        native = saved[args.branch]

    started = time.perf_counter()
    model = AutoModel.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        code_revision=CODE_REVISION,
        trust_remote_code=True,
        cache_dir=str(args.cache_dir),
        local_files_only=args.offline,
        dtype=dtype,
    ).eval()
    model.to(device)
    load_s = time.perf_counter() - started

    tokenizer = AutoTokenizer.from_pretrained(
        args.preprocessor, trust_remote_code=True, local_files_only=args.offline
    )
    processor = AutoImageProcessor.from_pretrained(
        args.preprocessor, trust_remote_code=True, use_fast=False, local_files_only=args.offline
    )
    if args.branch == "text":
        inputs = tokenizer(
            TEXTS,
            return_tensors="pt",
            padding="max_length",
            max_length=512,
            truncation=True,
        )["input_ids"].to(device)
        input_shape = [3, 512]
        item_unit = "sequences"
    else:
        image = Image.open(args.image).convert("RGB")
        inputs = processor(images=[image], return_tensors="pt")["pixel_values"].to(
            device=device, dtype=dtype
        )
        input_shape = [1, 3, 512, 512]
        item_unit = "images"

    def infer() -> np.ndarray:
        with torch.inference_mode():
            if args.branch == "text":
                output = model.text_model(inputs)
            else:
                output = model.vision_model(inputs)
        return output.detach().to("cpu", dtype=torch.float32).numpy()

    output = infer()
    finite = bool(np.isfinite(output).all())
    diff = cosine_diff(native, output) if finite else None
    native_ok = finite and native_matches(diff)
    base_result = {
        "branch": args.branch,
        "device": args.device,
        "dtype": args.dtype,
        "host": platform.node(),
        "gpu_name": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "code_revision": CODE_REVISION,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "input_shape": input_shape,
        "finite": finite,
        "native_ok": native_ok,
        f"{args.branch}_vs_native": diff,
    }
    if not native_ok:
        result = {**base_result, "ok": False}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
        )
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        raise RuntimeError("Native output does not match the shared reference")

    def synchronize() -> None:
        if args.device == "cuda":
            torch.cuda.synchronize()

    print(f"[warmup] iterations={args.warmup}", flush=True)
    for _ in range(args.warmup):
        infer()
    synchronize()

    latencies = []
    for index in range(args.iterations):
        synchronize()
        started = time.perf_counter()
        infer()
        synchronize()
        elapsed = time.perf_counter() - started
        latencies.append(elapsed)
        print(f"[infer] iteration={index + 1} seconds={elapsed:.6f}", flush=True)

    values = np.asarray(latencies, dtype=np.float64)
    mean_s = float(values.mean())
    result = {
        **base_result,
        "ok": True,
        "model_load_seconds_excluded_from_benchmark": load_s,
        "warmup_iterations": args.warmup,
        "measured_iterations": args.iterations,
        "latency_seconds": {
            "min": float(values.min()),
            "p50": float(np.percentile(values, 50)),
            "mean": mean_s,
            "p90": float(np.percentile(values, 90)),
            "p95": float(np.percentile(values, 95)),
            "max": float(values.max()),
        },
        "throughput_items_per_second_from_mean": input_shape[0] / mean_s,
        "item_unit": item_unit,
        "raw_latency_seconds": latencies,
        "image_sha256": sha256_file(args.image),
        "notes": (
            "Native PyTorch inference after warmup; model loading excluded; CUDA synchronized."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    print(f"[result] {args.output}", flush=True)


if __name__ == "__main__":
    main()
