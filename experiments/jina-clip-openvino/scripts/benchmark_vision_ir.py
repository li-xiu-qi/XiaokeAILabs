#!/usr/bin/env python3
"""Validate and benchmark an existing Jina vision IR on one OpenVINO device."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import openvino as ov
from PIL import Image
from transformers import AutoImageProcessor

from run_openvino_smoke import (
    VISION_INPUT_SHAPE,
    cosine_diff,
    infer_compiled,
    native_matches,
    sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--ir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--preprocessor", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 1:
        raise ValueError("warmup must be >= 0 and iterations must be >= 1")

    with np.load(args.reference) as saved:
        native_vision = saved["vision"]
    processor = AutoImageProcessor.from_pretrained(
        args.preprocessor, trust_remote_code=True, use_fast=False
    )
    image = Image.open(args.image).convert("RGB")
    inputs = processor(images=[image], return_tensors="pt")

    core = ov.Core()
    print(f"[compile] {args.ir} on {args.device}", flush=True)
    started = time.perf_counter()
    compiled = core.compile_model(str(args.ir), args.device)
    compile_s = time.perf_counter() - started
    try:
        actual_compiler_type = str(compiled.get_property("NPU_COMPILER_TYPE"))
    except RuntimeError:
        actual_compiler_type = None
    try:
        npu_driver_version = str(core.get_property("NPU", "NPU_DRIVER_VERSION"))
    except RuntimeError:
        npu_driver_version = None

    output = infer_compiled(compiled, inputs, "pixel_values")
    finite = bool(np.isfinite(output).all())
    diff = cosine_diff(native_vision, output) if finite else None
    native_ok = finite and native_matches(diff)
    base_result = {
        "device": args.device,
        "host": platform.node(),
        "openvino": ov.__version__,
        "npu_driver_version": npu_driver_version,
        "actual_compiler_type": actual_compiler_type,
        "ir": str(args.ir),
        "image": args.image.name,
        "image_sha256": sha256_file(args.image),
        "input_shape": VISION_INPUT_SHAPE,
        "finite": finite,
        "vision_vs_native": diff,
        "native_ok": native_ok,
    }
    if not native_ok:
        result = {**base_result, "ok": False}
        serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
        args.output.write_text(serialized, encoding="utf-8")
        print(serialized, flush=True)
        raise RuntimeError("Benchmark refused: output does not match the native reference")

    print(f"[warmup] iterations={args.warmup}", flush=True)
    for _ in range(args.warmup):
        infer_compiled(compiled, inputs, "pixel_values")

    latencies = []
    for index in range(args.iterations):
        started = time.perf_counter()
        infer_compiled(compiled, inputs, "pixel_values")
        elapsed = time.perf_counter() - started
        latencies.append(elapsed)
        print(f"[infer] iteration={index + 1} seconds={elapsed:.6f}", flush=True)

    values = np.asarray(latencies, dtype=np.float64)
    mean_s = float(values.mean())
    result = {
        **base_result,
        "ok": True,
        "compile_seconds_excluded_from_benchmark": compile_s,
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
        "throughput_images_per_second_from_mean": 1 / mean_s,
        "raw_latency_seconds": latencies,
        "notes": "Synchronous batch-1 vision inference after warmup; compile time excluded.",
    }
    serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    args.output.write_text(serialized, encoding="utf-8")
    print(serialized, flush=True)
    print(f"[result] {args.output}", flush=True)


if __name__ == "__main__":
    main()
