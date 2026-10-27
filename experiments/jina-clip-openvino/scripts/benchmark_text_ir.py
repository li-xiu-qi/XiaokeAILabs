#!/usr/bin/env python3
"""Validate and benchmark an existing Jina text IR on one OpenVINO device."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import openvino as ov
from transformers import AutoTokenizer

from run_openvino_smoke import TEXT_INPUT_SHAPE, TEXTS, cosine_diff, infer_compiled, native_matches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--ir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--preprocessor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 1:
        raise ValueError("warmup must be >= 0 and iterations must be >= 1")

    with np.load(args.reference) as saved:
        native_text = saved["text"]
    tokenizer = AutoTokenizer.from_pretrained(args.preprocessor, trust_remote_code=True)
    text_inputs = tokenizer(
        TEXTS,
        return_tensors="pt",
        padding="max_length",
        max_length=TEXT_INPUT_SHAPE[1],
        truncation=True,
    )

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

    output = infer_compiled(compiled, text_inputs, "input_ids")
    finite = bool(np.isfinite(output).all())
    diff = cosine_diff(native_text, output) if finite else None
    native_ok = finite and native_matches(diff)
    if not native_ok:
        result = {
            "device": args.device,
            "host": platform.node(),
            "openvino": ov.__version__,
            "npu_driver_version": npu_driver_version,
            "ir": str(args.ir),
            "input_shape": TEXT_INPUT_SHAPE,
            "finite": finite,
            "text_vs_native": diff,
            "native_ok": False,
            "ok": False,
        }
        serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
        args.output.write_text(serialized, encoding="utf-8")
        print(serialized, flush=True)
        raise RuntimeError("Benchmark refused: output does not match the native reference")

    print(f"[warmup] iterations={args.warmup}", flush=True)
    for _ in range(args.warmup):
        infer_compiled(compiled, text_inputs, "input_ids")

    latencies = []
    for index in range(args.iterations):
        started = time.perf_counter()
        infer_compiled(compiled, text_inputs, "input_ids")
        elapsed = time.perf_counter() - started
        latencies.append(elapsed)
        print(f"[infer] iteration={index + 1} seconds={elapsed:.6f}", flush=True)

    values = np.asarray(latencies, dtype=np.float64)
    mean_s = float(values.mean())
    result = {
        "device": args.device,
        "host": platform.node(),
        "openvino": ov.__version__,
        "npu_driver_version": npu_driver_version,
        "actual_compiler_type": actual_compiler_type,
        "ir": str(args.ir),
        "input_shape": TEXT_INPUT_SHAPE,
        "texts": TEXTS,
        "finite": finite,
        "text_vs_native": diff,
        "native_ok": native_ok,
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
        "throughput_sequences_per_second_from_mean": TEXT_INPUT_SHAPE[0] / mean_s,
        "raw_latency_seconds": latencies,
        "notes": (
            "Synchronous batch-3 inference after warmup; compile time excluded. "
            "Padded token counts are not used as a throughput metric."
        ),
    }
    serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    args.output.write_text(serialized, encoding="utf-8")
    print(serialized, flush=True)
    print(f"[result] {args.output}", flush=True)


if __name__ == "__main__":
    main()
