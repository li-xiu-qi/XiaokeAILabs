#!/usr/bin/env python3
"""Compile an existing text IR on one device and compare it with native output."""

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="NPU")
    parser.add_argument("--ir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--preprocessor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compiler-type", help="Optional NPU compiler type, for example DRIVER")
    parser.add_argument(
        "--activations-scale-factor", type=float, help="Scale activations down before inference"
    )
    args = parser.parse_args()

    with np.load(args.reference) as saved:
        native_text = saved["text"]
    tokenizer = AutoTokenizer.from_pretrained(args.preprocessor, trust_remote_code=True)
    text_inputs = tokenizer(
        TEXTS,
        return_tensors="pt",
        padding="max_length",
        max_length=512,
        truncation=True,
    )
    core = ov.Core()
    config = {}
    if args.compiler_type:
        config["NPU_COMPILER_TYPE"] = args.compiler_type
    if args.activations_scale_factor is not None:
        config["ACTIVATIONS_SCALE_FACTOR"] = args.activations_scale_factor
    started = time.perf_counter()
    print(f"[compile] {args.ir} on {args.device}; config={config}", flush=True)
    compiled = core.compile_model(str(args.ir), args.device, config)
    compile_s = time.perf_counter() - started
    try:
        actual_compiler_type = str(compiled.get_property("NPU_COMPILER_TYPE"))
    except RuntimeError:
        actual_compiler_type = None
    try:
        npu_driver_version = str(core.get_property("NPU", "NPU_DRIVER_VERSION"))
    except RuntimeError:
        npu_driver_version = None

    started = time.perf_counter()
    output = infer_compiled(compiled, text_inputs, "input_ids")
    infer_s = time.perf_counter() - started
    finite = bool(np.isfinite(output).all())
    diff = cosine_diff(native_text, output) if finite else None
    native_ok = finite and native_matches(diff)
    ok = finite and native_ok
    result = {
        "device": args.device,
        "host": platform.node(),
        "openvino": ov.__version__,
        "npu_driver_version": npu_driver_version,
        "ir": str(args.ir),
        "input_shape": TEXT_INPUT_SHAPE,
        "compiler_type": args.compiler_type,
        "actual_compiler_type": actual_compiler_type,
        "activations_scale_factor": args.activations_scale_factor,
        "finite": finite,
        "nan_count": int(np.isnan(output).sum()),
        "inf_count": int(np.isinf(output).sum()),
        "text_vs_native": diff,
        "native_ok": native_ok,
        "ok": ok,
        "compile_seconds": compile_s,
        "inference_seconds": infer_s,
    }
    serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    args.output.write_text(serialized, encoding="utf-8")
    print(serialized, flush=True)
    print(f"[result] {args.output}", flush=True)
    if not ok:
        raise RuntimeError("Existing IR validation failed")


if __name__ == "__main__":
    main()
