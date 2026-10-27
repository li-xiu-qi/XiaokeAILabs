#!/usr/bin/env python3
"""Convert Jina CLIP v2 to OpenVINO IR and smoke-test one device."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
if "--offline" in sys.argv:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import openvino as ov
import torch
import transformers
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

MODEL_ID = "jinaai/jina-clip-v2"
MODEL_REVISION = "e10d47f5691d0454a0fb5d13f46f2199b74cb436"
CODE_REVISIONS = {
    "jinaai/jina-clip-implementation": "39e6a55ae971b59bea6e44675d237c99762e7ee2",
    "jinaai/xlm-roberta-flash-implementation": "bd55a5ec8e6c0fb1d6c26efb4b6a4a74ce8a88d3",
}
TEXTS = ["a photo of a beach", "a photo of a dog", "a photo of a seal"]
TEXT_INPUT_SHAPE = [len(TEXTS), 512]
VISION_INPUT_SHAPE = [1, 3, 512, 512]
MIN_NATIVE_COSINE = 0.999
MAX_NATIVE_ABS = 0.02


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="CPU", help="OpenVINO device, for example CPU or NPU")
    parser.add_argument(
        "--image", type=Path, required=True, help="Input image used for zero-shot labels"
    )
    parser.add_argument(
        "--expected-label", choices=TEXTS, help="Optional label expected to rank first"
    )
    parser.add_argument(
        "--work-dir", type=Path, default=Path("artifacts"), help="IR and result directory"
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("models/huggingface"), help="HF cache"
    )
    parser.add_argument(
        "--force-convert", action="store_true", help="Rebuild IR and native references"
    )
    parser.add_argument(
        "--single-task",
        action="store_true",
        help="Statically expand the shared retrieval.query task instead of using torch.unique",
    )
    parser.add_argument(
        "--text-fp32",
        action="store_true",
        help="Keep the text IR in FP32 while leaving the vision IR in FP16",
    )
    parser.add_argument(
        "--offline", action="store_true", help="Use only already cached model and custom code"
    )
    return parser.parse_args()


def normalize(values: np.ndarray) -> np.ndarray:
    return values / (np.linalg.norm(values, axis=1, keepdims=True) + 1e-12)


def cosine_diff(reference: np.ndarray, actual: np.ndarray) -> dict:
    ref = normalize(reference)
    got = normalize(actual)
    per_row = np.sum(ref * got, axis=1)
    return {
        "max_abs": float(np.max(np.abs(ref - got))),
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


def source_manifest(args: argparse.Namespace) -> dict:
    with Image.open(args.image) as image:
        image_size = list(image.size)
    return {
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "code_revisions": CODE_REVISIONS,
        "openvino": ov.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "texts": TEXTS,
        "single_task_unique_workaround": args.single_task,
        "text_fp32": args.text_fp32,
        "input_shapes": {"input_ids": TEXT_INPUT_SHAPE, "pixel_values": VISION_INPUT_SHAPE},
        "image_sha256": sha256_file(args.image),
        "image_size": image_size,
    }


def load_preprocessors(args: argparse.Namespace, ir_dir: Path):
    saved = ir_dir / "preprocessor"
    if args.offline and (saved / "tokenizer_config.json").exists():
        return (
            AutoTokenizer.from_pretrained(saved, trust_remote_code=True),
            AutoImageProcessor.from_pretrained(saved, trust_remote_code=True, use_fast=False),
        )
    kwargs = {
        "revision": MODEL_REVISION,
        "trust_remote_code": True,
        "cache_dir": str(args.cache_dir),
        "local_files_only": args.offline,
    }
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **kwargs)
    processor = AutoImageProcessor.from_pretrained(MODEL_ID, use_fast=False, **kwargs)
    return tokenizer, processor


def make_inputs(tokenizer, processor, image_path: Path):
    text_inputs = tokenizer(
        TEXTS,
        return_tensors="pt",
        padding="max_length",
        max_length=512,
        truncation=True,
    ).to("cpu")
    image = Image.open(image_path).convert("RGB")
    vision_inputs = processor(images=[image], return_tensors="pt")
    return text_inputs, vision_inputs


@contextmanager
def single_task_unique(enabled: bool):
    """Replace torch.unique only when every adapter task id is identical."""
    if not enabled:
        yield
        return
    original_unique = torch.unique

    def replacement(input: torch.Tensor, *args, **kwargs):
        if kwargs.get("dim") is not None or args:
            raise RuntimeError("Single-task workaround only supports flattened torch.unique")
        if input.numel() == 0:
            raise RuntimeError("Single-task workaround received an empty adapter mask")
        first = int(input.reshape(-1)[0].item())
        if not bool(torch.all(input == first).item()):
            raise RuntimeError("Single-task workaround requires one task id across the batch")
        values = torch.full((1,), first, dtype=input.dtype, device=input.device)
        outputs = [values]
        if kwargs.get("return_inverse", False):
            outputs.append(torch.zeros_like(input, dtype=torch.int64))
        if kwargs.get("return_counts", False):
            outputs.append(torch.full((1,), input.numel(), dtype=torch.int64, device=input.device))
        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    torch.unique = replacement
    try:
        yield
    finally:
        torch.unique = original_unique


def convert_and_make_reference(args: argparse.Namespace, ir_dir: Path, manifest: dict):
    print(f"[native] loading {MODEL_ID}@{MODEL_REVISION}", flush=True)
    model = AutoModel.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        code_revision=CODE_REVISIONS["jinaai/jina-clip-implementation"],
        trust_remote_code=True,
        cache_dir=str(args.cache_dir),
        local_files_only=args.offline,
        dtype=torch.float32,
    ).eval()
    tokenizer, processor = load_preprocessors(args, ir_dir)
    text_inputs, vision_inputs = make_inputs(tokenizer, processor, args.image)

    with torch.no_grad():
        reference = {
            "text": model.text_model(text_inputs["input_ids"]).cpu().numpy(),
            "vision": model.vision_model(vision_inputs["pixel_values"]).cpu().numpy(),
        }

    preprocessor_dir = ir_dir / "preprocessor"
    tokenizer.save_pretrained(preprocessor_dir)
    processor.save_pretrained(preprocessor_dir)
    np.savez(ir_dir / "native_reference.npz", **reference)
    (ir_dir / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[convert] text branch (single_task={args.single_task})", flush=True)
    with single_task_unique(args.single_task):
        text_model = ov.convert_model(model.text_model, example_input=text_inputs["input_ids"])
    set_static_input_shape(text_model, text_inputs["input_ids"])
    text_name = "jina-clip-text_v2_fp32.xml" if args.text_fp32 else "jina-clip-text_v2_fp16.xml"
    ov.save_model(text_model, ir_dir / text_name, compress_to_fp16=not args.text_fp32)
    del text_model
    gc.collect()

    print("[convert] vision branch", flush=True)
    vision_model = ov.convert_model(model.vision_model, example_input=vision_inputs["pixel_values"])
    set_static_input_shape(vision_model, vision_inputs["pixel_values"])
    ov.save_model(vision_model, ir_dir / "jina-clip-vision_v2_fp16.xml", compress_to_fp16=True)
    del vision_model, model
    gc.collect()
    return reference


def set_static_input_shape(model, example_input: torch.Tensor) -> None:
    input_name = model.input(0).get_any_name()
    model.reshape({input_name: list(example_input.shape)})


def infer_compiled(compiled, inputs: dict, key: str) -> np.ndarray:
    result = compiled(inputs[key])
    if len(result) != 1:
        raise RuntimeError(f"Expected one output, got {len(result)}")
    return list(result.values())[0]


def validate_reuse(ir_dir: Path, current_manifest: dict) -> None:
    manifest_path = ir_dir / "source_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("Artifact manifest is missing; rerun with --force-convert")
    saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    saved_manifest.setdefault("text_fp32", False)
    if saved_manifest != current_manifest:
        raise RuntimeError(
            "Artifacts belong to another model, environment, text set, or image; "
            "use a new --work-dir or rerun with --force-convert"
        )


def main() -> None:
    args = parse_args()
    args.image = args.image.resolve()
    args.work_dir = args.work_dir.resolve()
    args.cache_dir = args.cache_dir.resolve()
    ir_dir = args.work_dir / ("fp32-text" if args.text_fp32 else "fp16")
    ir_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    text_name = "jina-clip-text_v2_fp32.xml" if args.text_fp32 else "jina-clip-text_v2_fp16.xml"
    text_path = ir_dir / text_name
    vision_path = ir_dir / "jina-clip-vision_v2_fp16.xml"
    reference_path = ir_dir / "native_reference.npz"
    manifest = source_manifest(args)
    must_convert = args.force_convert or not all(
        path.exists() for path in (text_path, vision_path, reference_path)
    )

    core = ov.Core()
    print(f"[env] OpenVINO={ov.__version__} torch={torch.__version__} "
          f"transformers={transformers.__version__}", flush=True)
    print(f"[env] devices={core.available_devices}; requested={args.device}", flush=True)
    if args.device not in core.available_devices:
        raise RuntimeError(f"Device {args.device} is unavailable: {core.available_devices}")
    try:
        npu_driver_version = str(core.get_property("NPU", "NPU_DRIVER_VERSION"))
    except RuntimeError:
        npu_driver_version = None

    if must_convert:
        reference = convert_and_make_reference(args, ir_dir, manifest)
    else:
        validate_reuse(ir_dir, manifest)
        with np.load(reference_path) as saved:
            reference = {"text": saved["text"], "vision": saved["vision"]}

    tokenizer = AutoTokenizer.from_pretrained(ir_dir / "preprocessor", trust_remote_code=True)
    processor = AutoImageProcessor.from_pretrained(
        ir_dir / "preprocessor", trust_remote_code=True, use_fast=False
    )
    text_inputs, vision_inputs = make_inputs(tokenizer, processor, args.image)

    started = time.perf_counter()
    print(f"[compile] text branch on {args.device}", flush=True)
    compiled_text = core.compile_model(str(text_path), args.device)
    text_compile_s = time.perf_counter() - started
    started = time.perf_counter()
    print(f"[compile] vision branch on {args.device}", flush=True)
    compiled_vision = core.compile_model(str(vision_path), args.device)
    vision_compile_s = time.perf_counter() - started

    started = time.perf_counter()
    text_output = infer_compiled(compiled_text, text_inputs, "input_ids")
    text_infer_s = time.perf_counter() - started
    started = time.perf_counter()
    vision_output = infer_compiled(compiled_vision, vision_inputs, "pixel_values")
    vision_infer_s = time.perf_counter() - started

    text_features = normalize(text_output)
    vision_features = normalize(vision_output)
    similarities = (vision_features @ text_features.T)[0]
    best_index = int(np.argmax(similarities))
    text_diff = cosine_diff(reference["text"], text_output)
    vision_diff = cosine_diff(reference["vision"], vision_output)
    native_ok = native_matches(text_diff) and native_matches(vision_diff)
    ranking_ok = None
    if args.expected_label is not None:
        ranking_ok = TEXTS[best_index] == args.expected_label
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "code_revisions": CODE_REVISIONS,
        "device": args.device,
        "available_devices": core.available_devices,
        "host": platform.node(),
        "python": platform.python_version(),
        "openvino": ov.__version__,
        "npu_driver_version": npu_driver_version,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "image": str(args.image),
        "image_sha256": manifest["image_sha256"],
        "texts": TEXTS,
        "input_shapes": manifest["input_shapes"],
        "text_precision": "FP32" if args.text_fp32 else "FP16",
        "similarities": {text: float(similarities[i]) for i, text in enumerate(TEXTS)},
        "best_label": TEXTS[best_index],
        "expected_label": args.expected_label,
        "ranking_ok": ranking_ok,
        "text_shape": list(text_output.shape),
        "vision_shape": list(vision_output.shape),
        "text_vs_native": text_diff,
        "vision_vs_native": vision_diff,
        "native_ok": native_ok,
        "ok": native_ok and ranking_ok is not False,
        "compile_seconds": {"text": text_compile_s, "vision": vision_compile_s},
        "inference_seconds": {"text_batch3": text_infer_s, "vision_batch1": vision_infer_s},
        "ir_files": [str(text_path), str(vision_path)],
    }
    precision_suffix = "-fp32" if args.text_fp32 else ""
    result_path = args.work_dir / f"smoke-{args.device.upper()}{precision_suffix}.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    print(f"[result] {result_path}", flush=True)
    if not result["ok"]:
        raise RuntimeError("Smoke validation failed; inspect the saved result JSON")
    print("[ok] device smoke passed", flush=True)


if __name__ == "__main__":
    main()
