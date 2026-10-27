#!/usr/bin/env python3
"""Evaluate bidirectional image-text retrieval with Jina CLIP task adapters."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from benchmark_native_baseline import (
    CODE_REVISION,
    MODEL_ID,
    MODEL_REVISION,
    normalize,
)


def load_pairs(path: Path, limit: int | None) -> list[dict]:
    pairs = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("Subset must be a non-empty JSON list")
    if limit is not None:
        pairs = pairs[:limit]
    for index, pair in enumerate(pairs):
        if not pair.get("caption") or not pair.get("image"):
            raise ValueError(f"Invalid pair at index {index}")
    return pairs


def infer_openvino(model, inputs: dict, key: str) -> np.ndarray:
    result = model(inputs[key])
    if len(result) != 1:
        raise RuntimeError(f"Expected one OpenVINO output, got {len(result)}")
    return list(result.values())[0]


def ranks_for(similarity: np.ndarray) -> np.ndarray:
    order = np.argsort(-similarity, axis=1, kind="stable")
    targets = np.arange(similarity.shape[0])[:, None]
    return np.argmax(order == targets, axis=1) + 1


def metrics_for(ranks: np.ndarray) -> dict:
    return {
        "count": int(len(ranks)),
        "recall_at_1": float(np.mean(ranks <= 1)),
        "recall_at_5": float(np.mean(ranks <= 5)),
        "recall_at_10": float(np.mean(ranks <= 10)),
        "mrr": float(np.mean(1 / ranks)),
        "mrr_at_10": float(np.mean(np.where(ranks <= 10, 1 / ranks, 0.0))),
        "ndcg_at_10": float(np.mean(np.where(ranks <= 10, 1 / np.log2(ranks + 1), 0.0))),
        "median_rank": float(np.median(ranks)),
        "mean_rank": float(np.mean(ranks)),
        "max_rank": int(ranks.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["openvino", "native"], required=True)
    parser.add_argument("--device", required=True, help="CPU, NPU, cpu, or cuda")
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--preprocessor", type=Path, required=True)
    parser.add_argument("--query-ir", type=Path)
    parser.add_argument("--document-ir", type=Path)
    parser.add_argument("--vision-ir", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=Path("models/huggingface"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    pairs = load_pairs(args.subset, args.limit)
    captions = [pair["caption"] for pair in pairs]
    image_root = args.image_root or args.subset.parent
    image_paths = [str(image_root / pair["image"]) for pair in pairs]

    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.preprocessor, trust_remote_code=True, local_files_only=args.offline
    )
    processor = AutoImageProcessor.from_pretrained(
        args.preprocessor,
        trust_remote_code=True,
        use_fast=False,
        local_files_only=args.offline,
    )

    compiled = {}
    native_model = None
    device = None
    compile_seconds = {}
    if args.backend == "openvino":
        import openvino as ov

        for name, path in [
            ("query", args.query_ir),
            ("document", args.document_ir),
            ("vision", args.vision_ir),
        ]:
            if path is None:
                raise ValueError(f"--{name.replace('_', '-')}-ir is required for OpenVINO")
        core = ov.Core()
        for name, path in [
            ("query", args.query_ir),
            ("document", args.document_ir),
            ("vision", args.vision_ir),
        ]:
            print(f"[compile] {name} on {args.device}", flush=True)
            started = time.perf_counter()
            compiled[name] = core.compile_model(str(path), args.device)
            compile_seconds[name] = time.perf_counter() - started
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        print(f"[load] native model on {args.device}", flush=True)
        started = time.perf_counter()
        from transformers import AutoModel

        native_model = AutoModel.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            code_revision=CODE_REVISION,
            trust_remote_code=True,
            cache_dir=str(args.cache_dir),
            local_files_only=args.offline,
            dtype=torch.float32,
        ).eval()
        native_model.to(device)
        compile_seconds["model_load"] = time.perf_counter() - started

    def encode_texts(texts: list[str], role: str) -> np.ndarray:
        outputs = []
        if args.backend == "openvino":
            model = compiled[role]
            for start in range(0, len(texts), 3):
                group = texts[start : start + 3]
                actual = len(group)
                if actual < 3:
                    group = group + [""] * (3 - actual)
                encoded = tokenizer(
                    group,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=512,
                    truncation=True,
                )
                outputs.append(infer_openvino(model, encoded, "input_ids")[:actual])
        else:
            original_loraid = native_model.text_model.default_loraid
            if role == "query" and original_loraid is None:
                raise RuntimeError("Native query adapter is not retrieval.query")
            native_model.text_model._default_loraid = original_loraid if role == "query" else None
            try:
                for start in range(0, len(texts), args.batch_size):
                    group = texts[start : start + args.batch_size]
                    encoded = tokenizer(
                        group,
                        return_tensors="pt",
                        padding="max_length",
                        max_length=512,
                        truncation=True,
                    )["input_ids"].to(device)
                    with torch.inference_mode():
                        output = native_model.text_model(encoded)
                        outputs.append(output.detach().to("cpu", dtype=torch.float32).numpy())
            finally:
                native_model.text_model._default_loraid = original_loraid
        return np.concatenate(outputs, axis=0).astype(np.float32)

    def encode_images(paths: list[str]) -> np.ndarray:
        outputs = []
        if args.backend == "openvino":
            for path in paths:
                image = Image.open(path).convert("RGB")
                inputs = processor(images=[image], return_tensors="pt")
                outputs.append(infer_openvino(compiled["vision"], inputs, "pixel_values"))
        else:
            for start in range(0, len(paths), args.batch_size):
                images = [
                    Image.open(path).convert("RGB")
                    for path in paths[start : start + args.batch_size]
                ]
                inputs = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
                with torch.inference_mode():
                    output = native_model.vision_model(inputs)
                    outputs.append(output.detach().to("cpu", dtype=torch.float32).numpy())
                for image in images:
                    image.close()
        return np.concatenate(outputs, axis=0).astype(np.float32)

    print(f"[encode] query_text count={len(captions)}", flush=True)
    started = time.perf_counter()
    query_text = encode_texts(captions, "query")
    text_seconds = time.perf_counter() - started
    print(f"[encode] document_text count={len(captions)}", flush=True)
    started = time.perf_counter()
    document_text = encode_texts(captions, "document")
    document_seconds = time.perf_counter() - started
    print(f"[encode] images count={len(image_paths)}", flush=True)
    started = time.perf_counter()
    images = encode_images(image_paths)
    image_seconds = time.perf_counter() - started

    if not all(np.isfinite(value).all() for value in [query_text, document_text, images]):
        raise RuntimeError("Non-finite embeddings produced")
    query_text = normalize(query_text)
    document_text = normalize(document_text)
    images = normalize(images)
    text_to_image_ranks = ranks_for(query_text @ images.T)
    image_to_text_ranks = ranks_for(images @ document_text.T)
    result = {
        "backend": args.backend,
        "device": args.device,
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "code_revision": CODE_REVISION,
        "subset": args.subset.name,
        "pair_count": len(pairs),
        "query_task": "retrieval.query",
        "document_task": None,
        "text_to_image": metrics_for(text_to_image_ranks),
        "image_to_text": metrics_for(image_to_text_ranks),
        "text_to_image_ranks": text_to_image_ranks.tolist(),
        "image_to_text_ranks": image_to_text_ranks.tolist(),
        "encoding_seconds": {
            "query_text_total": text_seconds,
            "document_text_total": document_seconds,
            "images_total": image_seconds,
            "query_text_per_item": text_seconds / len(pairs),
            "document_text_per_item": document_seconds / len(pairs),
            "images_per_item": image_seconds / len(pairs),
        },
        "compile_or_load_seconds_excluded": compile_seconds,
        "native_batch_size": args.batch_size if args.backend == "native" else None,
        "openvino_text_batch_size": 3 if args.backend == "openvino" else None,
        "ok": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {key: value for key, value in result.items() if not key.endswith("_ranks")}
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"[result] {args.output}", flush=True)


if __name__ == "__main__":
    main()
