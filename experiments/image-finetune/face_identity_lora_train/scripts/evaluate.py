#!/usr/bin/env python3
"""Measure how much of an identity the model reproduces.

Three sub-commands share one generation and scoring path, so the numbers stay
comparable: identical prompt (only the trigger token differs), identical seeds
(1000 + i), identical sampler settings.

    baseline    no LoRA, with three references: the target's own cross-split
                similarity (upper bound), the negative-control identity, and
                the base model's output on the same prompt
    lora        one LoRA checkpoint, scored against both references
    scan        every checkpoint in a run directory, to locate the best step
                without paying for extra training

    python scripts/evaluate.py baseline
    python scripts/evaluate.py lora --lora output/.../face_identity_sdxl_lora.safetensors
    python scripts/evaluate.py scan --run-dir output_face/face_identity_sdxl_lora

Results are written as JSON so the README figures can be traced to raw numbers.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from common import (
    NEGATIVE_ID,
    OUT_DIR,
    PROMPT,
    SDXL_BASE,
    TARGET_ID,
    FaceEncoder,
    _require,
    collect_embeddings,
    identity_dir,
    prepare_output_dir,
    prototype,
    to_rgb_array,
)

BASE_PROMPT = "a photo of a person, front facing, sharp focus"
STEPS, GUIDANCE, SAMPLER = 30, 7.0, "dpmsolver++"


def load_pipeline(lora: str | Path | None = None):
    torch = _require("torch")
    try:
        from diffusers import StableDiffusionXLPipeline
    except ImportError as exc:
        raise SystemExit(
            "missing dependency: diffusers\nthe evaluation stack needs "
            "`pip install torch diffusers transformers accelerate`"
        ) from exc

    pipe = StableDiffusionXLPipeline.from_pretrained(
        SDXL_BASE, torch_dtype=torch.bfloat16, use_safetensors=True, variant=None
    ).to("cuda")
    if lora is not None:
        pipe.load_lora_weights(str(lora))
    pipe.set_progress_bar_config(disable=True)
    return pipe


def generate(pipe, prompt: str, n: int, size: int, out_dir: Path, prefix: str) -> list[dict]:
    """Generate n images at fixed seeds; return per-image embeddings."""
    torch = _require("torch")
    encoder = FaceEncoder()
    records = []
    for i in range(n):
        generator = torch.Generator("cuda").manual_seed(1000 + i)
        image = pipe(
            prompt,
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE,
            height=size,
            width=size,
            generator=generator,
        ).images[0]
        path = out_dir / f"{prefix}{i:02d}.jpg"
        image.save(path, quality=92)
        emb = encoder.embed(to_rgb_array(image))
        if emb is not None:
            records.append({"file": path.name, "emb": emb})
    print(f"    {len(records)}/{n} images with a detectable face")
    return records


def summary(values: list[float]) -> dict | None:
    np = _require("numpy")
    if not values:
        return None
    arr = np.array(values)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n": len(values),
    }


def references(encoder: FaceEncoder) -> dict:
    """Target prototype, target cross-split baseline, negative-control prototype."""
    train = collect_embeddings(encoder, identity_dir(TARGET_ID, "train"))
    test = collect_embeddings(encoder, identity_dir(TARGET_ID, "test"))
    target_proto = prototype(test)
    negative = collect_embeddings(encoder, identity_dir(NEGATIVE_ID, "test"))
    return {
        "target_proto": target_proto,
        "negative_proto": prototype(negative),
        "cross_split": summary([float(e @ target_proto) for e in train]),
        "n_train": len(train),
        "n_test": len(test),
        "n_negative": len(negative),
    }


def score(records: list[dict], refs: dict) -> dict:
    against_target = [float(r["emb"] @ refs["target_proto"]) for r in records]
    against_negative = [float(r["emb"] @ refs["negative_proto"]) for r in records]
    target_stats = summary(against_target)
    negative_stats = summary(against_negative)
    return {
        "vs_target": target_stats,
        "vs_other": negative_stats,
        "net_gain": (
            target_stats["mean"] - negative_stats["mean"]
            if target_stats and negative_stats
            else None
        ),
        "per_image_vs_target": against_target,
    }


def cmd_baseline(args) -> None:
    encoder = FaceEncoder()
    refs = references(encoder)
    print(f"target {TARGET_ID}: {refs['n_train']} train / {refs['n_test']} test images")
    print(f"cross-split similarity (the ceiling): {refs['cross_split']['mean']:.4f}")

    out_dir = prepare_output_dir("baseline")
    pipe = load_pipeline()
    records = generate(pipe, BASE_PROMPT, args.n, args.size, out_dir, "base_s")
    scored = score(records, refs)

    result = {
        "prompt": BASE_PROMPT,
        "n_requested": args.n,
        "n_scored": len(records),
        "reference": {
            "celeb_id": TARGET_ID,
            "n_train": refs["n_train"],
            "n_test": refs["n_test"],
            "cross_split": refs["cross_split"],
        },
        "baseline_vs_target": scored["vs_target"],
        "baseline_vs_other": scored["vs_other"],
        "per_image": scored["per_image_vs_target"],
    }
    path = out_dir / "baseline.json"
    path.write_text(json.dumps(result, indent=1), encoding="utf-8")

    print("\n=== base model, no LoRA ===")
    print(f"  vs target identity : {scored['vs_target']['mean']:+.4f}")
    print(f"  vs other identity  : {scored['vs_other']['mean']:+.4f}")
    print(f"  same-person ceiling: {refs['cross_split']['mean']:.4f}")
    print(
        "\nInterpretation: if the base model scores near zero against the target\n"
        "and no higher than against a stranger, it cannot draw this person — the\n"
        "target is real and LoRA has something to prove."
    )
    print(f"result -> {path}")


def cmd_lora(args) -> None:
    encoder = FaceEncoder()
    refs = references(encoder)
    out_dir = prepare_output_dir("lora")
    pipe = load_pipeline(args.lora)
    records = generate(pipe, PROMPT, args.n, args.size, out_dir, "lora_s")
    scored = score(records, refs)

    baseline_path = OUT_DIR / "baseline" / "baseline.json"
    baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else None

    result = {
        "lora": str(args.lora),
        "prompt": PROMPT,
        "n_requested": args.n,
        "n_scored": len(records),
        "vs_target": scored["vs_target"],
        "vs_other": scored["vs_other"],
        "net_gain": scored["net_gain"],
        "per_image": scored["per_image_vs_target"],
    }
    path = out_dir / "eval.json"
    path.write_text(json.dumps(result, indent=1), encoding="utf-8")

    print("\n=== with LoRA ===")
    print(f"  vs target identity : {scored['vs_target']['mean']:.4f}  (n={len(records)})")
    print(f"  vs other identity  : {scored['vs_other']['mean']:.4f}")
    print(f"  net gain           : {scored['net_gain']:+.4f}")
    print(f"  same-person ceiling: {refs['cross_split']['mean']:.4f}")
    if baseline:
        print(f"  base model         : {baseline['baseline_vs_target']['mean']:+.4f}")
    else:
        print("  (run `evaluate.py baseline` first to compare against the base model)")
    print(f"result -> {path}")


def cmd_scan(args) -> None:
    run_dir = Path(args.run_dir)
    checkpoints = sorted(glob.glob(str(run_dir / "*.safetensors")))
    if not checkpoints:
        raise SystemExit(
            f"no .safetensors under {run_dir}\n"
            f"check that save_every is set in the training config"
        )
    encoder = FaceEncoder()
    refs = references(encoder)
    out_root = prepare_output_dir("scan")

    rows = []
    for ckpt in checkpoints:
        step = Path(ckpt).stem.split("_")[-1]
        print(f"\n--- step {step} ---")
        out_dir = out_root / f"step_{step}"
        out_dir.mkdir(parents=True, exist_ok=True)
        pipe = load_pipeline(ckpt)
        records = generate(pipe, PROMPT, args.n, args.size, out_dir, "s")
        del pipe
        _require("torch").cuda.empty_cache()
        scored = score(records, refs)
        rows.append(
            {
                "step": step,
                "checkpoint": Path(ckpt).name,
                "n": len(records),
                "vs_target": scored["vs_target"],
                "vs_other": scored["vs_other"],
                "net_gain": scored["net_gain"],
            }
        )
        (out_root / "scan.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
        print(
            f"    vs target {scored['vs_target']['mean'] if scored['vs_target'] else float('nan'):.4f}"
            f"  vs other {scored['vs_other']['mean'] if scored['vs_other'] else float('nan'):.4f}"
        )

    print("\n=== checkpoint scan ===")
    print(f"{'step':<10}{'vs target':>12}{'vs other':>12}{'net gain':>12}")
    for row in sorted(rows, key=lambda r: r["step"]):
        print(
            f"{row['step']:<10}{row['vs_target']['mean']:>12.4f}"
            f"{row['vs_other']['mean']:>12.4f}{row['net_gain']:>+12.4f}"
        )
    best = max(rows, key=lambda r: r["vs_target"]["mean"])
    print(
        f"\nbest step: {best['step']} (vs target {best['vs_target']['mean']:.4f}).\n"
        "A peak followed by a decline is overfitting; stop at the peak."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--n", type=int, default=20, help="images per condition (default: 20)")
    parser.add_argument("--size", type=int, default=768, help="square output size (default: 768)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("baseline", help="score the base model with no LoRA").set_defaults(
        func=cmd_baseline
    )

    lora = sub.add_parser("lora", help="score one LoRA checkpoint")
    lora.add_argument("--lora", required=True, type=Path, help="path to .safetensors")
    lora.set_defaults(func=cmd_lora)

    scan = sub.add_parser("scan", help="score every checkpoint in a run directory")
    scan.add_argument("--run-dir", required=True, type=Path, help="directory holding checkpoints")
    scan.add_argument("--size", type=int, default=512, help="square output size (default: 512)")
    scan.set_defaults(func=cmd_scan)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
