#!/usr/bin/env python3
"""Score how much of one identity a generation reproduces.

Three sub-commands share one generation and scoring path, so every number is
produced the same way: same prompt template (only the trigger token differs),
same seeds (1000 + i), same sampler, same resolution.

    baseline    the base model with no LoRA, scored against the target identity
                and against the negative-control identity
    lora        one LoRA checkpoint, scored against the same two references
    scan        every checkpoint in a run directory, to find the best step
                without paying for extra training

    python scripts/score_generations.py baseline
    python scripts/score_generations.py lora --lora output/face_identity_lora/face_identity_lora.safetensors
    python scripts/score_generations.py scan --run-dir output/face_identity_lora

Every result records the resolution it was generated at, because similarity is
not comparable across resolutions: a 768px face has more pixels for ArcFace to
work with than a 512px one. `lora` and `scan` refuse to print a gain against a
baseline generated at a different size rather than silently mixing the two.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from face_lib import (
    GUIDANCE,
    NEGATIVE_ID,
    OUT_DIR,
    PROMPT_BASE,
    PROMPT_WITH_LORA,
    SAMPLER,
    SDXL_BASE,
    STEPS,
    TARGET_ID,
    FaceEncoder,
    _require,
    collect_embeddings,
    identity_dir,
    prepare_output_dir,
    prototype,
    summary,
    to_rgb_array,
)

DEFAULT_SIZE = 512


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
    """Generate n images at fixed seeds; return the embeddings of those with a face."""
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


def references(encoder: FaceEncoder) -> dict:
    """Target prototype, target cross-split ceiling, negative-control prototype."""
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
        # The net gain compares two scores of the *same* generated images, so it
        # is valid regardless of resolution; absolute levels are not.
        "net_gain": (
            target_stats["mean"] - negative_stats["mean"]
            if target_stats and negative_stats
            else None
        ),
        "per_image_vs_target": against_target,
    }


def report_reference(refs: dict) -> None:
    print(f"target {TARGET_ID}: {refs['n_train']} train / {refs['n_test']} test images")
    print(f"negative control {NEGATIVE_ID}: {refs['n_negative']} test images")
    print(f"same-person ceiling (train vs test): {refs['cross_split']['mean']:.4f}")


def print_gain(scored: dict, size: int, baseline: dict | None) -> None:
    print(f"\n  resolution         : {size}px")
    print(f"  vs target identity : {scored['vs_target']['mean']:+.4f}  (n={scored['vs_target']['n']})")
    print(f"  vs other identity  : {scored['vs_other']['mean']:+.4f}")
    print(f"  net gain           : {scored['net_gain']:+.4f}   <- same images, both references")
    if baseline is None:
        print("\n  (no baseline.json yet — run `score_generations.py baseline` to add it)")
        return
    if baseline.get("size") != size:
        print(
            f"\n  baseline was generated at {baseline.get('size')}px, this run at {size}px.\n"
            "  Similarity is not comparable across resolutions, so no difference is shown.\n"
            f"  Re-run `score_generations.py baseline --size {size}` to get a matching one."
        )
        return
    print(f"  base model         : {baseline['vs_target']['mean']:+.4f}   <- same settings")


def cmd_baseline(args) -> None:
    encoder = FaceEncoder()
    refs = references(encoder)
    report_reference(refs)

    out_dir = prepare_output_dir("baseline")
    pipe = load_pipeline()
    records = generate(pipe, PROMPT_BASE, args.n, args.size, out_dir, "base_s")
    scored = score(records, refs)

    result = {
        "condition": "base model, no LoRA",
        "prompt": PROMPT_BASE,
        "size": args.size,
        "steps": STEPS,
        "guidance": GUIDANCE,
        "sampler": SAMPLER,
        "n_requested": args.n,
        "n_scored": len(records),
        "reference": {
            "celeb_id": TARGET_ID,
            "negative_id": NEGATIVE_ID,
            "n_train": refs["n_train"],
            "n_test": refs["n_test"],
            "cross_split": refs["cross_split"],
        },
        "vs_target": scored["vs_target"],
        "vs_other": scored["vs_other"],
        "per_image_vs_target": scored["per_image_vs_target"],
    }
    path = out_dir / "baseline.json"
    path.write_text(json.dumps(result, indent=1), encoding="utf-8")

    print("\n=== base model, no LoRA ===")
    print(f"  vs target identity : {scored['vs_target']['mean']:+.4f}")
    print(f"  vs other identity  : {scored['vs_other']['mean']:+.4f}")
    print(f"  same-person ceiling: {refs['cross_split']['mean']:.4f}")
    print(
        "\nInterpretation: if the base model scores near zero against the target and\n"
        "no higher than against a stranger, it cannot draw this person — the target\n"
        "is real and the LoRA has something to prove."
    )
    print(f"result -> {path}")


def cmd_lora(args) -> None:
    encoder = FaceEncoder()
    refs = references(encoder)
    report_reference(refs)

    out_dir = prepare_output_dir("lora")
    pipe = load_pipeline(args.lora)
    records = generate(pipe, PROMPT_WITH_LORA, args.n, args.size, out_dir, "lora_s")
    scored = score(records, refs)

    baseline_path = OUT_DIR / "baseline" / "baseline.json"
    baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else None

    result = {
        "condition": "LoRA",
        "lora": str(args.lora),
        "prompt": PROMPT_WITH_LORA,
        "size": args.size,
        "steps": STEPS,
        "guidance": GUIDANCE,
        "sampler": SAMPLER,
        "n_requested": args.n,
        "n_scored": len(records),
        "vs_target": scored["vs_target"],
        "vs_other": scored["vs_other"],
        "net_gain": scored["net_gain"],
        "per_image_vs_target": scored["per_image_vs_target"],
    }
    path = out_dir / "eval.json"
    path.write_text(json.dumps(result, indent=1), encoding="utf-8")

    print("\n=== with LoRA ===")
    print_gain(scored, args.size, baseline)
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
    report_reference(refs)
    out_root = prepare_output_dir("scan")

    rows = []
    for ckpt in checkpoints:
        step = Path(ckpt).stem.split("_")[-1]
        print(f"\n--- step {step} ---")
        out_dir = out_root / f"step_{step}"
        out_dir.mkdir(parents=True, exist_ok=True)
        pipe = load_pipeline(ckpt)
        records = generate(pipe, PROMPT_WITH_LORA, args.n, args.size, out_dir, "s")
        del pipe
        _require("torch").cuda.empty_cache()
        scored = score(records, refs)
        rows.append(
            {
                "step": step,
                "checkpoint": Path(ckpt).name,
                "size": args.size,
                "n": len(records),
                "vs_target": scored["vs_target"],
                "vs_other": scored["vs_other"],
                "net_gain": scored["net_gain"],
            }
        )
        # Rewrite after each checkpoint so an interrupted scan keeps its progress.
        (out_root / "scan.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
        print(
            f"    vs target {scored['vs_target']['mean']:.4f}"
            f"  vs other {scored['vs_other']['mean']:.4f}"
        )

    print(f"\n=== checkpoint scan ({args.size}px) ===")
    print(f"{'step':<10}{'vs target':>12}{'vs other':>12}{'net gain':>12}{'n':>4}")
    for row in sorted(rows, key=lambda r: r["step"]):
        print(
            f"{row['step']:<10}{row['vs_target']['mean']:>12.4f}"
            f"{row['vs_other']['mean']:>12.4f}{row['net_gain']:>+12.4f}{row['n']:>4}"
        )

    # Two metrics, two possible answers, and they need not agree: `vs target`
    # asks how much of the person is there, `net gain` asks how much more of this
    # person than of a stranger. With ~5-10 detected faces the second one is
    # dominated by whichever comparison happened to land near zero, so report
    # both peaks instead of quietly picking one.
    by_target = max(rows, key=lambda r: r["vs_target"]["mean"])
    by_gain = max(rows, key=lambda r: r["net_gain"])
    print(f"\npeak vs target : step {by_target['step']} ({by_target['vs_target']['mean']:.4f})")
    print(f"peak net gain  : step {by_gain['step']} ({by_gain['net_gain']:+.4f})")
    if by_target["step"] != by_gain["step"]:
        print(
            "  The two disagree. With this many images `net gain` is noisy — its\n"
            "  denominator can sit near zero at one step and inflate the gap. Prefer\n"
            "  the `vs target` peak, and read `n` before trusting either."
        )
    print(
        "\nA peak followed by a decline is overfitting; stop at the peak.\n"
        "Big n differences between steps are generation luck, not a property of the\n"
        "checkpoint — the same seed can yield a detectable face at one step and not\n"
        "at another."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--n", type=int, default=20, help="images per condition (default: 20)")
    parser.add_argument(
        "--size", type=int, default=DEFAULT_SIZE, help=f"square output size (default: {DEFAULT_SIZE})"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("baseline", help="score the base model with no LoRA").set_defaults(
        func=cmd_baseline
    )
    lora = sub.add_parser("lora", help="score one LoRA checkpoint")
    lora.add_argument("--lora", required=True, type=Path, help="path to .safetensors")
    lora.set_defaults(func=cmd_lora)
    scan = sub.add_parser("scan", help="score every checkpoint in a run directory")
    scan.add_argument("--run-dir", required=True, type=Path, help="directory holding checkpoints")
    scan.set_defaults(func=cmd_scan)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
