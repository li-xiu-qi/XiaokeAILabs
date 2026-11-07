"""Generate synthetic training images or matched held-out model comparisons."""
import argparse
import json
import time
from pathlib import Path


def prepare_output(path):
    """Prevent stale images or metadata from contaminating a new run."""
    path.mkdir(parents=True, exist_ok=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--lora")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    import torch
    import diffusers
    from diffusers import DiffusionPipeline

    rows = [json.loads(line) for line in args.prompts.read_text(encoding="utf-8").splitlines()]
    if args.limit is not None:
        rows = rows[:args.limit]
    prepare_output(args.out)
    pipe = DiffusionPipeline.from_pretrained(
        args.model, revision=args.revision, torch_dtype=torch.float16,
        use_safetensors=True,
    ).to("cuda")
    if args.lora:
        pipe.load_lora_weights(args.lora, weight_name="pytorch_lora_weights.safetensors")
    manifest = {"model": args.model, "revision": args.revision, "lora": args.lora,
                "torch": torch.__version__, "diffusers": diffusers.__version__,
                "scheduler": type(pipe.scheduler).__name__,
                "scheduler_config": dict(pipe.scheduler.config),
                "device": torch.cuda.get_device_name(), "steps": args.steps,
                "guidance": args.guidance, "size": args.size}
    (args.out / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    records = []
    for row in rows:
        torch.cuda.synchronize()
        start = time.perf_counter()
        image = pipe(
            row["prompt"], height=args.size, width=args.size,
            num_inference_steps=args.steps, guidance_scale=args.guidance,
            generator=torch.Generator(device="cuda").manual_seed(row["seed"]),
        ).images[0]
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        filename = row["id"] + ".png"
        image.save(args.out / filename)
        records.append({**row, "file_name": filename, "text": row["prompt"],
                        "seconds": elapsed, "model": args.model,
                        "revision": args.revision, "lora": args.lora,
                        "steps": args.steps, "guidance": args.guidance, "size": args.size})
        print(filename, round(elapsed, 2), flush=True)
    with (args.out / "metadata.jsonl").open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
