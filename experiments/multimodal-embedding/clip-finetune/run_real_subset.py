"""Run the CLIP training entry point on a bounded Flickr30k subset."""
import argparse
from collections import Counter
from pathlib import Path

from datasets import DatasetDict, load_dataset, load_from_disk

import data_common
import finetune_clip_with_freeze as training

REVISION = "a2ac1f0bcee3132331324f1c435bcea4527c08f1"
LIMITS = {"train": 256, "val": 64, "test": 64}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Local Transformers CLIP model directory")
    parser.add_argument("--out", required=True, help="New empty output directory")
    parser.add_argument("--local-dataset", help="Optional locally saved copy of the pinned dataset or subset")
    args = parser.parse_args()
    output = Path(args.out)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("--out must be a new or empty directory")
    if args.local_dataset:
        raw = load_from_disk(args.local_dataset)
        if isinstance(raw, DatasetDict):
            raw = raw["train"]
        print("Local dataset supplied: verify its source and revision independently.")
    else:
        raw = load_dataset("WhereIsAI/flickr30k-v2", revision=REVISION, split="train")
    counts = Counter()
    indices = []
    for i, split in enumerate(raw["split"]):
        if split in LIMITS and counts[split] < LIMITS[split]:
            counts[split] += 1
            indices.append(i)
        if all(counts[key] == value for key, value in LIMITS.items()):
            break
    if dict(counts) != LIMITS:
        raise ValueError(f"Insufficient source records: {dict(counts)}")
    subset = raw.select(indices)
    data_common.Flickr30kDataBuilder._load_raw = lambda self: DatasetDict({"train": subset})
    training.CONFIG.update(
        model_name=args.model,
        dataset_revision=REVISION if not args.local_dataset else None,
        output_dir=str(output),
        train_batch_size=16,
        eval_batch_size=32,
        num_epochs=1,
        bf16=True,
        fp16=False,
        fast_dev_run=False,
    )
    training.main()


if __name__ == "__main__":
    main()
