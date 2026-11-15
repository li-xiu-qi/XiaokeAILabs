"""Export a reproducible Flickr30k subset into the experiment CSV format."""
import argparse
import csv
import json
import random
from pathlib import Path


def choose_indices(metadata, counts, seed):
    groups = {key: [] for key in counts}
    seen = set()
    for index, row in enumerate(metadata):
        split = row["split"]
        if split not in groups:
            raise ValueError(f"Unsupported split at row {index}: {split!r}")
        try:
            image_id = int(row["img_id"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid img_id at row {index}") from exc
        if image_id in seen:
            raise ValueError(f"Duplicate image ID: {image_id}")
        seen.add(image_id)
        groups[split].append(index)
    rng = random.Random(seed)
    selected = []
    for split, count in counts.items():
        if count < 0 or len(groups[split]) < count:
            raise ValueError(f"{split} requires {count} images; found {len(groups[split])}")
        selected.extend(rng.sample(groups[split], count))
    return sorted(selected)


def export_rows(dataset, indices, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    images = output / "images"
    images.mkdir()
    with (output / "annotations.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "raw", "filename"])
        writer.writeheader()
        for index in indices:
            row = dataset[index]
            captions = row["caption"]
            if not isinstance(captions, (list, tuple)) or len(captions) != 5 or not all(isinstance(c, str) and c.strip() for c in captions):
                raise ValueError(f"Expected five nonempty captions at row {index}")
            image = row["image"]
            if image is None:
                raise ValueError(f"Missing image at row {index}")
            filename = f"image-{int(row['img_id'])}.png"
            image.convert("RGB").save(images / filename)
            writer.writerow({"split": row["split"], "raw": repr(list(captions)), "filename": filename})
    print(f"Exported {len(indices)} rows to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="New output directory")
    parser.add_argument("--revision", help="Dataset revision, when known")
    parser.add_argument("--train-count", type=int, default=512)
    parser.add_argument("--val-count", type=int, default=128)
    parser.add_argument("--test-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261107)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("--out must not already exist")
    from datasets import load_dataset
    data = load_dataset("WhereIsAI/flickr30k-v2", split="train", revision=args.revision)
    counts = {"train": args.train_count, "val": args.val_count, "test": args.test_count}
    selected = choose_indices(data.select_columns(["split", "img_id"]), counts, args.seed)
    export_rows(data, selected, args.out)
    (args.out / "source.json").write_text(json.dumps({
        "dataset": "WhereIsAI/flickr30k-v2", "revision": args.revision,
        "seed": args.seed, "counts": counts, "source_indices": selected,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
