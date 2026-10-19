#!/usr/bin/env python3
"""Export one CelebA identity into train/test splits and write training captions.

Reads a CelebA parquet shard, keeps every image belonging to `--celeb-id`,
splits them alternately (2 of every 3 go to train) so that lighting and pose
are represented on both sides, and writes a manifest for later auditing.

    python scripts/prepare_data.py --parquet /path/to/celeba_shard0.parquet

Output layout (under $FACE_DATA_DIR, default ./data):

    celeb_14/train/*.jpg + *.txt    20 images with captions
    celeb_14/test/*.jpg             10 images, never seen during training
    celeb_14/manifest.json          file, split, source index, size
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

from common import DATA_DIR, NEGATIVE_ID, TARGET_ID, TRIGGER


def image_bytes(value):
    """HF datasets store images either as raw bytes or as a {"bytes", "path"} struct."""
    return value.get("bytes") if isinstance(value, dict) else value


def export(parquet: Path, celeb_id: int, out_root: Path) -> None:
    try:
        import pyarrow.parquet as pq
        from PIL import Image
    except ImportError as exc:
        raise SystemExit(
            f"missing dependency: {exc.name}\nrequired: pyarrow, pillow — "
            "install with `pip install pyarrow pillow`"
        ) from exc

    table = pq.read_table(parquet)
    ids = table.column("celeb_id").to_pylist()
    indices = [i for i, v in enumerate(ids) if v == celeb_id]
    if not indices:
        raise SystemExit(f"celeb_id={celeb_id} not found in {parquet}")

    images = table.column("image").to_pylist()
    out_dir = out_root / f"celeb_{celeb_id}"
    for split in ("train", "test"):
        (out_dir / split).mkdir(parents=True, exist_ok=True)

    manifest = []
    for order, idx in enumerate(indices):
        raw = image_bytes(images[idx])
        if raw is None:
            print(f"  skip index {idx}: empty image column", file=sys.stderr)
            continue
        split = "test" if order % 3 == 2 else "train"
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        name = f"{split[:2]}_{idx:03d}.jpg"
        image.save(out_dir / split / name, quality=95)
        if split == "train":
            caption = f"{TRIGGER}, a photo of a person"
            (out_dir / split / name).with_suffix(".txt").write_text(caption, encoding="utf-8")
        manifest.append(
            {"file": name, "split": split, "src_idx": idx, "size": list(image.size)}
        )

    n_train = sum(1 for m in manifest if m["split"] == "train")
    n_test = len(manifest) - n_train
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {"celeb_id": celeb_id, "n_train": n_train, "n_test": n_test, "items": manifest},
            indent=1,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"celeb_id={celeb_id}: {n_train} train / {n_test} test -> {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parquet", required=True, type=Path, help="CelebA parquet shard")
    parser.add_argument("--celeb-id", type=int, default=TARGET_ID, help="identity to export (default: 14)")
    parser.add_argument(
        "--also-negative",
        action="store_true",
        help=f"also export the negative-control identity (default: {NEGATIVE_ID})",
    )
    args = parser.parse_args()

    export(args.parquet, args.celeb_id, DATA_DIR)
    if args.also_negative:
        export(args.parquet, NEGATIVE_ID, DATA_DIR)


if __name__ == "__main__":
    main()
