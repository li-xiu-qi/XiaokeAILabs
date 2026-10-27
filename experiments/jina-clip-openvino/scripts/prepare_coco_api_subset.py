#!/usr/bin/env python3
"""Prepare a COCO 2017 caption subset through Hugging Face Dataset Server."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

API_URL = "https://datasets-server.huggingface.co/rows"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="phiyodr/coco2017")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.limit < 1 or args.workers < 1:
        raise ValueError("limit and workers must be positive")

    candidates = []
    offset = 0
    while len(candidates) < args.limit:
        response = requests.get(
            API_URL,
            params={
                "dataset": args.dataset,
                "config": "default",
                "split": args.split,
                "offset": offset,
                "length": min(100, args.limit - len(candidates)),
            },
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        rows = payload.get("rows", [])
        if not rows:
            break
        for item in rows:
            row = item["row"]
            captions = row.get("captions") or []
            if not captions:
                continue
            filename = Path(row["file_name"]).name
            candidates.append(
                {
                    "image": f"images/{filename}",
                    "caption": captions[0].strip(),
                    "cocoid": row["image_id"],
                    "filename": filename,
                    "split": args.split,
                    "source_url": row["coco_url"],
                }
            )
            if len(candidates) >= args.limit:
                break
        offset += len(rows)
    if len(candidates) != args.limit:
        raise RuntimeError(f"Expected {args.limit} pairs, got {len(candidates)}")

    image_dir = args.output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    def download(pair: dict) -> None:
        destination = args.output_dir / pair["image"]
        if destination.exists() and destination.stat().st_size > 1000:
            if destination.read_bytes()[:2] == b"\xff\xd8":
                return
        last_error = None
        for _ in range(3):
            try:
                response = requests.get(pair["source_url"], timeout=60)
                response.raise_for_status()
                content = response.content
                if len(content) < 1000 or not content.startswith(b"\xff\xd8"):
                    raise ValueError(f"Not a valid JPEG: {pair['source_url']}")
                destination.write_bytes(content)
                return
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"Failed to download {pair['source_url']}: {last_error}")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        list(executor.map(download, candidates))

    subset_path = args.output_dir / "subset.json"
    subset_path.write_text(json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata = {
        "source": args.dataset,
        "dataset_server": API_URL,
        "split": args.split,
        "pair_count": len(candidates),
        "license": "CC BY 4.0",
        "caption_selection": "first caption per image",
    }
    (args.output_dir / "subset.meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"prepared_pairs={len(candidates)} subset={subset_path}")


if __name__ == "__main__":
    main()
