"""Produce a deterministic blinded review sheet and private identity key."""
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--key", type=Path, required=True,
                    help="Identity key outside the directory shared with reviewers")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    arms = ("teacher", "student-before", "student-after")
    rows = {}
    for arm in arms:
        manifest = args.results / arm / "metadata.jsonl"
        entries = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
        by_id = {entry["id"]: entry for entry in entries}
        if len(by_id) != len(entries):
            raise ValueError(f"duplicate IDs in {manifest}")
        rows[arm] = by_id
    ids = set(rows[arms[0]])
    if not ids or any(set(rows[arm]) != ids for arm in arms):
        raise ValueError("all arms must contain the same nonempty set of prompt IDs")
    args.out.mkdir(parents=True, exist_ok=True)
    tasks = []
    key = {}
    for identifier in sorted(ids):
        source = rows[arms[0]][identifier]
        for arm in arms:
            entry = rows[arm][identifier]
            if (entry["prompt"], entry["seed"]) != (source["prompt"], source["seed"]):
                raise ValueError(f"prompt or seed mismatch: {identifier}")
            img = args.results / arm / entry["file_name"]
            digest = hashlib.sha256(img.read_bytes()).hexdigest()
            code = hashlib.sha256(f"{args.seed}/{identifier}/{arm}".encode()).hexdigest()[:12]
            key[code] = {"arm": arm, "id": identifier, "source": str(img), "sha256": digest}
            tasks.append({"review_id": code, "prompt": source["prompt"],
                          "image": f"images/{code}.png",
                          "object_correct": "", "background_correct": "",
                          "complete_subject": "", "obvious_artifact": "", "notes": ""})
    random.Random(args.seed).shuffle(tasks)
    image_dir = args.out / "images"
    image_dir.mkdir(exist_ok=True)
    for code, value in key.items():
        (image_dir / f"{code}.png").write_bytes(Path(value["source"]).read_bytes())
    with (args.out / "review.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=tasks[0].keys())
        writer.writeheader()
        writer.writerows(tasks)
    if args.key.resolve().is_relative_to(args.out.resolve()):
        raise ValueError("identity key must be outside the reviewer directory")
    args.key.parent.mkdir(parents=True, exist_ok=True)
    args.key.write_text(
        json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Prepared {len(tasks)} images. Keep identity_key.json away from reviewers.")


if __name__ == "__main__":
    main()
