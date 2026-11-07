"""Summarize completed binary ratings without dropping missing evaluations."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", type=Path, required=True)
    ap.add_argument("--key", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    key = json.loads(args.key.read_text(encoding="utf-8"))
    with args.review.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != len(key) or {r["review_id"] for r in rows} != set(key):
        raise ValueError("review must contain every randomized item exactly once")
    if len({r["review_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate review ID")
    fields = ("object_correct", "background_correct", "complete_subject", "obvious_artifact")
    totals = defaultdict(lambda: {field: {"yes": 0, "no": 0, "missing": 0} for field in fields})
    for row in rows:
        arm = key[row["review_id"]]["arm"]
        for field in fields:
            value = row[field].strip().lower()
            if value not in ("", "0", "1"):
                raise ValueError(f"{field} must be 1, 0, or blank")
            bucket = {"1": "yes", "0": "no", "": "missing"}[value]
            totals[arm][field][bucket] += 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as output:
        json.dump(dict(totals), output, ensure_ascii=False, indent=2)
    print(json.dumps(dict(totals), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
