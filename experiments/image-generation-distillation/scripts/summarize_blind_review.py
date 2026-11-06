"""Reveal and aggregate a completed, anonymous binary image review."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

FIELDS = ("object_correct", "background_correct", "complete_subject", "obvious_artifact")
ARMS = ("teacher", "student-before", "student-after")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    key = json.loads(args.key.read_text(encoding="utf-8"))
    with args.review.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    ids = [r["review_id"] for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(key):
        raise ValueError("score file must contain every anonymized image exactly once")
    grouped = defaultdict(dict)
    totals = {arm: {field: {"yes": 0, "no": 0, "missing": 0} for field in FIELDS} for arm in ARMS}
    for row in rows:
        item = key[row["review_id"]]
        arm, case = item["arm"], item["id"]
        if arm not in ARMS or arm in grouped[case]:
            raise ValueError("unexpected or duplicate arm")
        grouped[case][arm] = row
        for field in FIELDS:
            value = row[field].strip()
            if value not in ("0", "1", ""):
                raise ValueError(f"invalid {field}: {value}")
            totals[arm][field][{"1": "yes", "0": "no", "": "missing"}[value]] += 1
    if any(set(arms) != set(ARMS) for arms in grouped.values()):
        raise ValueError("each prompt must have all three arms")
    pair = {field: {"before_0_after_1": 0, "before_1_after_0": 0,
                    "unchanged": 0, "missing": 0} for field in FIELDS}
    for arms in grouped.values():
        for field in FIELDS:
            before = arms["student-before"][field].strip()
            after = arms["student-after"][field].strip()
            bucket = ("missing" if not before or not after else
                      "unchanged" if before == after else
                      "before_0_after_1" if before == "0" else "before_1_after_0")
            pair[field][bucket] += 1
    result = {"n_prompts": len(grouped), "n_images": len(rows),
              "by_arm": totals, "student_before_after_pairs": pair}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
