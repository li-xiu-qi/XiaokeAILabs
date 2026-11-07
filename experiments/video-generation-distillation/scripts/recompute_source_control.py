"""Verify the locked blind-review evidence and recompute descriptive means."""
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean

root = Path(__file__).resolve().parents[1] / "results" / "source-control"
review = root / "blind-review"
hashes = json.loads((root / "blind-pre-unmask-sha256.json").read_text(encoding="utf-8"))

def verify(path, expected):
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch: {path}")

verify(root / "blind-key.json", hashes["blind-key.json"])
verify(review / "ratings.csv", hashes["ratings.csv"])
if len(hashes["videos"]) != 12:
    raise ValueError("Expected twelve videos")
key = {(entry["id"], entry["label"]): entry["group"] for entry in json.loads((root / "blind-key.json").read_text(encoding="utf-8"))}
with (review / "ratings.csv").open(encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))
if len(rows) != 12 or len({(r["id"], r["label"]) for r in rows}) != 12:
    raise ValueError("Expected twelve unique rated clips")
for group in ("baseline", "teacher", "student"):
    selected = [r for r in rows if key[(r["id"], r["label"])] == group]
    if len(selected) != 4 or any(r["motion_0_to_2"] != "cannot judge" for r in selected):
        raise ValueError(f"Unexpected review data: {group}")
    prompt = mean(int(r["prompt_match_0_to_2"]) for r in selected)
    quality = mean(int(r["visual_quality_0_to_2"]) for r in selected)
    print(f"{group}: n={len(selected)}, prompt={prompt:.2f}, quality={quality:.2f}, motion=not assessed")
print("Verified ratings and key SHA-256; video hashes are recorded but cannot be checked without the unpublished MP4 files.")