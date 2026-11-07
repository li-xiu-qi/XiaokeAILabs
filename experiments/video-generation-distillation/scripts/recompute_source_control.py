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
key_rows = json.loads((root / "blind-key.json").read_text(encoding="utf-8"))
key = {(entry["id"], entry["label"]): entry["group"] for entry in key_rows}
holdout = json.loads((root / "holdout.json").read_text(encoding="utf-8"))["eval"]
training = json.loads((root / "train-prompts.json").read_text(encoding="utf-8"))["train"]
ids = {row['id'] for row in holdout}
expected = {(qid, label) for qid in ids for label in ('A', 'B', 'C')}
if len(holdout) != 4 or len(ids) != 4 or len(key_rows) != 12 or set(key) != expected:
    raise ValueError('Holdout, labels, or mapping contain duplicate or missing entries')
if any({key[(qid, label)] for label in ('A', 'B', 'C')} != {'baseline', 'teacher', 'student'} for qid in ids):
    raise ValueError('Each holdout item must have exactly one of each condition')
if set(hashes['videos']) != {f'{qid}/{label}.mp4' for qid, label in expected}:
    raise ValueError('Recorded video hashes do not match the review items')
if {(r['prompt'], r['seed']) for r in training} & {(r['prompt'], r['seed']) for r in holdout}:
    raise ValueError('Training and holdout have identical prompt/seed records')
for row in holdout:
    if (review / row['id'] / 'prompt.txt').read_text(encoding='utf-8').strip() != row['prompt']:
        raise ValueError(f"Review prompt differs from holdout: {row['id']}")
with (review / "ratings.csv").open(encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))
if len(rows) != 12 or len({(r["id"], r["label"]) for r in rows}) != 12:
    raise ValueError("Expected twelve unique rated clips")
if {(r['id'], r['label']) for r in rows} != expected:
    raise ValueError('Ratings do not match holdout items and labels')
if any(r[field] not in ('0', '1', '2') for r in rows for field in ('prompt_match_0_to_2', 'visual_quality_0_to_2')):
    raise ValueError('Ratings must be integers from 0 to 2')
unmasked = json.loads((root / 'blind-unmasked-results.json').read_text(encoding='utf-8'))
expected_unmasked = {group: [] for group in ('baseline', 'teacher', 'student')}
for r in rows:
    expected_unmasked[key[(r['id'], r['label'])]].append({
        'id': r['id'], 'label': r['label'], 'prompt': int(r['prompt_match_0_to_2']),
        'quality': int(r['visual_quality_0_to_2'])})
if any(sorted(unmasked.get(group, []), key=lambda r: (r['id'], r['label'])) !=
       sorted(expected_unmasked[group], key=lambda r: (r['id'], r['label'])) for group in expected_unmasked):
    raise ValueError('Unmasked results differ from ratings and mapping')
for group in ("baseline", "teacher", "student"):
    selected = [r for r in rows if key[(r["id"], r["label"])] == group]
    if len(selected) != 4 or any(r["motion_0_to_2"] != "cannot judge" for r in selected):
        raise ValueError(f"Unexpected review data: {group}")
    prompt = mean(int(r["prompt_match_0_to_2"]) for r in selected)
    quality = mean(int(r["visual_quality_0_to_2"]) for r in selected)
    print(f"{group}: n={len(selected)}, prompt={prompt:.2f}, quality={quality:.2f}, motion=not assessed")
print("Verified ratings and key SHA-256; video hashes are recorded but cannot be checked without the unpublished MP4 files.")
