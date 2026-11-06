import json
from pathlib import Path

data = json.loads((Path(__file__).parent / "results.json").read_text(encoding="utf-8"))
for name in ("original", "supervised", "distilled"):
    entry = data[name]
    ranks = entry["ranks"]
    assert len(ranks) == entry["n"] == 192
    assert all(1 <= r <= entry["candidates"] == 64 for r in ranks)
    assert sum(r == 1 for r in ranks) == entry["r1_count"]
    assert sum(r <= 5 for r in ranks) == entry["r5_count"]
    print(f"{name}: R@1={entry['r1_count']}/192 R@5={entry['r5_count']}/192")
before, after = data["supervised"]["ranks"], data["distilled"]["ranks"]
improved = sum(a != 1 and b == 1 for a, b in zip(before, after))
worsened = sum(a == 1 and b != 1 for a, b in zip(before, after))
print(f"distillation vs supervised: {improved} entered first, {worsened} left first")
