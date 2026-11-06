"""Create deterministic, disjoint training and held-out product prompts."""
import argparse
import json
from pathlib import Path

TRAIN = ["ceramic mug", "glass bottle", "leather wallet", "wristwatch",
         "desk lamp", "headphones", "teapot", "backpack"]
HELD_OUT = ["sneaker", "perfume bottle", "camera", "sunglasses"]
COLORS = ["ivory", "navy", "terracotta", "sage green"]
STYLES = ["soft daylight", "dramatic side lighting"]


def prompt(obj, color, lighting):
    return (f"Professional studio product photograph of a single {obj}, "
            f"on a {color} seamless background, {lighting}, "
            "clean composition, realistic material texture, no text, no logo.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for split, objects in (("train", TRAIN), ("heldout", HELD_OUT)):
        rows = [{"id": f"{split}-{i:03d}", "prompt": prompt(obj, color, lighting),
                 "object": obj, "color": color, "lighting": lighting, "seed": 10000 + i}
                for i, (obj, color, lighting) in enumerate(
                    (o, c, l) for o in objects for c in COLORS for l in STYLES)]
        with (args.out / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(split, len(rows))
    assert set(TRAIN).isdisjoint(HELD_OUT)


if __name__ == "__main__":
    main()
