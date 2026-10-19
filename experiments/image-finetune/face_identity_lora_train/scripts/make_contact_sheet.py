#!/usr/bin/env python3
"""Build a contact sheet of the exported identity for human review before training.

Machine checks catch mislabels and leaks; they cannot catch "these twenty photos
do not look like the same person at a glance" or a duplicate that hashes
differently. Reviewing the sheet is the last gate before the GPU budget is spent.

    python scripts/make_contact_sheet.py
    python scripts/make_contact_sheet.py --celeb-id 15 --out negative_sheet.jpg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


from common import DATA_DIR, OUT_DIR, TARGET_ID, _require

CELL_W, CELL_H, PAD, COLS = 260, 320, 14, 6
TRAIN_COLOUR, TEST_COLOUR = "#2c6fbb", "#d98324"


def load_font(size: int, bold: bool = False):
    """Prefer a CJK-capable font; fall back to Pillow's bitmap default."""
    import glob as _glob

    from PIL import ImageFont

    candidates = ["/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"] if bold else []
    candidates += ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]
    candidates += sorted(_glob.glob("/usr/share/fonts/**/NotoSansCJK*", recursive=True))
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def build(celeb_id: int, output: Path) -> None:
    from PIL import Image, ImageDraw

    source = DATA_DIR / f"celeb_{celeb_id}"
    manifest_path = source / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(
            f"no manifest at {manifest_path}\nrun scripts/prepare_data.py first"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = manifest["items"]

    rows = (len(items) + COLS - 1) // COLS
    width = COLS * CELL_W + (COLS + 1) * PAD
    height = rows * CELL_H + (rows + 1) * PAD + 108
    sheet = Image.new("RGB", (width, height), "#f4f1ea")
    draw = ImageDraw.Draw(sheet)

    draw.text(
        (PAD, 14),
        f"celeb_id={celeb_id} — {len(items)} images",
        fill="#222",
        font=load_font(24, bold=True),
    )
    draw.text(
        (PAD, 46),
        f"blue = train ({manifest['n_train']}, used for LoRA)    "
        f"orange = test ({manifest['n_test']}, evaluation only, never trained on)",
        fill="#444",
        font=load_font(16),
    )
    draw.text(
        (PAD, 72),
        "Original resolution is 178x218. Metric: ArcFace similarity between a generated "
        "face and this identity's test set.",
        fill="#777",
        font=load_font(14),
    )

    for i, item in enumerate(items):
        row, col = divmod(i, COLS)
        x = PAD + col * (CELL_W + PAD)
        y = 92 + PAD + row * (CELL_H + PAD)
        path = source / item["split"] / item["file"]
        image = Image.open(path).convert("RGB")
        image = image.resize((CELL_W - 20, int(image.height * (CELL_W - 20) / image.width)))
        sheet.paste(image, (x + (CELL_W - image.width) // 2, y + (CELL_H - 46 - image.height) // 2))

        colour = TRAIN_COLOUR if item["split"] == "train" else TEST_COLOUR
        draw.rectangle([x, y, x + CELL_W, y + CELL_H - 40], outline=colour, width=4)
        draw.text((x + 4, y + CELL_H - 34), item["file"], fill="#333", font=load_font(13, bold=True))
        draw.text(
            (x + 4, y + CELL_H - 18),
            "train" if item["split"] == "train" else "test",
            fill=colour,
            font=load_font(13, bold=True),
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92)
    print(f"contact sheet -> {output}  ({sheet.width}x{sheet.height})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--celeb-id", type=int, default=TARGET_ID)
    parser.add_argument("--out", type=Path, default=None, help="output .jpg path")
    args = parser.parse_args()

    output = args.out or OUT_DIR / f"celeb_{args.celeb_id}_sheet.jpg"
    build(args.celeb_id, output)


if __name__ == "__main__":
    main()
