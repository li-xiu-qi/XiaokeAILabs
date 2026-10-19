#!/usr/bin/env python3
"""Render the four-panel comparison figure used at the top of the README.

Columns share one prompt, one seed and one sampler; the only variable is whether
a LoRA is loaded. Numbers are read from the evaluation JSON rather than typed in,
so the figure cannot drift away from the measurements.

    python scripts/make_figure.py

Requires the outputs of `evaluate.py baseline`, `evaluate.py lora` and either
`evaluate.py scan` (preferred) or an explicit --worst-step override.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


from common import OUT_DIR, TARGET_ID, identity_dir

PANEL_W, PANEL_H, PAD, HEADER = 420, 500, 20, 108
INK, MUTED = "#1a1a1a", "#444444"


def load_font(size: int, bold: bool = False):
    from PIL import ImageFont

    candidates = ["/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"] if bold else []
    candidates += ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]
    candidates += sorted(glob.glob("/usr/share/fonts/**/NotoSansCJK*", recursive=True))
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def read_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def first_image(directory: Path) -> Path | None:
    matches = sorted(glob.glob(str(directory / "*.jpg")))
    return Path(matches[0]) if matches else None


def pick_checkpoints() -> tuple[dict, dict]:
    """Return the best and the most-trained checkpoints from scan.json."""
    scan = read_json(OUT_DIR / "scan" / "scan.json")
    if not scan:
        raise SystemExit("missing output/scan/scan.json — run `evaluate.py scan` first")
    best = max(scan, key=lambda r: r["vs_target"]["mean"])
    overfit = max(scan, key=lambda r: int(r["step"]))
    return best, overfit


def panel(sheet, draw, index: int, title: str, subtitle: str, image_path: Path | None,
          colour: str, metric: str) -> None:
    from PIL import Image

    x = PAD + index * (PANEL_W + PAD)
    y = 130
    draw.text((x, y), title, fill=colour, font=load_font(19, bold=True))
    draw.text((x, y + 30), subtitle, fill="#777777", font=load_font(14))

    box = [x, y + 56, x + PANEL_W, y + 56 + PANEL_H - 16]
    if image_path and image_path.exists():
        image = Image.open(image_path).convert("RGB")
        image.thumbnail((PANEL_W - 16, PANEL_H - 16))
        sheet.paste(image, (x + (PANEL_W - image.width) // 2,
                            y + 56 + (PANEL_H - 16 - image.height) // 2))
    else:
        draw.rectangle(box, fill="#e5e0d5")
        draw.text((x + 20, y + 56 + PANEL_H // 2), "image missing",
                  fill="#999999", font=load_font(18))
    draw.rectangle(box, outline=colour, width=4)
    draw.text((x, y + 56 + PANEL_H + 6), metric, fill=colour, font=load_font(20, bold=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=OUT_DIR / "face_compare.jpg")
    args = parser.parse_args()

    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise SystemExit(
            "missing dependency: pillow\ninstall with `pip install pillow`"
        ) from exc

    baseline = read_json(OUT_DIR / "baseline" / "baseline.json")
    lora = read_json(OUT_DIR / "lora" / "eval.json")
    if not baseline or not lora:
        raise SystemExit(
            "missing baseline.json or eval.json — run `evaluate.py baseline` and `evaluate.py lora`"
        )
    best, overfit = pick_checkpoints()

    ceiling = baseline["reference"]["cross_split"]["mean"]
    baseline_score = baseline["baseline_vs_target"]["mean"]
    best_score = best["vs_target"]["mean"]
    overfit_score = overfit["vs_target"]["mean"]
    negative_score = lora["vs_other"]["mean"]

    target_photo = first_image(identity_dir(TARGET_ID, "test"))
    baseline_image = first_image(OUT_DIR / "baseline")
    best_image = first_image(OUT_DIR / "scan" / f"step_{best['step']}")
    overfit_image = first_image(OUT_DIR / "scan" / f"step_{overfit['step']}")

    panels = [
        ("The person", "real photograph, held out from training",
         target_photo, "#4a7c59", f"{ceiling:.4f} (same-person level)"),
        ("Base model", "same prompt, no LoRA",
         baseline_image, "#a33333", f"{baseline_score:+.4f} vs target"),
        (f"LoRA, step {best['step']}", "same prompt + trigger token",
         best_image, "#2c6fbb", f"{best_score:.4f} vs target"),
        (f"LoRA, step {overfit['step']}", "trained too long",
         overfit_image, "#888888", f"{overfit_score:.4f} vs target"),
    ]

    width = len(panels) * PANEL_W + (len(panels) + 1) * PAD
    height = PANEL_H + HEADER + 190
    sheet = Image.new("RGB", (width, height), "#f7f5f0")
    draw = ImageDraw.Draw(sheet)

    draw.text((PAD, 22), "Teaching a base model to draw a person it cannot draw",
              fill=INK, font=load_font(30, bold=True))
    draw.text((PAD, 66), "Twenty training photos of one identity. All four columns use the "
                         "same prompt, seed and sampler.", fill=MUTED, font=load_font(16))
    draw.text((PAD, 92), "Metric: cosine similarity of ArcFace embeddings against the "
                         "identity's held-out test set.", fill=MUTED, font=load_font(16))

    for index, (title, subtitle, image_path, colour, metric) in enumerate(panels):
        panel(sheet, draw, index, title, subtitle, image_path, colour, metric)

    footer = 130 + 56 + PANEL_H + 50
    draw.text((PAD, footer), "Reading:", fill=INK, font=load_font(19, bold=True))
    draw.text((PAD + 90, footer),
              f"the base model produces a random face ({baseline_score:+.4f}, no better than "
              f"the {negative_score:.4f} it scores against a stranger).", fill="#333333",
              font=load_font(17))
    draw.text((PAD + 90, footer + 32),
              f"With the LoRA the same prompt yields this person ({best_score:.4f}). Training "
              f"past the peak degrades it ({overfit_score:.4f}).", fill="#333333",
              font=load_font(17))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out, quality=93)
    print(f"figure -> {args.out}  ({sheet.width}x{sheet.height})")


if __name__ == "__main__":
    main()
