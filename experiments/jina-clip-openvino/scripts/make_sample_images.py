#!/usr/bin/env python3
"""Create two original geometric demonstration images; never overwrite files."""

import argparse
from pathlib import Path
from PIL import Image, ImageDraw


def render(path: Path, scene: str) -> None:
    image = Image.new("RGB", (640, 416))
    draw = ImageDraw.Draw(image)
    for y in range(416):
        if y < 220:
            ratio = y / 220
            color = (int(34 + 183 * ratio), int(70 + 74 * ratio), int(115 - 44 * ratio))
        else:
            ratio = (y - 220) / 196
            color = (int(26 + 57 * ratio), int(90 + 44 * ratio), int(120 - 20 * ratio))
        draw.line((0, y, 639, y), fill=color)
    draw.ellipse((418, 75, 490, 147), fill=(255, 205, 121))
    if scene == "coast":
        draw.polygon([(0, 320), (180, 282), (366, 325), (640, 280), (640, 416), (0, 416)], fill=(215, 168, 117))
        for y in (234, 251, 269, 289):
            draw.line((15, y, 625, y - 6), fill=(155, 205, 204), width=2)
    else:
        draw.polygon([(0, 260), (130, 128), (268, 260)], fill=(44, 65, 77))
        draw.polygon([(175, 260), (370, 100), (580, 260)], fill=(57, 77, 84))
        draw.polygon([(0, 310), (160, 270), (360, 295), (640, 250), (640, 416), (0, 416)], fill=(44, 99, 80))
    with path.open("xb") as destination:
        image.save(destination, format="PNG", optimize=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = [args.output_dir / "illustrated-coast.png", args.output_dir / "illustrated-mountains.png"]
    if any(path.exists() for path in paths):
        parser.error("an output image already exists; choose a fresh directory")
    for path, scene in zip(paths, ("coast", "mountains")):
        render(path, scene)
        print(path)


if __name__ == "__main__":
    main()
