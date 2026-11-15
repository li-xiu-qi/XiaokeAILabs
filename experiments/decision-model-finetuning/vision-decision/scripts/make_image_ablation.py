#!/usr/bin/env python
"""Create a test-only copy with original images or same-size white images."""
import argparse
import base64
import io
import json
from pathlib import Path

from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True, help='Prepared dataset directory')
    parser.add_argument('--out', type=Path, required=True, help='New output directory')
    parser.add_argument('--white', action='store_true', help='Replace each image with a same-size white image')
    args = parser.parse_args()
    source = args.data / 'test.jsonl'
    rows = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not rows:
        raise ValueError('Empty test split')
    if args.white:
        for row in rows:
            image = Image.open(io.BytesIO(base64.b64decode(row['image_png_b64'])))
            buffer = io.BytesIO()
            Image.new('RGB', image.size, 'white').save(buffer, format='PNG')
            row['image_png_b64'] = base64.b64encode(buffer.getvalue()).decode('ascii')
    args.out.mkdir(parents=True, exist_ok=False)
    with (args.out / 'test.jsonl').open('x', encoding='utf-8', newline='\n') as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(f'Prepared {len(rows)} test rows; white images: {args.white}')


if __name__ == '__main__':
    main()
