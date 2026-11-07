"""Check OCR output against the 12 known cells of the supplied synthetic image."""
import argparse
import json
from pathlib import Path

EXPECTED = (
    ("Item", "Units", "Price"),
    ("Notebook", "12", "6.50"),
    ("Marker", "24", "2.00"),
    ("Paper", "8", "4.00"),
)
X_EDGES = (40, 360, 650, 950)
Y_EDGES = (100, 175, 250, 325, 400)


def interval(value, edges):
    return next((i for i, (a, b) in enumerate(zip(edges, edges[1:]))
                 if a <= value < b), None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result', type=Path, help='JSON output from scripts/run.py')
    args = parser.parse_args()
    records = json.loads(args.result.read_text(encoding='utf-8'))
    cells = {}
    for record in records:
        box = record['box']
        x = sum(point[0] for point in box) / len(box)
        y = sum(point[1] for point in box) / len(box)
        cell = (interval(y, Y_EDGES), interval(x, X_EDGES))
        cells.setdefault(cell, []).append(record['prediction']['text'])
    errors = []
    for row, expected_row in enumerate(EXPECTED):
        for col, expected in enumerate(expected_row):
            actual = cells.pop((row, col), [])
            if actual != [expected]:
                errors.append(f'cell ({row + 1}, {col + 1}): expected {expected!r}, got {actual!r}')
    if cells:
        errors.append(f'boxes outside the table: {cells!r}')
    if errors:
        raise SystemExit('\n'.join(errors))
    print('All 12 known cells have one matching transcription.')


if __name__ == '__main__':
    main()
