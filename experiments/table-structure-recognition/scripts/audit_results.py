"""Inspect the published structure tokens; this is not a table benchmark."""
import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path


class Rows(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows = []
        self.current = None

    def handle_starttag(self, tag, attrs):
        if tag == 'tr':
            if self.current is not None:
                raise ValueError('nested row')
            self.current = []
        if tag in ('td', 'th'):
            if self.current is None:
                raise ValueError('cell outside row')
            attrs = dict(attrs)
            self.current.append({'colspan': int(attrs.get('colspan', 1)),
                                 'rowspan': int(attrs.get('rowspan', 1))})

    def handle_endtag(self, tag):
        if tag == 'tr':
            if self.current is None:
                raise ValueError('row closed without opening')
            self.rows.append(self.current)
            self.current = None


def row_cells(tokens):
    parser = Rows()
    parser.feed(''.join(tokens))
    parser.close()
    if parser.current is not None:
        raise ValueError('unclosed row')
    return parser.rows


def main():
    root = Path(__file__).resolve().parents[1]
    original = json.loads((root / 'results/uncropped.json').read_text(encoding='utf-8'))
    cropped = json.loads((root / 'results/cropped.json').read_text(encoding='utf-8'))
    image_hash = hashlib.sha256((root / 'figures/synthetic-table.png').read_bytes()).hexdigest()
    if image_hash != original['input_sha256']:
        raise ValueError('Published original image does not match recorded file hash')
    result = {name: {'rows': row_cells(data['result']['structure']),
                     'structure_score': data['result']['structure_score'],
                     'first_forward_s': data['first_forward_s']}
              for name, data in [('uncropped', original), ('cropped', cropped)]}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
