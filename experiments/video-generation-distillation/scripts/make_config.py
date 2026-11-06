"""Materialize local paths into the ai-toolkit training config."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--template', required=True)
parser.add_argument('--data', required=True)
parser.add_argument('--model', required=True)
parser.add_argument('--output', required=True, help='Checkpoint output directory')
parser.add_argument('--config', required=True, help='New generated YAML path')
args = parser.parse_args()
text = Path(args.template).read_text(encoding='utf-8')
for key, value in {'__DATA__': args.data, '__MODEL__': args.model, '__OUTPUT__': args.output}.items():
    if text.count(key) != 1:
        raise SystemExit(f'Invalid template: {key}')
    text = text.replace(key, json.dumps(str(Path(value).resolve())))
with Path(args.config).open('x', encoding='utf-8') as output:
    output.write(text)
