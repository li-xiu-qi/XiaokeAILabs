"""Copy pre-generated teacher videos and matching captions into a training folder."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--manifest', required=True)
parser.add_argument('--source', required=True, help='Directory containing <id>*.mp4')
parser.add_argument('--output', required=True, help='New directory for clips and captions')
parser.add_argument('--ids', nargs='+', required=True, help='Training IDs selected for this run')
args = parser.parse_args()
manifest = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
train = {entry['id']: entry for entry in manifest['train']}
if len(set(args.ids)) != len(args.ids) or not set(args.ids) <= train.keys():
    raise SystemExit('IDs must be unique members of the training split')
sources = []
for sample_id in args.ids:
    matches = list(Path(args.source).glob(sample_id + '*.mp4'))
    if len(matches) != 1:
        raise SystemExit(f'{sample_id}: expected exactly one clip; found {len(matches)}')
    sources.append((train[sample_id], matches[0]))
out = Path(args.output)
out.mkdir(parents=True, exist_ok=False)
records = []
for entry, source in sources:
    target = out / (entry['id'] + '.mp4')
    shutil.copy2(source, target)
    target.with_suffix('.txt').write_text(entry['prompt'] + '\n', encoding='utf-8')
    records.append({**entry, 'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
(out / 'selection.json').write_text(json.dumps(records, indent=2) + '\n', encoding='utf-8')
