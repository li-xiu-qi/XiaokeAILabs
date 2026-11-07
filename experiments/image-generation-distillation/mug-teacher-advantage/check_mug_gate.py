"""Apply the unchanged 32-pair gate to anonymized-then-revealed mug ratings."""
import argparse,csv,importlib.util,json
from pathlib import Path
root=Path(__file__).parent
spec=importlib.util.spec_from_file_location('camera_gate',root/'check_development_gate.py')
gate=importlib.util.module_from_spec(spec); spec.loader.exec_module(gate)
p=argparse.ArgumentParser()
p.add_argument('--scores',type=Path,required=True)
p.add_argument('--development',type=Path,default=root/'mug-frozen/development.jsonl')
a=p.parse_args()
data=[json.loads(line) for line in a.development.read_text(encoding='utf-8').splitlines()]
ids={r['id'] for r in data}
if len(data)!=32 or len(ids)!=32: raise ValueError('expected 32 unique development prompts')
with a.scores.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
for row in rows: row['single_camera']=row['single_mug']
print(json.dumps(gate.evaluate(rows,ids),indent=2))
