import argparse, ast, csv, json, random, hashlib
from pathlib import Path
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--annotations', type=Path, required=True)
parser.add_argument('--images', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
out = args.out; out.mkdir(parents=True, exist_ok=True)
imgroot = args.images
with args.annotations.open(encoding='utf8', newline='') as f:
    rows = list(csv.DictReader(f))
groups = {key:[r for r in rows if r['split'] == key] for key in ('train','val','test')}
assert all(len(groups[x]) for x in groups)
rng = random.Random(20261107)
train = rng.sample(groups['train'], 512)
val = rng.sample(groups['val'], 128)
test = random.Random(20261107).sample(groups['test'], 100)
used = train + val + test
assert len({r['filename'] for r in used}) == len(used)
all_split = {}
for split, group in groups.items():
    names = {r['filename'] for r in group}
    assert len(names) == len(group)
    all_split[split] = names
assert not (all_split['train'] & all_split['test'] or all_split['train'] & all_split['val'] or all_split['val'] & all_split['test'])
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
hashes={}; dimensions=[]; bad=[]; samples={}
for split, subset in [('train',train),('val',val),('test',test)]:
    samples[split]=[]
    for r in subset:
        path=imgroot/r['filename']
        try:
            caps=ast.literal_eval(r['raw'])
            assert isinstance(caps,list) and len(caps)==5 and all(isinstance(c,str) and c.strip() for c in caps)
            with Image.open(path) as im:
                im.verify()
            with Image.open(path) as im:
                dimensions.append(list(im.size))
            h=sha(path)
            hashes.setdefault(h,[]).append((split,r['filename']))
            samples[split].append({'filename':r['filename'],'caption':caps[0], 'sha256':h})
        except Exception as e:
            bad.append({'filename':r['filename'],'error':repr(e)})
dupes=[v for v in hashes.values() if len(v)>1]
audit={'seed':20261107,'dataset_counts':{k:len(v) for k,v in groups.items()},'sample_counts':{k:len(v) for k,v in samples.items()},'invalid':bad,'byte_identical':dupes,'min_image_wh':[min(d[i] for d in dimensions) for i in range(2)],'max_image_wh':[max(d[i] for d in dimensions) for i in range(2)],'scope':'hash of selected image files only; exact name checked over all three splits; neither perceptual similarity nor provenance established'}
(out/'data-audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
if bad or dupes: raise RuntimeError('data errors or byte-identical images; see data-audit.json')
(out/'samples-private.json').write_text(json.dumps(samples,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(audit,indent=2))
