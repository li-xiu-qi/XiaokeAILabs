"""Generate fully synthetic non-overlapping document pages; no external data."""
import argparse,hashlib,json,random
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);root=parser.parse_args().output
root.mkdir(parents=True,exist_ok=False)
rng=random.Random(20261107);codes=rng.sample(range(10000,99999),160); values=iter(rng.sample(sorted(set(range(10000,99999))-set(codes)),480))
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',25)
bold=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',35)
records=[]
for i,code in enumerate(codes):
 split='train' if i<96 else 'validation' if i<128 else 'test'
 folder=root/split;folder.mkdir(exist_ok=True)
 page=Image.new('RGB',(900,700),'white');draw=ImageDraw.Draw(page)
 draw.rectangle((0,0,900,95),fill='#143a43');draw.text((35,25),'OPERATIONS REPORT',font=bold,fill='white')
 draw.text((45,125),f'Site ID: ST-{code}',font=bold,fill='#193b3a')
 numbers=[next(values) for _ in range(3)]
 for j,(label,n) in enumerate(zip(['Revenue','Units shipped','Inventory'],numbers)):
  y=240+j*90;draw.rectangle((45,y-10,850,y+55),outline='#bfd1ce',width=2)
  draw.text((65,y),label,font=font,fill='#1b302f');draw.text((580,y),str(n),font=font,fill='#1b302f')
 image=folder/f'page-{i:03d}.png';page.save(image)
 for label,value in zip(['Revenue','Units shipped','Inventory'],numbers):
  records.append({'split':split,'entity':f'ST-{code}','image':str(image.relative_to(root)),
                  'query':f'Which report lists {label.lower()} as {value}?','field':label,'value':value})
entities={s:{r['entity'] for r in records if r['split']==s} for s in ('train','validation','test')}
assert all(not entities[a]&entities[b] for a,b in [('train','validation'),('train','test'),('validation','test')])
assert len({hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*/*.png')})==len(codes)
(root/'records.json').write_text(json.dumps(records,indent=2),encoding='utf8')
print({s:len(entities[s]) for s in entities},'queries',len(records),'unique page hashes',len(codes))
