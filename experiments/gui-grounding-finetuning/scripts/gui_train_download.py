import os
import hashlib, json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests

root = Path(os.environ['GUI_ROOT']).expanduser() / 'gui-groundcua-train'
root.mkdir(parents=True,exist_ok=True)
rows=[]
for offset in (0,1000,2500,4000,6000,8000,10000,12000):
    r=requests.get('https://datasets-server.huggingface.co/rows',params={'dataset':'PrentisAI/GroundGUI_RL','config':'default','split':'train','offset':offset,'length':16},timeout=90)
    r.raise_for_status(); rows.extend(r.json()['rows'])

def download(item):
    row=item['row']; image=row['images'][0]; meta=json.loads(row['additional_paras'])
    assert meta['image_size']==[image['width'],image['height']]
    bbox=row['solution']['arguments']['coordinate']
    assert len(bbox)==4 and 0<=bbox[0]<bbox[2]<=image['width'] and 0<=bbox[1]<bbox[3]<=image['height']
    user=row['messages'][1]['content'].replace('<image>','').strip()
    assert user
    r=requests.get(image['src'],timeout=90); r.raise_for_status()
    name=f"{item['row_idx']:05d}.png"; (root/name).write_bytes(r.content)
    return {'row_idx':item['row_idx'],'sample_id':row['sample_id'],'file':name,'sha256':hashlib.sha256(r.content).hexdigest(),
            'instruction':user,'bbox_pixels':bbox,'width':image['width'],'height':image['height'],
            'app':meta.get('app'),'type':meta.get('ui_type'),'element':meta.get('elem_text')}

with ThreadPoolExecutor(max_workers=4) as pool: results=list(pool.map(download,rows))
(root/'manifest.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({'count':len(results),'unique_images':len({r['sha256'] for r in results}),'apps':len({r['app'] for r in results})}))
