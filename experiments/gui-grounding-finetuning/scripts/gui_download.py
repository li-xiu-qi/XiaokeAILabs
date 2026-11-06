import os
import hashlib, json, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests

root = Path(os.environ['GUI_ROOT']).expanduser() / 'gui-screenspot'
root.mkdir(parents=True, exist_ok=True)
rows = []
for offset in (0, 450, 900):
    response = requests.get('https://datasets-server.huggingface.co/rows', params={'dataset':'bevaya/ScreenSpot','config':'default','split':'test','offset':offset,'length':10}, timeout=60)
    response.raise_for_status()
    rows.extend(response.json()['rows'])

def get(item):
    row = item['row']; image = row['image']; idx = item['row_idx']
    resp = requests.get(image['src'], timeout=90); resp.raise_for_status()
    name = f'{idx:04d}.png'
    (root/name).write_bytes(resp.content)
    return {'row_idx':idx,'file':name,'sha256':hashlib.sha256(resp.content).hexdigest(),
            'bbox_xyxy_normalized':row['bbox'],'instruction':row['instruction'],
            'data_type':row['data_type'],'data_source':row['data_source'],
            'width':image['width'],'height':image['height']}

with ThreadPoolExecutor(max_workers=4) as pool:
    result=list(pool.map(get,rows))
(root/'manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({'count':len(result),'sources':sorted(set(x['data_source'] for x in result)),'data_types':sorted(set(x['data_type'] for x in result))}))
