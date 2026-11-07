import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import torch
import transformers
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoImageProcessor, AutoModelForTableRecognition

parser = argparse.ArgumentParser()
parser.add_argument('--device', choices=['cpu'], default='cpu')
parser.add_argument('--model', required=True)
parser.add_argument('--out', required=True)
parser.add_argument('--crop-table', action='store_true')
args = parser.parse_args()

def serializable(x):
    if isinstance(x, torch.Tensor): return x.detach().cpu().tolist()
    if isinstance(x, dict): return {k: serializable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [serializable(v) for v in x]
    return x

torch.set_num_threads(2)
root = Path(args.out); root.mkdir(parents=True, exist_ok=False)
image = Image.new('RGB', (1000, 500), 'white'); draw = ImageDraw.Draw(image)
font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 30)
for x in [40, 360, 650, 950]: draw.line((x, 100, x, 400), fill='black', width=2)
for y in [100, 175, 250, 325, 400]: draw.line((40, y, 950, y), fill='black', width=2)
for y, items in zip([120,195,270,345], [('Item','Units','Price'),('Notebook','12','6.50'),('Marker','24','2.00'),('Paper','8','4.00')]):
    for x, text in zip([60,380,670], items): draw.text((x,y), text, fill='black', font=font)
if args.crop_table:
    image=image.crop((40,100,951,401))
image.save(root/'synthetic-table.png')
processor = AutoImageProcessor.from_pretrained(args.model, local_files_only=True)
model = AutoModelForTableRecognition.from_pretrained(args.model, local_files_only=True).eval().to(args.device)
inputs = processor(images=image, return_tensors='pt').to(args.device)
t=time.perf_counter()
with torch.inference_mode(): outputs = model(**inputs)
elapsed=time.perf_counter()-t
result=serializable(processor.post_process_table_recognition(outputs))
data={'model_directory':str(Path(args.model).resolve()),
      'model_sha256':hashlib.sha256((Path(args.model)/'model.safetensors').read_bytes()).hexdigest(),
      'input':'self-generated English table: 4 rows x 3 columns, 12 cells, no merges',
      'image_wh':list(image.size),'crop_table':args.crop_table,
      'input_sha256':hashlib.sha256((root/'synthetic-table.png').read_bytes()).hexdigest(),
      'purpose':'table structure only, no text recognition; single synthetic image; no benchmark',
      'device':args.device,'cpu':platform.processor(),'cpu_threads':2,'torch':torch.__version__,
      'transformers':transformers.__version__,'first_forward_s':elapsed,'result':result}
(root/'result.json').write_text(json.dumps(data, ensure_ascii=False, indent=2),encoding='utf8')
print(json.dumps(data, ensure_ascii=False, indent=2))
