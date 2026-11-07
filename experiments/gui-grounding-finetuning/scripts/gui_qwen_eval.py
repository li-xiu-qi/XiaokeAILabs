import os
import json, re, time, sys, hashlib, os
from pathlib import Path
from gui_validate import parse_point
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

ROOT = Path(os.environ['GUI_ROOT']).expanduser()
DATA = ROOT / 'gui-screenspot'
MODEL = Path(os.environ['GUI_MODEL']).expanduser()
adapter=os.environ.get('GUI_ADAPTER')
OUT = ROOT / ('qwen3vl-2b-lora-screenspot-30.jsonl' if adapter else 'qwen3vl-2b-screenspot-30.jsonl')
PROMPT = '''Your task is to help the user identify the precise coordinates (x, y) of a specific area/element/object on the screen based on a description.
- Your response should aim to point to the center or a representative point within the described area/element/object as accurately as possible.
- If the description is unclear or ambiguous, infer the most relevant area or element based on its likely context or purpose.
- Your answer should be a single string (x, y) corresponding to the point of the interest.
- Express each coordinate as an integer from 0 to 1000, relative to the width and height of the image.

Description: {description}

Answer:'''

def check():
    for line in open('/proc/meminfo'):
        if line.startswith('MemAvailable:') and int(line.split()[1]) < int(os.environ.get('GUI_MIN_AVAILABLE_GIB', '50')) * 1024 * 1024:
            raise RuntimeError('available system memory below 50 GiB')

samples = json.loads((DATA/'manifest.json').read_text(encoding='utf8'))
context_path=OUT.with_suffix('.context.json')
context={'manifest_sha256':hashlib.sha256((DATA/'manifest.json').read_bytes()).hexdigest(),
         'model_path':str(MODEL.resolve()),'adapter_path':str(Path(adapter).resolve()) if adapter else None,
         'prompt_sha256':hashlib.sha256(PROMPT.encode('utf8')).hexdigest()}
if OUT.exists():
    if not context_path.exists() or json.loads(context_path.read_text(encoding='utf8')) != context:
        raise RuntimeError('Evaluation output exists without matching context; choose a new GUI_ROOT')
else:
    context_path.write_text(json.dumps(context,indent=2),encoding='utf8')
done = [json.loads(x) for x in OUT.read_text(encoding='utf8').splitlines()] if OUT.exists() else []
assert len(samples) == 30 and all(done[i]['row_idx'] == samples[i]['row_idx'] for i in range(len(done)))
check(); torch.cuda.set_per_process_memory_fraction(.20)
processor = AutoProcessor.from_pretrained(MODEL, min_pixels=256*32*32, max_pixels=512*32*32, local_files_only=True)
model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16, local_files_only=True).to('cuda').eval()
if adapter:
    from peft import PeftModel
    model=PeftModel.from_pretrained(model,adapter).eval()
for item in samples[len(done):]:
    check()
    path=DATA/item['file']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
    img=Image.open(path).convert('RGB')
    assert img.size == (item['width'], item['height'])
    messages=[{'role':'user','content':[{'type':'image','image':img}, {'type':'text','text':PROMPT.format(description=item['instruction'])}]}]
    text=processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos=process_vision_info(messages)
    inputs=processor(text=[text], images=images, videos=videos, padding=True, return_tensors='pt').to('cuda')
    check(); torch.cuda.synchronize(); start=time.perf_counter()
    with torch.inference_mode():
        ids=model.generate(**inputs, max_new_tokens=24, do_sample=False)
    torch.cuda.synchronize(); sec=time.perf_counter()-start
    answer=processor.decode(ids[0][inputs['input_ids'].shape[-1]:],skip_special_tokens=True).strip()
    point=parse_point(answer)
    box=item['bbox_xyxy_normalized']
    hit=bool(point and 0 <= point[0] <= 1 and 0 <= point[1] <= 1 and box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3])
    result={'row_idx':item['row_idx'],'source':item['data_source'],'type':item['data_type'],'bbox':box,'prediction':answer,'point_normalized':point,'hit':hit,'generation_s':sec}
    with OUT.open('a',encoding='utf8') as f:
        f.write(json.dumps(result,ensure_ascii=False)+'\n'); f.flush()
    print(json.dumps(result,ensure_ascii=False), flush=True)
