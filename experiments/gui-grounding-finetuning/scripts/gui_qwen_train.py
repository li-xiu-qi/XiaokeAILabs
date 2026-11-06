import os
import json, random, time, hashlib
from importlib.metadata import version
from pathlib import Path
import torch
from PIL import Image
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

ROOT=Path(os.environ['GUI_ROOT']).expanduser(); DATA=ROOT/'gui-groundcua-train'; MODEL=Path(os.environ['GUI_MODEL']).expanduser()
OUT=ROOT/'qwen3vl-2b-lora-groundcua'; OUT.mkdir(exist_ok=True)
assert not (OUT/'loss.jsonl').exists(), 'existing training run found; choose a new output directory'
PROMPT='''Your task is to help the user identify the precise coordinates (x, y) of a specific area/element/object on the screen based on a description.
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

seed=20261107; random.seed(seed); torch.manual_seed(seed)
rows=json.loads((DATA/'manifest.json').read_text(encoding='utf8')); assert len(rows)==128
random.shuffle(rows)
check(); torch.cuda.set_per_process_memory_fraction(.20); torch.cuda.reset_peak_memory_stats()
processor=AutoProcessor.from_pretrained(MODEL,min_pixels=256*32*32,max_pixels=512*32*32,local_files_only=True)
base=Qwen3VLForConditionalGeneration.from_pretrained(MODEL,dtype=torch.bfloat16,local_files_only=True).to('cuda')
base.config.use_cache=False
model=get_peft_model(base,LoraConfig(r=8,lora_alpha=16,lora_dropout=0.05,target_modules=r'.*language_model\.layers\..*self_attn\.(q_proj|v_proj)',bias='none'))
model.gradient_checkpointing_enable(); model.enable_input_require_grads(); model.train()
params=[p for p in model.parameters() if p.requires_grad]; assert params
opt=torch.optim.AdamW(params,lr=2e-4,weight_decay=0.01)
opt.zero_grad(set_to_none=True); started=time.perf_counter(); log=[]
for n,item in enumerate(rows):
    check(); path=DATA/item['file']; assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
    img=Image.open(path).convert('RGB'); assert img.size==(item['width'],item['height'])
    x1,y1,x2,y2=item['bbox_pixels']; x=round((x1+x2)*500/item['width']); y=round((y1+y2)*500/item['height'])
    messages=[{'role':'user','content':[{'type':'image','image':img},{'type':'text','text':PROMPT.format(description=item['instruction'])}]}]
    prompt=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    images,videos=process_vision_info(messages)
    prefix=processor(text=[prompt],images=images,videos=videos,return_tensors='pt')
    full=processor(text=[prompt+f'({x}, {y})'+processor.tokenizer.eos_token],images=images,videos=videos,return_tensors='pt').to('cuda')
    assert torch.equal(prefix['input_ids'],full['input_ids'][:,:prefix['input_ids'].shape[-1]].cpu())
    labels=full['input_ids'].clone(); labels[:,:prefix['input_ids'].shape[-1]]=-100
    assert (labels!=-100).sum()>0
    loss=model(**full,labels=labels).loss
    if not torch.isfinite(loss): raise RuntimeError('nonfinite loss')
    (loss/4).backward()
    if (n+1)%4==0:
        torch.nn.utils.clip_grad_norm_(params,1.0); opt.step(); opt.zero_grad(set_to_none=True)
    log.append({'row_idx':item['row_idx'],'loss':float(loss.detach()),'step':(n+1)//4})
    with (OUT/'loss.jsonl').open('a',encoding='utf8') as f:
        f.write(json.dumps(log[-1])+'\n'); f.flush()
    if (n+1)%32==0: model.save_pretrained(OUT/'latest-adapter')
    print(json.dumps(log[-1]),flush=True)
model.save_pretrained(OUT/'adapter')
(OUT/'metadata.json').write_text(json.dumps({'dataset':'PrentisAI/GroundGUI_RL','dataset_revision':'6ecc90819dca07dc92d69d6ec8df4a951c9715d2','train_samples':len(rows),'seed':seed,'batch':1,'gradient_accumulation':4,'steps':len(rows)//4,'lr':2e-4,'lora_r':8,'lora_alpha':16,'loss':'causal LM cross entropy on assistant coordinate tokens only','elapsed_seconds':time.perf_counter()-started,'device':torch.cuda.get_device_name(0),'torch':torch.__version__,'transformers':version('transformers'),'peft':version('peft'),'dtype':'bfloat16','peak_allocated_bytes':torch.cuda.max_memory_allocated()},indent=2),encoding='utf8')
