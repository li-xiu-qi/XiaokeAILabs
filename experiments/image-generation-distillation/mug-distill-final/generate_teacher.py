"""Generate SDXL samples with model CPU offload, retaining reproducibility metadata."""
import argparse
import json
import time
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--model',required=True); p.add_argument('--prompts',type=Path,required=True)
p.add_argument('--out',type=Path,required=True); p.add_argument('--limit',type=int)
p.add_argument('--attention-slicing',action='store_true')
args=p.parse_args()
if args.out.exists(): raise SystemExit('refusing overwrite')
import torch, diffusers
from diffusers import DiffusionPipeline
rows=[json.loads(x) for x in args.prompts.read_text(encoding='utf-8').splitlines()]
if args.limit is not None: rows=rows[:args.limit]
args.out.mkdir(parents=True)
pipe=DiffusionPipeline.from_pretrained(args.model,torch_dtype=torch.float16,use_safetensors=True)
pipe.enable_model_cpu_offload()
if args.attention_slicing: pipe.enable_attention_slicing()
manifest={'model':args.model,'torch':torch.__version__,'diffusers':diffusers.__version__,
          'scheduler':type(pipe.scheduler).__name__,'scheduler_config':dict(pipe.scheduler.config),
          'device':torch.cuda.get_device_name(),'steps':30,'guidance':7.5,'size':1024,
          'cpu_offload':'model','attention_slicing':args.attention_slicing}
(args.out/'run_manifest.json').write_text(json.dumps(manifest,indent=2,default=str),encoding='utf-8')
records=[]
for row in rows:
    torch.cuda.synchronize(); started=time.perf_counter()
    image=pipe(row['prompt'],height=1024,width=1024,num_inference_steps=30,guidance_scale=7.5,
               generator=torch.Generator(device='cuda').manual_seed(row['seed'])).images[0]
    torch.cuda.synchronize(); elapsed=time.perf_counter()-started
    filename=row['id']+'.png'; image.save(args.out/filename)
    records.append({**row,'file_name':filename,'text':row['prompt'],'seconds':elapsed,'size':1024})
    print(filename,round(elapsed,2),flush=True)
(args.out/'metadata.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
