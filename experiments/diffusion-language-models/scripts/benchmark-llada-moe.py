"""LLaDA-MoE fixed-step baseline, adapted from the model-card reference."""
import argparse, json, os, signal, threading, time
from pathlib import Path
import torch
import transformers
from transformers import AutoModel, AutoTokenizer

PROMPTS = [
    'Explain in plain English why a database needs an index. Give one benefit and one tradeoff.',
    'Write a short Python function that returns the largest value in a list, or None for an empty list.',
    'Summarize three practical differences between training and inference of a language model.',
]
p = argparse.ArgumentParser()
p.add_argument('--model', required=True)
p.add_argument('--output', required=True)
p.add_argument('--revision', required=True)
p.add_argument('--repeats', type=int, default=3)
p.add_argument('--warmups', type=int, default=2)
p.add_argument('--steps', type=int, default=128)
p.add_argument('--gen-length', type=int, default=128)
p.add_argument('--block-length', type=int, default=32)
args = p.parse_args()
assert args.repeats > 0 and args.warmups > 0
assert args.gen_length % args.block_length == 0
assert args.steps % (args.gen_length // args.block_length) == 0
assert not Path(args.output).exists()
def guard():
    with open('/proc/meminfo') as f:
        avail = next(int(x.split()[1]) for x in f if x.startswith('MemAvailable:'))
    if avail < 50 * 1024 * 1024:
        raise RuntimeError('MemAvailable below 50 GiB')
def monitor():
    while True:
        try: guard()
        except Exception as e:
            print('memory guard:', e, flush=True)
            os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(2)
guard()
threading.Thread(target=monitor, daemon=True).start()
torch.cuda.set_per_process_memory_fraction(.2)
torch.manual_seed(20261107)
tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=True)
model = AutoModel.from_pretrained(args.model, local_files_only=True, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()
mask_id = 156895

@torch.inference_mode()
def run(prompt):
    messages = [{'role':'system','content':'You are a helpful assistant.'},{'role':'user','content':prompt}]
    rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inp = tok(rendered, add_special_tokens=False, return_tensors='pt').input_ids.cuda()
    x = torch.full((1,inp.shape[1] + args.gen_length),mask_id,device='cuda',dtype=torch.long)
    x[:, :inp.shape[1]] = inp
    blocks = args.gen_length // args.block_length
    steps_per_block = args.steps // blocks
    calls = 0
    torch.cuda.synchronize(); start=time.perf_counter()
    for block in range(blocks):
        start_ix = inp.shape[1] + block*args.block_length
        end_ix = start_ix + args.block_length
        base, rem = divmod(args.block_length, steps_per_block)
        for step in range(steps_per_block):
            logits = model(x).logits; calls += 1
            chosen = logits.argmax(dim=-1)
            confidence = torch.softmax(logits.float(),dim=-1).gather(-1,chosen.unsqueeze(-1)).squeeze(-1)
            eligible = x.eq(mask_id)
            eligible[:,end_ix:] = False
            confidence = torch.where(eligible,confidence,-torch.inf)
            count = base + (step < rem)
            if count:
                idx = confidence.topk(count,dim=-1).indices
                x.scatter_(1,idx,chosen.gather(1,idx))
    torch.cuda.synchronize(); elapsed=time.perf_counter()-start
    ids=x[0,inp.shape[1]:].tolist()
    eos = tok.eos_token_id
    end = ids.index(eos) + 1 if eos is not None and eos in ids else len(ids)
    visible_ids=ids[:end]
    text=tok.decode(visible_ids,skip_special_tokens=True)
    return {'input_tokens':inp.shape[1], 'returned_ids':ids, 'returned_token_ids':len(ids),
            'first_eos_inclusive_length':end,'eos_count':ids.count(eos) if eos is not None else None,
            'contains_mask_token':mask_id in ids,'text_tokens_reencoded':len(tok.encode(text,add_special_tokens=False)),
            'text':text,'forward_calls':calls,'seconds':elapsed,
            'output_utf8_bytes':len(text.encode('utf8'))}

warmup=[run(PROMPTS[0]) for _ in range(args.warmups)]
rows=[{'prompt':prompt,'repetition':i,**run(prompt)} for prompt in PROMPTS for i in range(args.repeats)]
out={'model':args.model,'revision':args.revision,'device':torch.cuda.get_device_name(0),
     'torch':torch.__version__,'transformers':transformers.__version__,'dtype':'bfloat16',
     'algorithm':'model-card low-confidence fixed-step baseline; no KV cache',
     'steps':args.steps,'gen_length':args.gen_length,'block_length':args.block_length,
     'seed':20261107,'warmup':warmup,'runs':rows,
     'timing':'synchronized fixed-step generation; excludes tokenization and text decoding'}
Path(args.output).write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({'run_count':len(rows),'output':args.output}))
