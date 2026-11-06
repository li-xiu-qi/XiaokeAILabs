"""Exploratory local benchmark; run each model in a separate process."""
import argparse
import json
import os
import platform
import signal
import threading
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    'Explain in plain English why a database needs an index. Give one benefit and one tradeoff.',
    'Write a short Python function that returns the largest value in a list, or None for an empty list.',
    'Summarize three practical differences between training and inference of a language model.',
]

def guard():
    with open('/proc/meminfo') as f:
        available = next(int(line.split()[1]) for line in f if line.startswith('MemAvailable:'))
    if available < 50 * 1024 * 1024:
        raise RuntimeError('MemAvailable fell below 50 GiB')

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True)
parser.add_argument('--kind', choices=['diffusion', 'autoregressive'], required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--revision', required=True)
parser.add_argument('--repeats', type=int, default=3)
parser.add_argument('--max-new-tokens', type=int, default=64)
parser.add_argument('--warmups', type=int, default=1)
args = parser.parse_args()
assert args.repeats >= 1 and args.warmups >= 1 and args.max_new_tokens > 0
if args.kind == 'diffusion':
    assert args.max_new_tokens % 32 == 0
guard()
def monitor_memory():
    while True:
        try:
            guard()
        except RuntimeError as exc:
            print('memory guard:', exc, flush=True)
            os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(2)
threading.Thread(target=monitor_memory, daemon=True).start()
assert not Path(args.output).exists(), 'refusing to overwrite benchmark output'
torch.cuda.set_per_process_memory_fraction(0.20)
torch.manual_seed(20261107)
tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=args.kind == 'diffusion')
model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
        trust_remote_code=args.kind == 'diffusion', dtype=torch.bfloat16).to('cuda').eval()
forward = model.forward
counter = {'calls': 0}
def checked_forward(*a, **kw):
    counter['calls'] += 1
    if counter['calls'] % 10 == 1:
        guard()
    return forward(*a, **kw)
model.forward = checked_forward

def generate(prompt):
    guard()
    messages = [{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': prompt}]
    chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(chat, return_tensors='pt', add_special_tokens=False).to('cuda')
    options = {'max_new_tokens': args.max_new_tokens}
    if args.kind == 'diffusion':
        options.update(tokenizer=tokenizer, block_size=32, small_block_size=8,
                       threshold=0.9, temperature=0, use_block_cache=False)
    else:
        options.update(do_sample=False, use_cache=True, pad_token_id=tokenizer.eos_token_id,
                       attention_mask=encoded['attention_mask'])
    count_before = counter['calls']
    torch.cuda.synchronize(); start = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(encoded['input_ids'], **options)
    torch.cuda.synchronize(); elapsed = time.perf_counter() - start
    generated = out[0, encoded['input_ids'].shape[1]:].tolist()
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return {'input_tokens': encoded['input_ids'].shape[1], 'returned_token_ids': len(generated),
            'text_tokens_reencoded': len(tokenizer.encode(text, add_special_tokens=False)),
            'output_utf8_bytes': len(text.encode('utf8')), 'seconds': elapsed,
            'forward_calls': counter['calls'] - count_before, 'text': text,
            'returned_ids': generated,
            'eos_count': generated.count(tokenizer.eos_token_id),
            'contains_mask_token': 151665 in generated if args.kind == 'diffusion' else None}

warmup = [generate(PROMPTS[0]) for _ in range(args.warmups)]
rows = []
for prompt in PROMPTS:
    for repetition in range(args.repeats):
        row = generate(prompt)
        rows.append({'prompt': prompt, 'repetition': repetition, **row})
result = {'kind': args.kind, 'model': args.model, 'revision': args.revision, 'dtype': 'bfloat16',
          'device': torch.cuda.get_device_name(0), 'torch': torch.__version__,
          'transformers': transformers.__version__, 'platform': platform.platform(),
          'seed': 20261107, 'max_new_tokens': args.max_new_tokens, 'batch_size': 1,
          'diffusion_settings': {'block_size':32, 'small_block_size':8, 'threshold':0.9,
                                 'temperature':0, 'use_block_cache':False} if args.kind == 'diffusion' else None,
          'warmup': warmup, 'runs': rows,
          'timing': 'synchronized generate() wall clock; excludes tokenization and text decoding'}
Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf8')
print(json.dumps({'kind': args.kind, 'run_count': len(rows), 'output': args.output}))
