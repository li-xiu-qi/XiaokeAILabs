"""Greedy exact-match evaluation for generated toy arithmetic data."""
import argparse
import json
import re
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument('--model', required=True)
p.add_argument('--data', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
args = p.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True).eval().to(args.device)
records = []
for line in args.data.read_text(encoding='utf-8-sig').splitlines():
    messages = json.loads(line)['messages']
    prompt = tokenizer.apply_chat_template(messages[:2], tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors='pt').to(args.device)
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=12, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    answer = tokenizer.decode(generated[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    expected = messages[2]['content']
    records.append({'input': messages[1]['content'], 'expected': expected, 'generated': answer,
                    'strict_exact_match': answer == expected})
result = {'count': len(records), 'strict_exact_match': sum(x['strict_exact_match'] for x in records) / len(records),
          'decoding': 'greedy, max_new_tokens=12', 'records': records}
args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open('x', encoding='utf-8') as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
print(json.dumps({k: v for k, v in result.items() if k != 'records'}))
