"""Audit the author's prefix-masking path with a local tokenizer; CPU only."""
import argparse
import json
from pathlib import Path
from transformers import AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument('--model', required=True)
p.add_argument('--data', type=Path, required=True)
p.add_argument('--max-length', type=int, default=128)
a = p.parse_args()
t = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
if t.padding_side != 'right' or not t.chat_template:
    raise ValueError('Requires a right-padding tokenizer with chat template')
count = 0
for line in a.data.read_text(encoding='utf-8-sig').splitlines():
    messages = json.loads(line)['messages']
    full = t.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    encoded = t(full, truncation=True, max_length=a.max_length, padding='max_length', add_special_tokens=False)
    prefix = t.apply_chat_template(messages[:-1], add_generation_prompt=True, return_dict=False)
    ids, mask = encoded['input_ids'], encoded['attention_mask']
    active = sum(mask)
    if ids[:len(prefix)] != prefix or len(prefix) >= active:
        raise ValueError(f'prefix mismatch or truncated target at example {count}')
    target = t.decode(ids[len(prefix):active], skip_special_tokens=True).strip()
    if target != messages[-1]['content']:
        raise ValueError(f'incorrect decoded target at example {count}: {target!r}')
    count += 1
print(json.dumps({'model': a.model, 'examples': count, 'padding_side': t.padding_side,
                  'max_length': a.max_length, 'prefix_and_target_ok': True,
                  'user_tail': t.convert_ids_to_tokens(t.apply_chat_template(messages[1:2], tokenize=True, return_dict=False)[-6:]),
                  'assistant_tail': t.convert_ids_to_tokens(t.apply_chat_template(messages[2:3], tokenize=True, return_dict=False)[-6:])}, ensure_ascii=False))
