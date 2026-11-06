"""Create a guarded local copy of the pinned author trainer with a masking fix."""
import argparse
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--source', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
s = a.source.read_text(encoding='utf-8')
old = '''            prompt = tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True)

        labels = [label if unmasked else -100 for label, unmasked in zip(input_ids, attention_mask)]'''
new = '''            prompt = tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True, return_dict=False)
        if input_ids[:len(prompt)] != prompt:
            raise ValueError("chat prompt token prefix mismatch")
        if len(prompt) >= sum(attention_mask):
            raise ValueError("assistant response truncated or missing")

        labels = [label if unmasked else -100 for label, unmasked in zip(input_ids, attention_mask)]'''
if s.count(old) != 1:
    raise ValueError('author source differs; refusing to patch')
s = s.replace(old, new)
s = s.replace('        overwrite_output_dir=True,', '')
s = s.replace('        logging_dir=f"{args.output_dir}/logs",', '')
s = s.replace('    # Start training\n', '    # Both paths use microbatch-mean loss; Trainer divides by accumulation steps.\n    trainer.model_accepts_loss_kwargs = False\n\n    # Start training\n')
s = s.replace('            tokenizer=tokenizer,', '            processing_class=tokenizer,')
needle = '    local_rank = int(os.environ.get(' + chr(39) + 'LOCAL_RANK' + chr(39) + ', 0))'
if s.count(needle) != 1:
    raise ValueError('cannot locate initialization seed position')
s = s.replace(needle, '    from transformers import set_seed' + chr(10) + '    set_seed(args.finetune_seed)' + chr(10) + needle)
with a.output.open('x', encoding='utf-8', newline='\n') as f:
    f.write(s)
