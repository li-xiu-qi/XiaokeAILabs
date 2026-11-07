"""Record and isolate a matched video generation evaluation."""
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def start_run(output, base, lora, prompts, negative_prompt, settings):
    lora_path = Path(lora)
    if not lora_path.is_file():
        raise ValueError('Provide a local LoRA weights file so its SHA-256 can be recorded')
    lora_hash = sha256(lora_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'status': 'incomplete', 'base': str(base), 'lora': str(lora_path),
                'lora_sha256': lora_hash, 'prompts': prompts,
                'negative_prompt': negative_prompt, **settings}
    (output / 'run.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return manifest


def finish_run(output, manifest):
    output = Path(output)
    if sha256(manifest['lora']) != manifest['lora_sha256']:
        raise ValueError('LoRA weights changed during evaluation')
    names = [f'p{i}_{group}{suffix}' for i in range(len(manifest['prompts']))
             for group in ('before', 'after') for suffix in ('.mp4', '_frames.png')]
    manifest['output_sha256'] = {name: sha256(output / name) for name in names}
    manifest['status'] = 'complete'
    (output / 'run.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
