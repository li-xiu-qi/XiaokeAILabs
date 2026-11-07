"""Record and isolate a matched video generation evaluation."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def base_snapshot(base):
    """Hash the local model files, including configs, tokenizers and weights."""
    root = Path(base).expanduser().resolve()
    if not root.is_dir():
        raise ValueError('Provide a local base model directory for reproducible evaluation')
    files = {}
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root)
        if any(part in ('.git', '.cache') for part in relative.parts):
            continue
        if path.is_symlink() and path.is_dir():
            raise ValueError(f'Symlinked model directory cannot be fingerprinted: {path}')
        if path.is_file():
            files[relative.as_posix()] = sha256(path)
    if not files:
        raise ValueError('Base model directory contains no files')
    encoded = json.dumps(files, sort_keys=True, ensure_ascii=True, separators=(',', ':')).encode()
    return {'sha256': hashlib.sha256(encoded).hexdigest(), 'files': files}


def dependency_versions():
    versions = {'python': platform.python_version()}
    for package in ('torch', 'diffusers', 'transformers', 'peft', 'accelerate', 'av', 'numpy', 'Pillow'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def start_run(output, base, lora, prompts, negative_prompt, settings):
    lora_path = Path(lora)
    if not lora_path.is_file():
        raise ValueError('Provide a local LoRA weights file so its SHA-256 can be recorded')
    lora_hash = sha256(lora_path)
    base_hashes = base_snapshot(base)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'status': 'incomplete', 'base': str(base), 'base_snapshot': base_hashes,
                'dependency_versions': dependency_versions(), 'lora': str(lora_path),
                'lora_sha256': lora_hash, 'prompts': prompts,
                'negative_prompt': negative_prompt, **settings}
    (output / 'run.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return manifest


def finish_run(output, manifest):
    output = Path(output)
    if base_snapshot(manifest['base']) != manifest['base_snapshot']:
        raise ValueError('Base model files changed during evaluation')
    if sha256(manifest['lora']) != manifest['lora_sha256']:
        raise ValueError('LoRA weights changed during evaluation')
    names = [f'p{i}_{group}{suffix}' for i in range(len(manifest['prompts']))
             for group in ('before', 'after') for suffix in ('.mp4', '_frames.png')]
    manifest['output_sha256'] = {name: sha256(output / name) for name in names}
    manifest['status'] = 'complete'
    (output / 'run.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
