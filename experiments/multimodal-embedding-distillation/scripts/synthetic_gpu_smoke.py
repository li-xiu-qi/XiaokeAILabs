"""Exercise the student training pipeline using synthetic data and random weights.

This is a pipeline check, not a model-quality or historical reproduction test.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import CLIPConfig, CLIPImageProcessor, CLIPModel, CLIPProcessor, PreTrainedTokenizerFast


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True, help='A new output directory outside the repository')
    args = parser.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit('A CUDA GPU with BF16 support is required')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    images = out / 'images'
    images.mkdir()
    work = out / 'work'
    work.mkdir()
    checkpoint = out / 'random-clip'
    captions = [f'generated sample number {i}' for i in range(44)]
    tokenizer = Tokenizer(models.BPE(unk_token='<unk>'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.train_from_iterator(captions, trainers.BpeTrainer(
        vocab_size=128, special_tokens=['<pad>', '<unk>', '<bos>', '<eos>']))
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token='<pad>',
                                   unk_token='<unk>', bos_token='<bos>', eos_token='<eos>')
    processor = CLIPProcessor(image_processor=CLIPImageProcessor(
        size={'shortest_edge': 32}, crop_size={'height': 32, 'width': 32}), tokenizer=fast)
    torch.manual_seed(123)
    config = CLIPConfig(
        text_config={'vocab_size': len(fast), 'hidden_size': 32, 'intermediate_size': 64,
                     'num_hidden_layers': 1, 'num_attention_heads': 4, 'max_position_embeddings': 77,
                     'pad_token_id': fast.pad_token_id, 'bos_token_id': fast.bos_token_id,
                     'eos_token_id': fast.eos_token_id},
        vision_config={'hidden_size': 32, 'intermediate_size': 64, 'num_hidden_layers': 1,
                       'num_attention_heads': 4, 'image_size': 32, 'patch_size': 8},
        projection_dim=32)
    CLIPModel(config).save_pretrained(checkpoint)
    processor.save_pretrained(checkpoint)

    rng = np.random.default_rng(123)
    samples = {}
    index = 0
    for split, count in (('train', 20), ('val', 16), ('test', 8)):
        samples[split] = []
        for _ in range(count):
            name = f'image-{index:03d}.png'
            Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(images / name)
            samples[split].append({'filename': name, 'caption': captions[index], 'sha256': sha(images / name)})
            index += 1
    samples_path = work / 'samples-private.json'
    samples_path.write_text(json.dumps(samples), encoding='utf8')
    # Perfect synthetic teacher vectors exercise the predeclared gate deterministically.
    vectors = {f'{split}_{side}': np.eye(count, 32, dtype=np.float32)
               for split, count in (('train', 20), ('val', 16)) for side in ('text', 'image')}
    np.savez_compressed(work / 'teacher-private.npz', **vectors)
    teacher = 'synthetic-identity-vectors'
    plan = {'teacher': teacher, 'student': str(checkpoint), 'minimum_gap': 0.05,
            'samples_sha256': sha(samples_path)}
    (work / 'teacher-plan.json').write_text(json.dumps(plan), encoding='utf8')
    manifest = {'teacher': teacher, 'samples_sha256': sha(samples_path),
                'vectors_sha256': sha(work / 'teacher-private.npz')}
    (work / 'teacher-manifest.json').write_text(json.dumps(manifest), encoding='utf8')
    subprocess.run([sys.executable, str(Path(__file__).with_name('train.py')), '--workdir', str(work),
                    '--images', str(images), '--student', str(checkpoint), '--min-teacher-gap', '0.05'],
                   check=True)
    results = json.loads((work / 'student-results.json').read_text(encoding='utf8'))
    assert (work / 'training-complete.json').is_file()
    assert all(results[name]['losses'] and all(np.isfinite(results[name]['losses']))
               for name in ('supervised', 'distilled'))
    assert all('test' in results[name] for name in ('baseline', 'supervised', 'distilled'))
    print('Synthetic student pipeline completed; no real teacher, dataset or quality result was tested.', flush=True)


if __name__ == '__main__':
    main()
