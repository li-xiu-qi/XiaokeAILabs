"""Score fixed candidates locally, refusing unsafe resumes."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def input_fingerprint(row, image_dir, model_identifier, image_hashes):
    """Bind each result to its entire candidate row, image bytes and model path."""
    images = []
    for candidate in row['candidates']:
        path = image_dir / candidate['filename']
        if path not in image_hashes:
            with path.open('rb') as f:
                image_hashes[path] = hashlib.file_digest(f, 'sha256').hexdigest()
        images.append(image_hashes[path])
    payload = {'row': row, 'image_sha256': images, 'model_identifier': model_identifier}
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def validate_resume(source, done, image_dir, model_identifier, image_hashes):
    if len(done) > len(source):
        raise ValueError('Output contains more rows than the current candidate file; use a new output file.')
    for i, result in enumerate(done):
        expected = input_fingerprint(source[i], image_dir, model_identifier, image_hashes)
        candidates = result.get('candidates', [])
        originals = source[i]['candidates']
        scores_valid = (len(candidates) == len(originals) and all(
            {k: v for k, v in candidate.items() if k != 'reranker_score'} == original
            and isinstance(candidate.get('reranker_score'), (int, float))
            and math.isfinite(candidate['reranker_score'])
            for candidate, original in zip(candidates, originals)))
        if (result.get('input_sha256') != expected
                or result.get('target_filename') != source[i]['target_filename']
                or not scores_valid):
            raise ValueError(f'Output row {i + 1} does not match current inputs or lacks a valid fingerprint; use a new output file.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--candidates', type=Path, required=True)
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    source = json.loads(a.candidates.read_text(encoding='utf-8'))
    done = [json.loads(x) for x in a.output.read_text(encoding='utf-8').splitlines()] if a.output.exists() else []
    model_path = Path(a.model).expanduser()
    model_identifier = str(model_path.resolve()) if model_path.exists() else a.model
    image_hashes = {}
    validate_resume(source, done, a.images, model_identifier, image_hashes)

    import torch
    from qwen3_vl_reranker import Qwen3VLReranker
    torch.cuda.set_per_process_memory_fraction(.20)
    model = Qwen3VLReranker(a.model, min_pixels=4096, max_pixels=65536, dtype=torch.bfloat16)
    for row in source[len(done):]:
        fingerprint = input_fingerprint(row, a.images, model_identifier, image_hashes)
        scores = []
        for start in range(0, len(row['candidates']), 2):
            batch = row['candidates'][start:start + 2]
            images = [a.images / c['filename'] for c in batch]
            values = model.process({'query': {'text': row['query']}, 'documents': [{'image': str(p)} for p in images]})
            if len(values) != len(batch) or not all(math.isfinite(s) for s in values):
                raise ValueError('Reranker returned invalid scores')
            scores.extend({**c, 'reranker_score': float(s)} for c, s in zip(batch, values))
        with a.output.open('a', encoding='utf-8') as f:
            f.write(json.dumps({'target_filename': row['target_filename'], 'input_sha256': fingerprint, 'candidates': scores}, ensure_ascii=False) + '\n')
        print('completed query', flush=True)


if __name__ == '__main__':
    main()
