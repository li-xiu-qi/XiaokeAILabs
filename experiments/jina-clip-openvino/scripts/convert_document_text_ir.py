#!/usr/bin/env python3
"""Convert the no-task-LoRA document encoder and validate against native output."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preprocessor', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True, help='New output directory')
    parser.add_argument('--device', default='CPU', help='OpenVINO validation device')
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    if args.out.exists():
        parser.error('--out must not already exist')
    import numpy as np
    import openvino as ov
    import torch
    from transformers import AutoModel, AutoTokenizer
    from run_openvino_smoke import (MODEL_ID, MODEL_REVISION, CODE_REVISIONS,
        TEXTS, cosine_diff, native_matches, set_static_input_shape)

    tokenizer = AutoTokenizer.from_pretrained(str(args.preprocessor),
        trust_remote_code=True, local_files_only=args.offline)
    inputs = tokenizer(TEXTS, return_tensors='pt', padding='max_length',
        max_length=512, truncation=True)['input_ids']
    model = AutoModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
        code_revision=CODE_REVISIONS['jinaai/jina-clip-implementation'],
        trust_remote_code=True, cache_dir=str(args.cache_dir),
        local_files_only=args.offline, dtype=torch.float32).eval()
    text_model = model.text_model
    if not hasattr(text_model, 'default_loraid'):
        raise RuntimeError('Model does not expose the expected task-LoRA setting')
    original_loraid = text_model.default_loraid
    text_model._default_loraid = None
    if text_model.default_loraid is not None:
        raise RuntimeError('Could not disable the default task-LoRA adapter')
    with torch.no_grad():
        reference = text_model(inputs).cpu().numpy()
    converted = ov.convert_model(text_model, example_input=inputs)
    set_static_input_shape(converted, inputs)
    args.out.mkdir(parents=True, exist_ok=False)
    ir = args.out / 'jina-clip-text_v2_fp16.xml'
    ov.save_model(converted, ir, compress_to_fp16=True)
    compiled = ov.Core().compile_model(str(ir), args.device)
    results = compiled(inputs)
    if len(results) != 1:
        raise RuntimeError('Expected one model output')
    actual = np.asarray(list(results.values())[0])
    if actual.shape != reference.shape:
        raise RuntimeError(f'Output shape mismatch: {actual.shape} != {reference.shape}')
    diff = cosine_diff(reference, actual)
    ok = bool(np.isfinite(actual).all() and native_matches(diff))
    record = {'model': MODEL_ID, 'revision': MODEL_REVISION,
        'code_revision': CODE_REVISIONS['jinaai/jina-clip-implementation'],
        'original_default_loraid': original_loraid, 'document_loraid': None,
        'device': args.device, 'input_shape': list(inputs.shape),
        'comparison': diff, 'ok': ok}
    (args.out / 'validation.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps(record, indent=2))
    if not ok:
        raise SystemExit('Document IR did not pass native validation; do not use it')


if __name__ == '__main__':
    main()
