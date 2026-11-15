"""Exercise cross-encoder scoring and student updates on synthetic data."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True,
                        help='New directory for synthetic models and results')
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit('Output directory must be empty')
    if not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable')
    args.out.mkdir(parents=True, exist_ok=True)
    model_dir = args.out / 'initial-model'
    model_dir.mkdir()
    (model_dir / 'vocab.txt').write_text(
        '[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nquery\npositive\nnegative\n', encoding='utf-8')
    tokenizer = BertTokenizerFast(vocab_file=str(model_dir / 'vocab.txt'))
    tokenizer.save_pretrained(model_dir)
    torch.manual_seed(7)
    config = BertConfig(vocab_size=len(tokenizer), hidden_size=32, num_hidden_layers=1,
                        num_attention_heads=2, intermediate_size=64, num_labels=1)
    BertForSequenceClassification(config).save_pretrained(model_dir)

    data = args.out / 'synthetic-data'
    (data / 'scifact').mkdir(parents=True)
    (data / 'prepared').mkdir()
    queries = [{'_id': f'{split}{n}', 'text': 'query'}
               for split in ('train', 'dev', 'test') for n in (1, 2)]
    docs = [{'_id': 'p', 'text': 'positive'}, {'_id': 'n', 'text': 'negative'}]
    for name, rows in (('queries', queries), ('corpus', docs)):
        (data / 'scifact' / f'{name}.jsonl').write_text(
            ''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
    pools = {split: {f'{split}{n}': {'relevant': ['p'], 'unjudged': ['n'],
                                     'pool': [['p', 1], ['n', 0]]}
                     for n in (1, 2)} for split in ('train', 'dev', 'test')}
    (data / 'prepared' / 'pools.json').write_text(json.dumps(pools), encoding='utf-8')
    cli_out = args.out / 'cli-smoke'
    subprocess.run([sys.executable, str(Path(run.__file__)), '--data-root', str(data),
                    '--teacher', str(model_dir), '--student', str(model_dir),
                    '--out', str(cli_out), '--device', 'cuda', '--smoke'], check=True)
    stopped = json.loads((cli_out / 'stopped.json').read_text(encoding='utf-8'))
    if 'smoke data cannot establish the gate' not in stopped.get('reason', ''):
        raise SystemExit('CLI smoke did not stop at the teacher advantage gate')

    run.ROOT = args.out
    run.STUDENT = model_dir
    run.DEVICE = 'cuda'
    run.STEPS = 2
    run.queries = {'q1': 'query', 'q2': 'query'}
    run.docs = {'p': 'positive', 'n': 'negative'}
    run.pools = {'train': {q: {'relevant': ['p'], 'unjudged': ['n']}
                           for q in run.queries}}
    pairs = [(q, d) for q in run.queries for d in ('p', 'n')]
    tokenizer, teacher = run.teacher_model(model_dir, 'cross-encoder')
    assert next(teacher.parameters()).is_cuda
    scores, _ = run.score(teacher, tokenizer, pairs)
    del teacher, tokenizer
    torch.cuda.empty_cache()
    initial = BertForSequenceClassification.from_pretrained(model_dir).state_dict()
    for mode in ('supervised', 'distilled'):
        tokenizer, student = run.train(mode, scores)
        assert next(student.parameters()).is_cuda
        changed = any(not torch.equal(initial[name], value.detach().cpu())
                      for name, value in student.state_dict().items())
        if not changed:
            raise SystemExit(f'{mode}: parameters did not update')
        del student, tokenizer
        torch.cuda.empty_cache()
    (args.out / 'summary.json').write_text(json.dumps({
        'device': torch.cuda.get_device_name(), 'torch': torch.__version__,
        'scored_pairs': len(scores), 'training_steps_per_mode': run.STEPS,
        'modes': ['supervised', 'distilled'], 'cli_smoke_stopped_at_gate': True,
        'synthetic': True,
    }, indent=2) + '\n', encoding='utf-8')
    print('Synthetic GPU scoring and supervised/distilled updates completed')


if __name__ == '__main__':
    main()
