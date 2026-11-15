"""Compare hard and soft teacher signals using identical pre-scored pairs."""
import argparse
import json
import math
import random
import os
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup
import run as base


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--student', type=Path, required=True)
    p.add_argument('--teacher-scores', type=Path, required=True,
                   help='Scores for development candidates and the fixed training sample schedule')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--min-teacher-gap', type=float, required=True)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--min-available-gib', type=float, default=0,
                   help='On Linux, stop if available system memory falls below this threshold')
    args = p.parse_args()
    if args.min_available_gib < 0:
        raise SystemExit('Memory threshold must be nonnegative')
    if args.min_available_gib:
        def guard():
            while True:
                try:
                    with open('/proc/meminfo') as f:
                        available = next(int(x.split()[1]) for x in f if x.startswith('MemAvailable:'))
                    if available < args.min_available_gib * 1024 * 1024:
                        print('Available memory below threshold; stopping', flush=True)
                        os._exit(3)
                except (OSError, StopIteration, ValueError):
                    print('Cannot read available memory; stopping', flush=True)
                    os._exit(3)
                time.sleep(2)
        threading.Thread(target=guard, daemon=True).start()
    if not math.isfinite(args.min_teacher_gap) or args.min_teacher_gap < .05:
        raise SystemExit('A prespecified absolute development NDCG@10 gap of at least 0.05 is required')
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit('Output directory must be empty; refusing to overwrite')
    args.out.mkdir(parents=True, exist_ok=True)
    base.ROOT = args.out
    base.STUDENT = args.student
    base.DEVICE = args.device
    torch.set_num_threads(2)
    base.queries = {x['_id']: x['text'] for x in base.read_jsonl(args.data_root/'scifact/queries.jsonl')}
    base.docs = {x['_id']: (x.get('title', '')+' '+x['text']).strip() for x in base.read_jsonl(args.data_root/'scifact/corpus.jsonl')}
    base.pools = json.loads((args.data_root/'prepared/pools.json').read_text())
    if set(base.pools['train']) & (set(base.pools['dev']) | set(base.pools['test'])) or set(base.pools['dev']) & set(base.pools['test']):
        raise SystemExit('Query splits overlap')
    teacher = json.loads(args.teacher_scores.read_text())
    schedule = []
    rng = random.Random(base.SEED)
    ids = list(base.pools['train'])
    while len(schedule) < base.STEPS:
        rng.shuffle(ids)
        for start in range(0, len(ids), base.BATCH):
            batch = []
            for q in ids[start:start+base.BATCH]:
                row = base.pools['train'][q]
                batch.append((q, str(rng.choice(row['relevant'])), str(rng.choice(row['unjudged']))))
            schedule.append(batch)
            if len(schedule) >= base.STEPS:
                break
    pairs = [(q, p, n) for batch in schedule for q, p, n in batch]
    devpairs = list(dict.fromkeys((q, str(d)) for q, row in base.pools['dev'].items() for d, _ in row['pool']))
    required = {f'{q}:{d}' for q, p, n in pairs for d in (p, n)} | {f'{q}:{d}' for q, d in devpairs}
    if not required <= set(teacher) or not all(math.isfinite(teacher[k]) for k in required):
        raise SystemExit('Teacher scores are missing or nonfinite')
    # Compare the teacher to the untrained student, on the same development candidates.
    t, m = base.model(base.STUDENT)
    student_scores, _ = base.score(m, t, devpairs)
    teacher_dev = base.metrics(base.pools['dev'], teacher)
    student_dev = base.metrics(base.pools['dev'], student_scores)
    del m, t
    if args.device == 'cuda':
        torch.cuda.empty_cache()
    a = {x['qid']: x['ndcg10'] for x in teacher_dev['per_query']}
    b = {x['qid']: x['ndcg10'] for x in student_dev['per_query']}
    if a.keys() != b.keys() or len(a) < 2:
        raise SystemExit('Development queries do not match')
    delta = np.array([a[q]-b[q] for q in sorted(a)])
    boot = np.random.default_rng(base.SEED).choice(delta, (10000, len(delta)), replace=True).mean(axis=1)
    gate = {'teacher_ndcg10': teacher_dev['ndcg10'], 'student_ndcg10': student_dev['ndcg10'],
            'difference': float(delta.mean()), 'required_gap': args.min_teacher_gap,
            'paired_bootstrap_95ci': np.quantile(boot, [.025, .975]).tolist()}
    base.write('teacher-gate.json', gate)
    if gate['difference'] < args.min_teacher_gap or gate['paired_bootstrap_95ci'][0] <= 0:
        raise SystemExit('Teacher advantage gate failed; training stopped')
    diff = np.array([teacher[f'{q}:{p}']-teacher[f'{q}:{n}'] for q, p, n in pairs])
    unique = list(dict.fromkeys(pairs))
    udiff = np.array([teacher[f'{q}:{p}']-teacher[f'{q}:{n}'] for q, p, n in unique])
    groups = ['human', 'teacher_hard', 'teacher_soft', 'human_hard', 'human_soft']
    seeds = [20261107, 20261108, 20261109]
    plan = {'groups': groups, 'seeds': seeds, 'fixed_sampling_seed': base.SEED,
            'steps': base.STEPS, 'batch_queries': base.BATCH, 'temperature': base.TEMP,
            'mix_weight': 0.5, 'student_max_length': base.MAXLEN,
            'pools_sha256': base.sha(args.data_root/'prepared/pools.json'),
            'teacher_scores_sha256': base.sha(args.teacher_scores),
            'label_comparison': {'sampled_pairs': len(diff), 'same': int((diff>0).sum()),
              'reverse': int((diff<0).sum()), 'tie': int((diff==0).sum()),
              'margin_quantiles': np.quantile(diff, [0, .1, .25, .5, .75, .9, 1]).tolist(),
              'unique_pairs': len(unique), 'unique_same': int((udiff>0).sum()),
              'unique_reverse': int((udiff<0).sum()), 'unique_tie': int((udiff==0).sum())},
            'notes': 'Unjudged documents are not guaranteed negative; seed changes training randomness, not pair sampling. Teacher hard ties choose the first candidate.'}
    base.write('plan.json', plan)
    for seed in seeds:
        for mode in groups:
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            if args.device == 'cuda': torch.cuda.manual_seed_all(seed)
            t, m = base.model(base.STUDENT); m.train()
            opt = torch.optim.AdamW(m.parameters(), lr=2e-5, weight_decay=.01)
            scheduler = get_linear_schedule_with_warmup(opt, num_warmup_steps=18, num_training_steps=base.STEPS)
            log = []
            for step, batch in enumerate(schedule, 1):
                x = [(base.queries[q], base.docs[d]) for q, pos, neg in batch for d in (pos, neg)]
                s = m(**base.tok(t, x)).logits.reshape(-1, 2)
                target = torch.tensor([[teacher[f'{q}:{pos}'], teacher[f'{q}:{neg}']] for q, pos, neg in batch], device=base.DEVICE)
                human = F.cross_entropy(s, torch.zeros(len(batch), device=base.DEVICE, dtype=torch.long))
                hard = F.cross_entropy(s, target.argmax(dim=-1))
                soft = F.kl_div(F.log_softmax(s/base.TEMP, dim=-1), F.softmax(target/base.TEMP, dim=-1), reduction='batchmean')*base.TEMP**2
                loss = {'human': human, 'teacher_hard': hard, 'teacher_soft': soft,
                        'human_hard': .5*(human+hard), 'human_soft': .5*(human+soft)}[mode]
                if not torch.isfinite(loss): raise SystemExit('Nonfinite loss')
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.)
                opt.step(); scheduler.step()
                log.append({'step': step, 'loss': float(loss.detach().cpu())})
            out = args.out/f'{mode}-{seed}'; out.mkdir()
            m.save_pretrained(out); t.save_pretrained(out)
            base.write(f'{mode}-{seed}-training.json', log)
            print('TRAINED', mode, seed, flush=True)
            del m, t
            if args.device == 'cuda': torch.cuda.empty_cache()
    evalpairs = list(dict.fromkeys((q, str(d)) for split in ('dev', 'test') for q, row in base.pools[split].items() for d, _ in row['pool']))
    for seed in seeds:
        for mode in groups:
            t, m = base.model(args.out/f'{mode}-{seed}')
            scores, seconds = base.score(m, t, evalpairs)
            base.write(f'{mode}-{seed}-eval.json', {'seconds': seconds, **{s: base.metrics(base.pools[s], scores) for s in ('dev', 'test')}})
            base.write(f'{mode}-{seed}-scores.json', scores)
            print('EVALUATED', mode, seed, flush=True)
            del m, t
            if args.device == 'cuda': torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
