import argparse
from pathlib import Path
from datasets import load_dataset
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--revision',default='49bf8f13e13c41dd8cdb0cae5314e31c1da1e0d6');args=p.parse_args()
if Path(args.output).exists(): p.error('output already exists; choose a new path')
data=load_dataset('vidore/docvqa_test_subsampled',revision=args.revision,split='test')
assert len(data)==500
data.save_to_disk(args.output)
print(f'saved {len(data)} rows to {args.output}')
