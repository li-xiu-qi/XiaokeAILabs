# Retesting on Synthetic Pages Not Used for Teacher Selection

[中文](README.md)

This evaluation compared three existing models from the parent experiment on new pages that had not been used for training, tuning, or teacher selection. [evaluate.py](evaluate.py) generated the data with seed 20261108: 64 pages, three queries per page, and retrieval from the same set of 64 pages using each query's known paired page. The script verifies that entities, field values, and image SHA-256 hashes do not overlap with the old data.

| Model | Recall@1 | Recall@5 |
|---|---:|---:|
| Original ColSmol | 149/192 = 0.7760 | 186/192 = 0.9688 |
| Supervised fine-tuning | 154/192 = 0.8021 | 187/192 = 0.9740 |
| Supervised fine-tuning + distillation | 151/192 = 0.7865 | 186/192 = 0.9688 |

Compared with the supervised group, distillation moved 2 queries into first place and 5 out, a net loss of 3. This result does not show an improvement from distillation. A single training run, one teacher, and one data generation run are insufficient to establish a general conclusion. The three queries from each page are related and should not be treated as 192 fully independent samples.

The new pages share the earlier page layout, fields, and query templates, but use previously unseen six-digit entities and values. This can test the earlier result on new synthetic examples; it cannot establish generalization to real documents, different layouts, or semantic matching. The teacher and model selection did use the previous test set; this new set was not used in that earlier selection.

The run used an NVIDIA DGX Spark, the same ColSmol base model, initial adapter, and two trained adapters as the main experiment, with BF16 inference and ColPali Engine 0.3.13. Peak PyTorch allocation was 3,021,570,048 bytes. The lowest system available memory observed by a separate monitor was 90,478,160 KiB. These numbers describe one run and are not general hardware requirements.

## Recompute the recorded metrics

[results.json](results.json) stores anonymous per-item ranks. [recompute.py](recompute.py) uses only the Python standard library and requires no GPU, images, or weights. Run this command from the parent experiment directory:

```bash
python independent-evaluation/recompute.py
```

The script checks the 192 ranks and aggregate metrics, then prints the relative changes between groups. It cannot revalidate model scores or image labels.

## Prepare an independent retest

The old source data, old images, and trained adapters are not distributed in this repository. **The model inference in this evaluation cannot be reproduced from this repository alone.** You need the original `synthetic/records.json` and old images in its sibling directory, plus the three adapters and base model used by the parent experiment: the initial ColSmol adapter, the supervised-trained adapter, and the supervised-plus-distillation adapter. Sources, pinned revisions, and historical software versions for the initial model and base model are described in [the parent experiment's environment setup](../README.en.md#environment-setup); the other two adapters must come from the original run's saved outputs. The current distillation entry point in the parent experiment stops immediately and cannot regenerate the missing distillation adapter. Newly generated training artifacts must not be presented as the original models.

The execution environment follows the parent experiment's Linux, CUDA BF16, and 50 GiB minimum available-system-memory guard. `evaluate.py` reads fonts from fixed paths, so install DejaVu Sans and check both files in advance. Rendered pixels depend on the font version, and rankings may vary with software and hardware:

```bash
test -f /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf
test -f /usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf
```

Make sure each `image` entry in `records.json` points to an existing old image relative to the directory containing that file, for example `synthetic/train/page-000.png`. The script checks that all referenced images exist and computes their SHA-256 hashes against the new images; it errors if images are missing or the records are empty. It also checks entity and field-value overlap based on `records.json`.

Activate the environment prepared for the parent experiment and run from the experiment root. Replace all six paths; `--out` must name a new directory that does not already exist:

```bash
python independent-evaluation/evaluate.py \
  --source /path/to/original/synthetic/records.json \
  --model /path/to/original-adapter \
  --base /path/to/base-model \
  --supervised /path/to/supervised-adapter \
  --distilled /path/to/distilled-adapter \
  --out /path/to/new-output
```

After completion, inspect `manifest.json` and `results.json` in the new directory. An interrupted run may leave only `results.partial.json`. This script performs inference for all three models; it is not a lightweight import check. It reads model files with `local_files_only=True`, so all files must be available locally before execution. Interpret results against the exact model snapshots, source data, and font version used.

## Independent synthetic holdout

The three existing models were evaluated on 64 newly generated pages with 192 queries; none of these pages were used for the earlier teacher selection. Top-1 results were 149/192 for the original ColSmol model, 154/192 for supervised fine-tuning, and 151/192 for supervised fine-tuning plus distillation. Compared with supervised fine-tuning, distillation moved 2 queries into first place and 5 out, a net decrease of 3. This run shows no improvement from distillation.

The new pages share the earlier template, and each page contributes three correlated queries. This is a fresh synthetic holdout, not evidence of generalization to real documents or different layouts. Model weights are not distributed here; the recorded ranks can be checked with [recompute.py](recompute.py).
