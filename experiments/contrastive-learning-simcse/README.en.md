# SimCSE Contrastive Learning: Dropout Positives and NLI Hard Negatives

This directory provides scripts to reproduce two SimCSE training objectives (PyTorch and Transformers): the unsupervised version forms positives from two dropout-enabled forward passes of the same sentence; the supervised version uses NLI entailment pairs and hard negatives. Evaluation reports STS semantic similarity (Spearman/Pearson) and retrieval Recall@k for comparing backbones, temperatures, and pooling methods.

See [SimCSE Contrastive Learning](../../docs/zh/retrieval/embedding/SimCSE对比学习.md) for how dropout positives are formed, how InfoNCE and temperature work, and the experiment design choices. The geometric background on representation collapse, alignment, and uniformity is in [Contrastive Learning and Semantic Representations](../../docs/zh/retrieval/embedding/对比学习与表征几何.md).

[scripts](scripts/) · [assets](assets/)

## Scope

This experiment examines whether unsupervised SimCSE's dropout-positive mechanism makes sentence embeddings less anisotropic on general corpora, as reflected in STS score changes; compares supervised NLI triplets with the unsupervised version and studies hard negatives; and checks the effect of temperature, batch size, and CLS versus mean pooling on convergence and metrics.

It does not evaluate instruction-tuned large embedding models such as E5, BGE-M3, or Qwen3-Embedding, whose instruction tuning and large-scale contrastive pretraining differ from SimCSE. It does not measure end-to-end retrieval (ANN index or chunking strategy); retrieval metrics are auxiliary signals of representation quality. Cross-lingual and multimodal settings are also out of scope.

## Setup

| Item | Setting | Notes |
|---|---|---|
| Unsupervised positives | Dropout sampling | Two forward passes of one sentence with dropout enabled in `train` mode |
| Supervised data | NLI entailment, neutral, and contradiction | Anchor and positive are an entailment pair; negative is a contradiction |
| Similarity evaluation | STS-B | Spearman is the primary metric; Pearson is auxiliary |
| Retrieval evaluation | Locally assembled candidate pool | Recall@10, used only to observe representation separability |
| Dependencies | PyTorch, Transformers | No sentence-transformers; pooling is implemented here |

To compare with the paper, use `bert-base-uncased`. The original SimCSE paper (Gao et al., EMNLP 2021, [arXiv:2104.08821](https://arxiv.org/abs/2104.08821)) reports 76.85 on STS-B and 76.25 average across seven STS tasks for unsupervised BERT-base; the supervised results are 84.25 and 81.57. The repository scripts have not completed the full paper setup on any backbone. Commands below describe a reproduction path, not a verified one-click workflow.

The root `requirements.txt` does not contain runtime dependencies. The following Bash example creates an isolated Python environment; before preparing data, install CUDA-enabled PyTorch compatible with the local GPU. Full-training memory depends on model and batch size. In particular, the supervised default batch size is 512 and is not guaranteed to fit every GPU:

```bash
python3 -m venv /path/to/venvs/retrieval
source /path/to/venvs/retrieval/bin/activate
# Install PyTorch matching the system, architecture, and GPU: https://pytorch.org/get-started/locally/
python -m pip install -r requirements.txt
python -m pip check
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA build:', torch.version.cuda, 'GPU:', torch.cuda.is_available()); assert torch.cuda.is_available(), 'Training requires an available CUDA GPU'"
```

Direct dependencies are pinned to `transformers==5.16.1`, `datasets==5.2.0`, and `numpy==2.5.4`; install a PyTorch CUDA build separately for the target device. Transitive dependencies are not fully locked, so this is not a complete cross-platform lockfile.

On 2026-11-15, these dependencies were installed in a fresh isolated environment on an NVIDIA GB10 after installing PyTorch: Python 3.12.3, PyTorch 2.15.0+cu132, CUDA 13.2, Transformers 5.16.1, Datasets 5.2.0, and NumPy 2.5.4. `pip check`, imports, CUDA availability, and BF16 checks passed. This run checked environment setup only; it did not download a model or data or run training. See the [clean-install record](evidence/clean-install-2026-11-15.json).

On 2026-11-15, a fresh isolated environment also completed a small real-data GPU pipeline. After clearing inherited proxy variables for this run, CUDA PyTorch and the pinned experiment dependencies installed successfully and `pip check` passed. The run downloaded revision-pinned `bert-base-uncased`, Wikipedia, SNLI, and GLUE STS-B from the configured Hugging Face mirror; prepared 32 unsupervised sentences, 32 SNLI triplets, and 100 STS-B validation pairs; trained the unsupervised and supervised variants for four steps each; and evaluated the saved models on CUDA. STS-B Spearman was -0.0155 / 0.0387 / 0.3337 for the base / unsupervised / supervised models. The sample and step counts are intentionally tiny: these values confirm the data, train, save, and evaluation path, not model quality or improvement over a baseline. Some dataset file reads needed timeout retries, but the run completed. See the [real-data GPU smoke record](evidence/real-data-smoke-2026-11-15.json). An earlier refused Hub request remains documented in the [failed attempt record](evidence/clean-env-real-data-blocked-2026-11-15.json) as evidence of that specific network state.

On 2026-11-13, a separate minimal GPU smoke test completed in an existing environment on NVIDIA GB10: Python 3.12.3, PyTorch 2.15.0+cu132, Transformers 5.22.2, Datasets 5.2.0, NumPy 2.5.4, tokenizers 0.23.3. With `tiny-random-bert`, unsupervised and supervised training each ran for one step on four synthetic sentences or triplets, then the saved models evaluated four synthetic sentence pairs. All processes exited with code 0. It reused an existing environment and cached model, not a clean installation. Synthetic data does not measure model quality or establish full training or paper reproduction; these versions are not the historical environment. `pip check` failed in that environment with `nvidia-cusparselt-cu13 0.8.1 is not supported on this platform`. A successful small GPU smoke test does not imply a complete dependency state.

Sentence-transformers is not required; sentence embeddings use pooling in `scripts/simcse_lib.py`.

## Prepare data

Training data and evaluation sets are not included. Run the following from this experiment directory to create a new temporary run directory outside the repository (Linux Bash example), so data and weights stay out of the code repository:

```bash
export RUN_DIR="$(mktemp -d "${TMPDIR:-/tmp}/simcse.XXXXXXXX")"
export DATA="$RUN_DIR/data"
export OUT="$RUN_DIR/output"
mkdir "$DATA" "$OUT"
printf 'Run directory: %s\n' "$RUN_DIR"
```

Unsupervised training uses single sentences from a general corpus such as Wikipedia, one JSON object per line:

```json
{"sentence": "A storm dropped five inches of rain on the valley."}
```

This example uses [Wikimedia Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia), configuration `20231101.en`. It is not equivalent to older corpora or the paper's preprocessing. Streaming requires network access; output size depends on filtering. It writes up to 100,000 records, splitting paragraphs on `. ` and retaining sentences 21–199 characters long:

```bash
python - <<'PY'
import json, os
from itertools import islice
from datasets import load_dataset

ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
with open(os.path.join(os.environ["DATA"], "unsup_wiki.jsonl"), "x", encoding="utf-8") as f:
    for row in islice(ds, 100000):
        for sent in row["text"].split(". "):
            sent = sent.strip()
            if 20 < len(sent) < 200:
                f.write(json.dumps({"sentence": sent}, ensure_ascii=False) + "\n")
PY
```

Supervised training uses SNLI or MNLI triplets, one JSON object per line:

```json
{"anchor": "A man is running.", "positive": "Someone jogs.", "negative": "A man sleeps."}
```

The example below pairs entailment and contradiction hypotheses for the same premise, deduplicates them, and skips premises without both labels. This is a convenient preparation example, not the paper's original preprocessing:

```bash
python - <<'PY'
import json, os
from collections import defaultdict
from datasets import load_dataset

ds = load_dataset("stanfordnlp/snli", split="train")
by_premise = defaultdict(lambda: {0: set(), 2: set()})
for row in ds:
    if row["label"] in (0, 2) and row["premise"] and row["hypothesis"]:
        by_premise[row["premise"]][row["label"]].add(row["hypothesis"])
with open(os.path.join(os.environ["DATA"], "snli_triplets.jsonl"), "x", encoding="utf-8") as f:
    for premise, groups in by_premise.items():
        negatives = sorted(groups[2] - {premise})
        if not negatives:
            continue
        for positive in sorted(groups[0] - {premise}):
            if positive != negatives[0]:
                f.write(json.dumps({"anchor": premise, "positive": positive,
                                    "negative": negatives[0]}, ensure_ascii=False) + "\n")
PY
```

STS-B similarity data has one JSON object per line:

```json
{"sentence1": "A man is playing a guitar.", "sentence2": "A man plays guitar.", "score": 3.8}
```

Create the validation file from `nyu-mll/glue`, configuration `stsb`, split `validation`, mapping `sentence1`, `sentence2`, and `label` to the fields above:

```bash
python - <<'PY'
import json, os
from datasets import load_dataset

ds = load_dataset("nyu-mll/glue", "stsb", split="validation")
with open(os.path.join(os.environ["DATA"], "stsb_val.jsonl"), "x", encoding="utf-8") as f:
    for row in ds:
        f.write(json.dumps({"sentence1": row["sentence1"], "sentence2": row["sentence2"],
                            "score": row["label"]}, ensure_ascii=False) + "\n")
PY
```

## Reproduce and evaluate

Run from `experiments/contrastive-learning-simcse/`. `OUT` must be a new directory outside the repository. The data writers use exclusive creation; after an interrupted run, inspect incomplete files and choose a new run directory. Training and evaluation overwrite same-named models or results, so create a fresh run directory before rerunning.

Unsupervised training (paper hyperparameters: batch 64, learning rate 3e-5, temperature 0.05, one epoch):

```bash
python scripts/train_unsup_simcse.py --model bert-base-uncased --data "$DATA/unsup_wiki.jsonl" --out "$OUT/unsup" --batch-size 64 --lr 3e-5 --temperature 0.05 --epochs 1 --device cuda
```

Supervised training (paper hyperparameters: batch 512, learning rate 3e-5, temperature 0.05, three epochs):

```bash
python scripts/train_sup_simcse.py --model bert-base-uncased --data "$DATA/snli_triplets.jsonl" --out "$OUT/sup" --batch-size 512 --lr 3e-5 --temperature 0.05 --epochs 3 --device cuda
```

Evaluate the backbone and both fine-tuned models with the same protocol:

```bash
python scripts/evaluate_sts.py --model bert-base-uncased --sts-files "$DATA/stsb_val.jsonl" --json-out "$OUT/base-eval.json"
python scripts/evaluate_sts.py --model "$OUT/unsup" --sts-files "$DATA/stsb_val.jsonl" --json-out "$OUT/unsup-eval.json"
python scripts/evaluate_sts.py --model "$OUT/sup" --sts-files "$DATA/stsb_val.jsonl" --json-out "$OUT/sup-eval.json"
```

Add `--retrieval-file` for an auxiliary retrieval metric; see the docstring in `scripts/evaluate_sts.py` for its format.

### Smoke test and recorded synthetic run

The smoke test uses randomly templated Chinese sentences to check training, saving, loading, and evaluation. If the default model is not cached, it downloads it. This checks the code path, not model quality:

```bash
python scripts/smoke_test.py --model hf-internal-testing/tiny-random-bert
```

An already-cached small model can be used instead: `python scripts/smoke_test.py --model /path/to/local/model`. This script fixes the device to CPU; it does not verify GPU operation.

### Small real-data run

On 2026-11-14, a copied and adjusted isolated environment on Linux ARM64 and NVIDIA GB10 passed `pip check` with Python 3.12.3, PyTorch 2.15.0+cu132, Transformers 5.16.1, and Datasets 5.2.0. Starting from `bert-base-uncased` revision `86b5e0934494bd15c9632b12f734a8a67f723594`, the unsupervised and supervised scripts each ran one CUDA epoch with batch size 8 and four steps, using 32 Wikipedia sentences and 32 SNLI triplets respectively. The base model and both outputs were evaluated on the same 100 STS-B validation examples.

Spearman correlations on this subset were 0.0013 for the base, 0.0387 for the unsupervised model, and 0.3337 for the supervised model. The sample and training budgets are small, and dataset revisions were not pinned. This checks training, saving, and evaluation only; it does not establish model quality, reproduce the full paper results, or validate installation from scratch.

The model revision, environment, training configuration and evaluation metrics are recorded in the [run summary](evidence/real-data-smoke-2026-11-14.json). Raw training metrics and evaluation logs remain in the internal run archive; the summary does not invent dataset revisions or file hashes that were not recorded.

An existing synthetic run record uses `hf-internal-testing/tiny-random-bert` and 500 template sentences for three epochs and 96 steps. One training record gives `elapsed_seconds` as 1.70 seconds; another summary labels hardware NVIDIA GB10 and records `train_duration_sec` as 6.059 seconds. Their timing boundaries are unspecified, so the values cannot be directly compared. See [training record](evidence/synthetic-training-metrics.json) and [metric summary](evidence/synthetic-eval-summary.json).

The summary reports Spearman 0.8215 and Pearson 0.6975 on 500 sentence pairs. Of the saved pairs, 262 are identical texts labeled 5; the other 238 contain different texts labeled below 2. This template data checks the workflow and does not establish performance on real semantic-similarity tasks. The original pairs and model weights are not distributed, so the public summary cannot independently reconstruct these metrics. Run date and full environment have not been confirmed from the surviving summary; timing and memory values are not performance benchmarks.

## Outputs and limitations

- The `--out` directory contains backbone weights, tokenizer, `pooler.pt`, `simcse_config.json`, and `train_metrics.json` (loss curve and final-segment mean).
- `--json-out` writes evaluation JSON with Spearman, Pearson, and sample count for each evaluation file.
- Runtime `results/` and `models/` directories are ignored by the public `.gitignore`. Copy metrics into a report if they need to be retained.
- The real-data run above is a small functional check. Clean-environment setup was verified separately, but training was not run in that environment. Full-data training and the paper results have not been reproduced; paper metrics are cited for reference only.
- The optional MLM auxiliary loss is simplified: masked positions are replaced directly by `[MASK]`, without the 80/10/10 replacement strategy, and it is disabled by default.
- The unsupervised version has no momentum encoder or queue (MoCo style); large batches depend on available memory. If memory is insufficient, lower `--batch-size` and adjust the learning rate accordingly.
- Retrieval evaluation computes Recall@10 on one candidate pool. Pool size changes absolute values, so comparisons require a fixed pool.
- Temperature and pooling affect results substantially. When changing backbones, rerun the same hyperparameter set; do not compare across configurations directly.
- Windows can run the CPU smoke test. Full reproduction requires a Linux CUDA environment.

[中文说明](README.md)
