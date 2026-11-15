# Reproducing laya RLCD Fine-Tuning

[中文说明](README.md)

For classification, scoring, or choice decisions, a model may need to output probabilities as well as an answer so downstream systems can choose actions according to risk. This experiment compares two training methods: RLCD adds policy gradients based on scoring rules to the distillation loss, while `soft-ce` uses only the distillation loss. The question is whether the former improves accuracy or probability calibration on the same dataset.

This is a single-GPU reproduction of laya's RLCD fine-tuning on a DGX Spark (GB10, 128 GB unified memory, aarch64). It starts from the public `typed-decisions` dataset and laya English checkpoint, then trains a decision model with RLCD, which combines a GRPO-style policy gradient with proper-scoring-rule rewards and soft cross-entropy, and compares its fit to the reference distributions.

RLCD stands for Reinforcement Learning for Calibrated Decisions, the official training method from TypeSafe Jev. laya is an open-source project that publishes a complete RLCD training implementation. Training is implemented in `laya/train.py`; its notebook is `notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb` in the laya repository. The official notebook targets free Kaggle 2xT4 DDP. This experiment adapts it to a single-GPU CUDA path without DDP.

## Recipe (aligned with the notebook)

| Item | Value | Source |
|---|---|---|
| Backbone | `convaiinnovations/laya` English checkpoint (ModernBERT-large, 421M) | Notebook cell 6 |
| Effective batch size | 64 = micro_batch 8 x grad_accum 8 | `TrainConfig` comment in `train.py` (notebook uses 8 x 2 GPUs x 4 accumulation steps) |
| Loss | `rlcd` (default): GRPO-style Gaussian-noise sampling + proper-scoring-rule reward + soft CE | `rlcd_loss` in `train.py` |
| `rl_samples` / `sigma` | 4 / 0.4 → 0.1 linear annealing | `TrainConfig` defaults |
| `w_sph` / `w_rps` | 0.75 / 1.0 | `TrainConfig` defaults |
| Epochs | 4 | Notebook |
| `max_len` / `head_max_len` | 512 / 192 (included with the English checkpoint) | `rl_agent_config.json` |
| AMP / gradient checkpointing | Enabled (CUDA default) | `train_model` |
| Calibration | `calib_frac` 0.1, `calib_max` 400, one temperature per question type | `TrainConfig` defaults |
| Data | `LocalLLaMA/typed-decisions`: 1,200 train cases / 400 test cases (held-out evaluation) | Hugging Face dataset |

## Runtime environment

| Item | Value |
|---|---|
| Hardware | NVIDIA DGX Spark (GB10 Grace Blackwell, 121 GB unified memory, aarch64) |
| OS | Ubuntu 24.04 LTS |
| Python | 3.12 |
| torch | 2.15.0+cu132 (verified in the fresh environment) |
| datasets / huggingface_hub | 5.2.0 / 1.33.0 |
| laya | source commit `3a38cbe83f6ef556f3125fb0065fdc5a55d05792`, installed version 0.4.31 |

## Prepare data and run

The following combination was installed in a fresh isolated environment on a DGX Spark (Linux ARM64, Ubuntu 24.04, Python 3.12.3), passed dependency checks, and completed a full GPU A/B run. For other devices, select a PyTorch build compatible with their architecture and CUDA version; do not reuse the GB10 wheel blindly. Laya, the dataset, and the model are pinned to public Hugging Face or source commits:

```bash
python3 -m venv .venv
source .venv/bin/activate
# Verified on DGX Spark / Linux ARM64; choose a compatible wheel for other devices
python -m pip install --index-url https://download.pytorch.org/whl/cu132 "torch==2.15.0+cu132"
python -m pip install -r requirements.txt
python -m pip check
python -c "import torch, datasets, huggingface_hub; from laya.agent import _fix_tokenizer_config; from laya.train import TrainConfig, finetune; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), datasets.__version__, huggingface_hub.__version__); assert torch.cuda.is_available()"
```

`requirements.txt` pins direct dependencies and the Laya source commit, but does not lock every transitive dependency; indirect versions may change across installation dates. The CUDA wheel, data, and model require network downloads.

```bash
export DATASET_REVISION=cc1a2e795f5160d417d4d7131a71d1ab60ea33ca
export MODEL_REVISION=85cc74ddae6109077ebe2853a9b3077fb6606a78
export XL_DATA_DIR="$PWD/data/typed-decisions-$DATASET_REVISION"
export XL_OUT_DIR="$PWD/runs/rlcd-ab-$DATASET_REVISION"
export MODEL_DIR="$PWD/models/laya-$MODEL_REVISION"
mkdir -p models
test ! -e "$MODEL_DIR"
MODEL_DIR="$MODEL_DIR" MODEL_REVISION="$MODEL_REVISION" python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(
    "convaiinnovations/laya",
    revision=os.environ["MODEL_REVISION"],
    local_dir=os.environ["MODEL_DIR"],
    allow_patterns=["rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"],
)
PY
python scripts/prepare_data.py --revision "$DATASET_REVISION" --out-dir "$XL_DATA_DIR"
export PYTHON="$PWD/.venv/bin/python"
bash scripts/run_ab_compare.sh
```

The model is downloaded into a private directory because the upstream tokenizer fix may change `tokenizer/tokenizer_config.json`. Data and run outputs also use separate directories; set new `XL_DATA_DIR` and `XL_OUT_DIR` values for each run because the script will not overwrite existing arm outputs. Before starting, Linux checks available memory and requires at least 50 GiB by default; adjust with `MIN_AVAILABLE_GIB`. This is a startup gate, not continuous monitoring. The full run took about 26 minutes per arm and about 52 minutes total; actual time depends on the device and system load.

Data preparation checks for and exclusively creates both JSONL files; it stops if either target exists. An interrupted download or conversion may leave incomplete files; keep anything you need and retry in a new directory. `run_ab_compare.sh` uses the directories set by `XL_DATA_DIR` and `XL_OUT_DIR`. For this pinned revision, the train and test JSONL SHA-256 values were `6c84f9a369e25a14c5ae47fc567cc86553f56bac128efb48127a94971d77dcc2` and `0c6712fc6657dca6371ab96863b3c8f1306a67631141623ea98d066fabbd58c4`; verify them with `sha256sum` after preparing the data.

## Files

```text
rlcd-policy/
├── scripts/
│   ├── prepare_data.py        # typed-decisions → {state, questions, gold} JSONL
│   ├── run_finetune.py        # Single-GPU training entry point (--loss rlcd / soft-ce comparison arms)
│   └── run_ab_compare.sh      # One-command A/B comparison (runs both arms serially and compares them side by side)
├── data/                      # JSONL dataset prepared at runtime
├── evidence/                  # Saved run records and clean-environment GPU A/B evidence
└── runs/                      # Checkpoints and train_report.json generated at runtime
```

## November 15, 2026: full GPU A/B run in a clean environment

In a fresh isolated environment, RLCD and soft-ce each trained for four epochs and were evaluated on the same 2,000 held-out questions. Both arms used the same seed, model, data, and training settings; the intended difference was the loss. Each arm completed 352 optimizer updates.

| Metric after fine-tuning | RLCD | soft-ce | Lower value |
|---|---:|---:|---|
| Top-1 agreement with teacher reference labels | 0.7220 | 0.7815 | — |
| ECE | 0.1071 | 0.1464 | RLCD |
| Multiclass Brier | 0.0753 | 0.0508 | soft-ce |
| Top-1 Brier | 0.1792 | 0.1685 | soft-ce |
| Mean confidence | 0.6151 | 0.6351 | — |
| Final evaluation loss | 0.9019 | 0.8581 | soft-ce |

In this single comparison, soft-ce had higher accuracy and lower Brier scores, while RLCD had lower ECE. There is only one new run per method, so these results do not establish a stable advantage. The dataset contains synthetic cases with teacher-generated soft labels; the metrics measure fit to reference distributions and do not establish real-world calibration. The upstream trainer also warned of conflicting schemas for `action`, `urgency`, and `disposition`, keeping the first definition encountered. The RLCD arm warned that the learning-rate scheduler is stepped before the optimizer, which may skip the first scheduled learning rate. Both arms emitted a Transformers deprecation warning for the gradient-checkpointing API. See the [clean-environment full GPU A/B record](evidence/clean-gpu-ab-2026-11-15.json).

## Earlier run results (seed 0, 2,000 held-out questions)

In earlier records, top-1 agreement with the reference label was 0.7510 and 0.7700 for two RLCD runs, and 0.7855 for one `soft-ce` run. Those records did not pin the dataset and model revisions used in the new clean-environment run above, so they remain separate and should not be pooled as directly comparable results. The cases are synthetic, and the labels are soft distributions generated by a teacher model rather than real-world outcomes. ECE and Brier scores are also measured against the reference distributions; they do not establish real-world probability calibration. The current upstream implementation splits training and calibration data after expanding cases into questions, so questions from the same case may appear in both splits. The two earlier RLCD runs record the same configuration and seed, but the cause of their difference has not been identified. Taken together, the earlier records and this new comparison are still too few to determine the general advantage or stability of either loss, or to establish that they are equivalent.

Comparisons with external results are limited by differences in evaluation protocol and the traceability of evidence. Annotator agreement cannot be treated directly as an upper bound on model capability. See the [RLCD fine-tuning report](../../../docs/zh/decision-models/RLCD微调报告.md) for details. The historical comparison took about 54 minutes; rerun time depends on the environment.

Upstream sources: the [laya repository](https://github.com/NandhaKishorM/laya) (Apache-2.0), the [`LocalLLaMA/typed-decisions` dataset](https://huggingface.co/datasets/LocalLLaMA/typed-decisions), and the [`convaiinnovations/laya` weights](https://huggingface.co/convaiinnovations/laya).
