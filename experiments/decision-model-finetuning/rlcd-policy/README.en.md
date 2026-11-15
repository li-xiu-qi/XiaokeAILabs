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
| torch | 2.14.0+cu130 (aarch64 CUDA 13 wheel) |
| laya | 0.4.1 (historically recorded; exact commit unverified) |

## Prepare data and run

Create an isolated Linux Python environment in this directory. First install a CUDA-compatible PyTorch build for the device, then install the data dependencies and upstream Laya source. The pinned commit below is a public source revision that contains `laya.train`; **it has not been confirmed as the revision used in the historical run**. These installation commands have not been verified end to end. Check the imports and training interface before running:

```bash
python3 -m venv .venv
source .venv/bin/activate
# Choose and install a CUDA-enabled PyTorch build at https://pytorch.org/get-started/locally/
python -m pip install -r requirements.txt
python -c "import torch, datasets; from laya.agent import _fix_tokenizer_config; from laya.train import TrainConfig, finetune; print('torch:', torch.__version__, 'CUDA:', torch.cuda.is_available()); assert torch.cuda.is_available()"
```

Installation requires Git and network access. Confirm that the check reports CUDA as available. `laya` and its transitive dependencies are not fully pinned, and the exact Laya commit used in the historical run has not been verified. If the interface or dependencies are incompatible, investigate before running. After preparing the data, point the script to this environment's interpreter:

```bash
python scripts/prepare_data.py --out-dir data
export PYTHON="$(command -v python)"
bash scripts/run_ab_compare.sh
```

Neither data downloads nor the default model download pin a revision. Record upstream versions and file hashes before rerunning. The upstream tokenizer fix may change `tokenizer/tokenizer_config.json` in the model directory, which defaults to the shared Hugging Face cache. To protect shared caches, prepare a separate model copy and run `MODEL_DIR=/path/to/model bash scripts/run_ab_compare.sh`, or use `--model-dir` with the training entry directly. The script defaults to `python` in the active environment; set `PYTHON` to select another interpreter. On Linux it checks available memory using `/proc/meminfo`, requiring at least 50 GiB by default; adjust this with `MIN_AVAILABLE_GIB`. This check runs only before startup and does not monitor memory continuously during training. To prevent overwrites, the script exits if `runs/rlcd`, `runs/soft-ce`, or their corresponding logs already exist. The exact upstream revision used by the historical run has not been verified, so this is a reproduction path and does not guarantee identical numbers.

Data preparation ran on Linux aarch64 with Python 3.12 and datasets 5.2.0, producing 1,200 training cases and 400 test cases; this did not verify RLCD training. Data preparation checks for and exclusively creates two JSONL files before downloading; it will not overwrite either target if it already exists. If downloading or conversion is interrupted, outputs may be empty or incomplete. Keep any files you need and prepare data in a new directory. `run_ab_compare.sh` reads `data/typed_decisions_train.jsonl` and `data/typed_decisions_test.jsonl` from this directory. To use another output directory, first verify the data, then place it at those paths in a new experiment working copy.

## Files

```text
rlcd-policy/
├── scripts/
│   ├── prepare_data.py        # typed-decisions → {state, questions, gold} JSONL
│   ├── run_finetune.py        # Single-GPU training entry point (--loss rlcd / soft-ce comparison arms)
│   └── run_ab_compare.sh      # One-command A/B comparison (runs both arms serially and compares them side by side)
├── data/                      # JSONL dataset prepared at runtime
├── evidence/                  # Three saved experiment records
└── runs/                      # Checkpoints and train_report.json generated at runtime
```

## Results (seed 0, 2,000 held-out questions)

The top-1 agreement with the reference label was 0.7510 and 0.7700 for two RLCD runs, and 0.7855 for the single `soft-ce` run. The cases are synthetic, and the labels are soft distributions generated by a teacher model rather than real-world outcomes. ECE and Brier scores are also measured against the reference distributions; they do not establish real-world probability calibration. The current upstream implementation splits training and calibration data after expanding cases into questions, so questions from the same case may appear in both splits. The two RLCD runs record the same configuration and seed, but the cause of their difference has not been identified. There are too few repeats to determine the general advantage of either loss or to establish that they are equivalent. ECE against the reference labels was lower in both RLCD runs than in the `soft-ce` run, but more repeats are needed to assess stability.

Comparisons with external results are limited by differences in evaluation protocol and the traceability of evidence. Annotator agreement cannot be treated directly as an upper bound on model capability. See the [RLCD fine-tuning report](../../../docs/zh/decision-models/RLCD微调报告.md) for details. The historical comparison took about 54 minutes; rerun time depends on the environment.

Upstream sources: the [laya repository](https://github.com/NandhaKishorM/laya) (Apache-2.0), the [`LocalLLaMA/typed-decisions` dataset](https://huggingface.co/datasets/LocalLLaMA/typed-decisions), and the [`convaiinnovations/laya` weights](https://huggingface.co/convaiinnovations/laya).
