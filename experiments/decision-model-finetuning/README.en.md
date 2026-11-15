# Training and Fine-Tuning Discriminative Decision Models

This directory contains engineering experiments in training and fine-tuning dedicated discriminative decision models, including full-parameter classification-head fine-tuning, an RLCD (rule-based decision reinforcement) versus soft-label loss comparison, and multimodal visual decision exploration. See [Decision Models Guide](../../docs/zh/decision-models/README.md) for the accompanying guide.

## Training projects

| Directory | Training objective and backbone | Data and method | Key finding and status |
|---|---|---|---|
| [agent-jev-text/](agent-jev-text/) | Text-only discriminative decision model (Qwen3-0.6B backbone) | `LocalLLaMA/typed-decisions`; full-parameter fine-tuning of the classification head | Training, dev-set calibration, test-set evaluation, and a lightweight inference service wrapper are complete |
| [rlcd-policy/](rlcd-policy/) | Policy-gradient RLCD versus soft cross-entropy comparison | Fine-tunes Laya on one DGX Spark GPU; compares rule-reward training with pure distillation fitting | Measured probability-calibration differences between soft-label supervision and policy gradients |
| [vision-decision/](vision-decision/) | Visual discriminative decision model (multimodal backbone plus scalar-score head) | Scores multiple image-text candidates for a decision | **Important limitation:** historical data included the reference answer field `solution`; retraining and evaluation with the corrected data preparation are not yet complete |

## Reproduction prerequisites

- Prepare a separate Python environment and CUDA-enabled PyTorch for each project. See each subdirectory README for dependencies, downloads, and run steps. Dependency lists are not complete locks of the historical environments.
- `agent-jev-text` and `rlcd-policy` use the synthetic `typed-decisions` dataset; `vision-decision` uses ScienceQA. Downloading data and weights requires network access and disk space, and use must comply with upstream licenses.
- The historical `rlcd-policy` run used a DGX Spark (GB10 unified memory). Training memory requirements and reproducibility on other devices have not been established. `vision-decision` relies on Linux memory checks; its historical data leaked answers, so the historical results cannot establish visual question-answering performance.

[中文说明](README.md)
