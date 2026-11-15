# DPO preference alignment

[中文](README.md)

This experiment applies Direct Preference Optimization (DPO) to a small existing chat model using public preference pairs. It tests whether training changes the model's relative likelihood of the two supplied answers. See the [explanation](../../docs/en/alignment/direct-preference-optimization.md) and the separate [language model lifecycle](../llm-lifecycle-fullchain/README.en.md) experiment, which includes reward- and sampling-based alignment.

## Environment

Use Linux, Python 3.12, a BF16-capable NVIDIA CUDA GPU, and enough disk and device memory for the data, model and training outputs. Minimum memory has not been established. The frozen reference and trainable model share the base weights; reference scores are computed with LoRA disabled. Install CUDA PyTorch appropriate for the driver first:

```bash
cd experiments/dpo-preference-alignment
python3.12 -m venv .venv
source .venv/bin/activate
# GB10 / Linux aarch64: reproduce the version used here; select a matching build for other devices
python -m pip install --index-url https://download.pytorch.org/whl/cu132 "torch==2.15.0+cu132"
python -c "import importlib.metadata as m; print('torch==' + m.version('torch'))" > torch-constraint.txt
python -m pip install -c torch-constraint.txt -r requirements.txt
python -m pip check
python -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()"
```

The constraint prevents the remaining installation from replacing the existing PyTorch build. On 2026-11-15, a new virtual environment on NVIDIA GB10 (Linux aarch64) was installed with these commands; `pip check`, imports and CUDA/BF16 checks passed. A further run started from a new empty cache, downloaded the pinned model and dataset, and completed 64 training steps and evaluation on 128 pairs in the same clean environment. Rerun with the script below and fresh external directories. This verifies Linux aarch64 / Python 3.12.3 / CUDA 13.2; the requirements are not a full lock file, so pip still resolves transitive dependencies.

```bash
export XL_DATA_DIR=/path/to/dpo-clean-data/run-001
export XL_OUT_DIR=/path/to/dpo-clean-output/run-001
export XL_RUN_ID=run-001
export XL_REQUIRE_EXTERNAL_ENV=1
bash scripts/clean_env_gpu_smoke.sh
```

The script creates a new virtual environment, Hugging Face cache and data files under `XL_DATA_DIR`, and saves the adapter and report under `XL_OUT_DIR`. It refuses to overwrite existing targets.

## Model and data

- [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B), revision `c1899de289a04d12100db370d81485cdf75e47ca`, Apache-2.0. This is a post-trained chat model, not the Base model.
- [HuggingFaceH4/ultrafeedback_binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized), revision `3949bf5f8c17c394422ccfab0c31ea9c20bdeb85`, declared MIT. Preference labels are based on model scores rather than direct human annotation; the dataset card records earlier mislabeled examples and benchmark contamination.
- Sample the official `train_prefs` and `test_prefs` splits with a fixed seed, check exact prompt overlap between selected examples and preserve input metadata. Near duplicates and pretraining overlap remain possible.

This run downloaded the pinned revisions from a Hugging Face mirror. The actual SHA-256 values of the model and two preference files matched official repository LFS metadata:

| File | SHA-256 |
|---|---|
| Model `model.safetensors` | `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b` |
| Data `train_prefs-00000-of-00001.parquet` | `0f951ca4502001d31f3e4c70716ae51d20e4ce4f847d12b6a6695a40d4d353a8` |
| Data `test_prefs-00000-of-00001.parquet` | `e9dab2789f419d4204d73ec2c860af6d88d466b906e0109e69b96075467eb389` |

## Run

The command downloads the model and data, trains a LoRA adapter and evaluates the log-probability difference for provided answer pairs. Use a fresh output directory; do not commit caches or checkpoints.

```bash
export XL_DATA_DIR="$PWD/.work/data"
export XL_OUT_DIR="$PWD/.work/run-01"
export HF_HOME="$XL_DATA_DIR/huggingface"
python scripts/run_dpo.py --train-size 256 --test-size 128 --steps 64 --seed 42
```

After downloading, compare the output hashes to the table above:

```bash
sha256sum "$XL_DATA_DIR/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca/model.safetensors"
sha256sum "$HF_HOME/hub/datasets--HuggingFaceH4--ultrafeedback_binarized/snapshots/3949bf5f8c17c394422ccfab0c31ea9c20bdeb85/data/"{train_prefs,test_prefs}-00000-of-00001.parquet
```

Inspect `$XL_OUT_DIR/report.json` when finished. Prompts retain their last 192 tokens; answers retain their first 191 tokens followed by EOS. Truncation can remove information relevant to the preference. Pairs whose tokenized answers become identical are removed; the report counts truncations and remaining pairs, but truncation can also change the preference meaning of other pairs. The test compares the sum of answer-token log probabilities, which is affected by length; it does not measure generation quality, user satisfaction or safety. A short, single-seed run and a small test set cannot establish general alignment improvement. Check licenses and resource requirements before running.

`HF_HOME` directs the general Hugging Face cache into the data directory. The 2026-11-15 fresh-environment rerun verified downloading the model and dataset from an empty cache and completing the training/evaluation flow.

## Status

On 2026-11-15, a Linux aarch64 run on NVIDIA GB10 completed 64 CUDA LoRA updates and evaluated 128 pairs using Python 3.12, PyTorch 2.15.0+cu132, Transformers 5.16.1, datasets 5.2.0, PEFT 0.21.3 and Accelerate 1.15.0. The earlier training used an isolated copy of an existing environment. An additional rerun started from a fresh virtual environment and empty cache, downloaded the model and dataset, and completed the full workflow. Pair accuracy remained 71/128 before and after training; the bootstrap 95% interval for the accuracy change was [0, 0], and the mean reference-relative margin change was +0.174. A prior run with the same nominal seed recorded +0.136, so this value was not exactly reproducible under the current implementation and hardware; we do not claim deterministic reproduction. PyTorch peak allocated CUDA tensor memory was about 3.60 GiB, not total system memory or a general minimum. See the [sanitized fresh-environment run evidence](evidence/clean-env-gpu-smoke-2026-11-15.json).

Comparing summed log probabilities for the supplied answers, both the original and trained model matched the dataset preference on 71/128 test pairs. No individual pair changed predicted direction. The paired bootstrap 95% interval for the accuracy difference was [0, 0], reflecting no flips in this sample rather than proving a zero difference on future data. Mean reference-relative margin changed by +0.136, which does not establish better generation. In training, 69/256 prompts, 128/256 preferred and 116/256 rejected answers were truncated; the test counts were 21/128, 69/128 and 66/128. No pairs became identical after truncation, but remaining preferences can still be affected. Model-scored labels, short training, one seed, length bias, possible near duplicates and pretraining overlap limit the conclusion. The first mean DPO loss was 0.693147, consistent with an initially near-zero reference-relative margin; A separate follow-up initialized a new adapter with the same model and LoRA configuration and compared logits with the adapter enabled and disabled for all 256 answers in the 128 test pairs; the maximum absolute difference was 0. This checks a new initialization after training, not outputs recorded at the start of the original run.

See [summary.json](evidence/summary.json) for metrics and settings and the [explanation](../../docs/en/alignment/direct-preference-optimization.md).

## References

- Rafailov et al., [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290), 2023.
- Cui et al., [UltraFeedback](https://arxiv.org/abs/2310.01377), 2023.
