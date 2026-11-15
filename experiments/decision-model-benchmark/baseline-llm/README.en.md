# Jev Series Experiment: Generative Baseline

This is the control group, not a competing entry. It answers one question: how long does an ordinary generative LLM take to perform the same classification task? Without this reference, the advantage of discriminative models has no baseline.

## What the scripts measure

`qwen4b-latency.py` benchmarks Qwen3-VL-4B. Its text input, task definition, and all four categories are copied verbatim from `../laya-dual-branch/laya-latency.py`. It runs on the same machine and in the same environment, with the same warm-up-before-timing procedure.

Generative-model latency has two parts. **Prefill** is the forward pass that reads the input without producing any tokens; it is the wait before the first token and scales with input length. **Decode** is the subsequent autoregressive generation of tokens, and scales with output length. **Full** is their sum.

The script tests two prompt formats. The minimal format permits only a category name, minimizing decoding and giving this model its most favorable case. The JSON format requires a complete JSON object, which is more common in engineering and requires substantially more decoding. Both are needed to show the real cost of structured output with a generative approach.

## Environment setup and how to run

The scripts target Linux and a CUDA GPU that supports bfloat16. You need Python, a device-compatible CUDA build of PyTorch, and Transformers and Accelerate versions that support Qwen3-VL. `requirements.txt` pins Transformers 5.22.2 and Accelerate 1.15.0; install PyTorch separately for the device. This is not a complete lock of transitive dependencies, so use an isolated environment. The default model is `Qwen/Qwen3-VL-4B-Instruct`. Both scripts accept another model or local path through `QWEN_MODEL`; set `QWEN_MODEL_REVISION` to pin its revision.

For example, after creating a virtual environment and installing a CUDA build of PyTorch compatible with your device, install the other direct dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
# Select and install a CUDA build of PyTorch compatible with your device first
python -m pip install -r requirements.txt
python -c "import torch, transformers; print('torch:', torch.__version__, 'CUDA:', torch.cuda.is_available(), 'transformers:', transformers.__version__); assert torch.cuda.is_available()"
```

The diagnostic script loads the selected model tokenizer in forced offline mode. Prepare the cache while online first, then run from this directory. To pin the model version used in this benchmark:

```bash
export QWEN_MODEL=Qwen/Qwen3-VL-4B-Instruct
export QWEN_MODEL_REVISION=ebb281ec70b05090aa6165b016eac8ec08e71b17
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('$QWEN_MODEL', revision='$QWEN_MODEL_REVISION')"
python qwen4b-diag.py
BENCH_OUT=qwen3-vl-result.json python qwen4b-latency.py
```

`qwen4b-diag.py` does not load model weights, but loads the tokenizer selected by `QWEN_MODEL` and `QWEN_MODEL_REVISION` offline; the tokenizer cache and dependencies must already be available. It checks that the template follows the text-only path. `BENCH_OUT` selects where the benchmark results JSON is saved.

`scripts/clean_env_gpu_smoke.sh` creates a fresh isolated environment, installs CUDA PyTorch and the dependencies listed here, then runs one CUDA generation check with the cached or downloadable Qwen2.5-0.5B model. Run it from this directory:

```bash
bash scripts/clean_env_gpu_smoke.sh
```

The default PyTorch package is `torch==2.15.0+cu132`, an example for Linux ARM64; set `PYTORCH_SPEC` and `PYTORCH_INDEX_URL` to a build compatible with another device and driver. Set `XL_DATA_DIR` and `XL_OUT_DIR` to choose the environment and result directories. The script refuses to reuse an existing environment directory. Reserve disk space for the model weights.

On 2026-11-15, the final `scripts/clean_env_gpu_smoke.sh` created and completed a fresh isolated environment on DGX Spark (Linux ARM64, Python 3.12.3, NVIDIA GB10). PyTorch 2.15.0+cu132, Transformers 5.22.2, and Accelerate 1.15.0 installed, and `pip check` reported no dependency issues. Qwen2.5-0.5B-Instruct revision `7ae557604adf67be50417f59c2c2f167def9a775` generated `Billing` (2 tokens) on `cuda:0`; model loading took 8.39 seconds and peak GPU allocation was 0.95 GiB. The script pins this model revision by default; set `QWEN_MODEL_ID` and `QWEN_MODEL_REVISION` when selecting another model. See the [GPU smoke record](evidence/2026-11-15-clean-env-gpu-smoke.json). This run verifies the setup entry point, model loading, and one GPU generation; it does not reproduce the default Qwen3-VL-4B benchmark or evaluate output quality.

### Check inference with a smaller model

Before downloading the default model, the following runs one forward pass and generation on CUDA from this directory. The first run downloads `Qwen/Qwen2.5-0.5B-Instruct`; set `QWEN_MODEL` to a local model directory if it is already available. This uses a different model and one English input without the complete timing loop, so it does not reproduce the Qwen3-VL-4B metrics below.

```bash
QWEN_MODEL=Qwen/Qwen2.5-0.5B-Instruct python - <<'PY'
import runpy

benchmark = runpy.run_path("qwen4b-latency.py", run_name="benchmark_module")
tokenizer, model, _ = benchmark["load"]()
prompt = benchmark["build_prompt"]("minimal", "en", benchmark["EN"])
result = benchmark["bench"](model, tokenizer, prompt, 12)
print("device:", next(model.parameters()).device)
print("output:", repr(result["text"]))
assert next(model.parameters()).is_cuda and result["n_new"] > 0 and result["text"].strip()
PY
```

On 2026-11-13, this check ran with the cached model in an existing Linux ARM64 environment on NVIDIA GB10, using PyTorch 2.15.0+cu132, Transformers 5.22.2, and Accelerate 1.15.0. It ran on `cuda:0` and returned `Billing`. A fresh environment install, the default 4B model, the complete benchmark, and matching model revisions were not verified in that run.

On its first run, `qwen4b-latency.py` automatically downloads about 8.3 GB of weights, takes about 54 seconds to load, and then keeps 8.3 GB resident. Release the model after the run; do not keep it resident alongside the Nimble experiment.

Results correspond to excerpt No. 08 in `../evidence/`. Excerpt No. 08b records an earlier failed debugging run where the template was not applied; the later script fixed this, so 08b is not the normal reproduction path.

## Why use AutoTokenizer instead of a processor

A vision-language model's processor inserts image-placeholder tokens into the prompt. Prefill time then grows with the virtual image, so the measurement no longer represents a text-only task.

## Measured results

On 2026-11-15, the full benchmark ran on DGX Spark (Linux ARM64, NVIDIA GB10) in an isolated environment, using the pinned Qwen3-VL-4B revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`. Each input was warmed up 3 times and measured 30 times. Model loading took 442.23 seconds as logged, including the first download of about 8.3 GB of weights. The loaded model used 8.27 GiB resident memory and peaked at 8.34 GiB of CUDA allocation. There were no empty outputs in any of the four cases. The p50 full latency was 98.78 ms for English minimal output, 100.00 ms for Chinese minimal output, 1,410.73 ms for English JSON, and 1,123.87 ms for Chinese JSON. See the [GPU benchmark record](evidence/2026-11-15-qwen3-vl-gpu-rerun.json) for all measurements.

This run verifies the complete Qwen3-VL timing procedure. Laya was not rerun under matching conditions, so this result should not be used to calculate a direct speed ratio. Historical runs measured a 49–52 ms single forward pass and about 1,050–1,450 ms for JSON output; this run's prefill p50 ranged from 47.84 to 50.41 ms, consistent with that range.

The difference between discriminative and generative models lies in cost structure, not just absolute speed. Generative cost grows with output length; a discriminative model generates no text, so its latency does not change with the number of output fields. This makes discriminative models suitable for high-throughput decision paths, but not for tasks that need generated explanations.

[中文](README.md)
