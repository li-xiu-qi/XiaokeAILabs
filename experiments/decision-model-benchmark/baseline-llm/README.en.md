# Jev Series Experiment: Generative Baseline

This is the control group, not a competing entry. It answers one question: how long does an ordinary generative LLM take to perform the same classification task? Without this reference, the advantage of discriminative models has no baseline.

## What the scripts measure

`qwen4b-latency.py` benchmarks Qwen3-VL-4B. Its text input, task definition, and all four categories are copied verbatim from `../laya-dual-branch/laya-latency.py`. It runs on the same machine and in the same environment, with the same warm-up-before-timing procedure.

Generative-model latency has two parts. **Prefill** is the forward pass that reads the input without producing any tokens; it is the wait before the first token and scales with input length. **Decode** is the subsequent autoregressive generation of tokens, and scales with output length. **Full** is their sum.

The script tests two prompt formats. The minimal format permits only a category name, minimizing decoding and giving this model its most favorable case. The JSON format requires a complete JSON object, which is more common in engineering and requires substantially more decoding. Both are needed to show the real cost of structured output with a generative approach.

## How to run

The scripts target Linux and a CUDA GPU that supports bfloat16. You need Python, a device-compatible CUDA build of PyTorch, and versions of Transformers and Accelerate that support Qwen3-VL. The repository does not provide a complete dependency lock file for these scripts; versions may conflict with other experiments, so an isolated environment is recommended. The default model is `Qwen/Qwen3-VL-4B-Instruct`. The benchmark accepts another model or local path through `QWEN_MODEL`, but the diagnostic script always reads the tokenizer cache for the default model.

For example, after creating a virtual environment and installing a CUDA build of PyTorch compatible with your device, install the other direct dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
# Select and install a CUDA build of PyTorch compatible with your device first
python -m pip install -r requirements.txt
python -c "import torch, transformers; print('torch:', torch.__version__, 'CUDA:', torch.cuda.is_available(), 'transformers:', transformers.__version__); assert torch.cuda.is_available()"
```

The diagnostic script loads the default model tokenizer in forced offline mode. Prepare the cache while online first, then run from this directory:

```bash
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen3-VL-4B-Instruct')"
python qwen4b-diag.py
python qwen4b-latency.py
```

`qwen4b-diag.py` does not load model weights, but loads the corresponding tokenizer offline; the tokenizer cache and dependencies must already be available. It checks that the template follows the text-only path.

The environment setup and tokenizer download commands above have not been end-to-end verified in a newly created environment. Record the actual installed versions and model revision used.

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

A single forward pass takes 49–52 ms, in the same range as Laya's 17 ms. But requiring a JSON object increases one complete decision from about 105 ms to 1,150–1,450 ms.

The difference between discriminative and generative models lies in cost structure, not just absolute speed. Generative cost grows with output length; a discriminative model generates no text, so its latency does not change with the number of output fields. This makes discriminative models suitable for high-throughput decision paths, but not for tasks that need generated explanations.

[中文](README.md)
