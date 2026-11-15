# Jev Series Experiment: metask-jev (4B)

This directory contains an on-device evaluation and reproduction setup for `wayfind/metask-jev-4b-policy-mix`.

The model is fine-tuned from Qwen3.5-4B. Its core technique constrains each candidate answer to a single token (A–Z), reads the corresponding output-layer logits directly, and normalizes them into probabilities. In a fresh-environment rerun on an NVIDIA GB10 on 2026-11-15, it scored 20/20 on the Chinese test set and used 8.65 GB peak CUDA memory. This result applies only to this small test set.

## Files and responsibilities

- `metask-install.sh`: Automated setup. Creates an isolated Python environment, downloads about 8.5 GB of model weights, and checks token-encoding integrity (A–Z must each be one token, preserving the decoding-mapping contract).
- `metask-bench.py`: Inference performance and task-accuracy benchmark covering the API contract, single-call latency, parallel English and Chinese ticket tests, temperature calibration effects, and memory monitoring.
- `metask-temperature.json`: Per-question-type confidence-calibration temperatures supplied upstream.
- `metask-jev_scorer.py` and `metask-jev_schema.py`: Archived copies of the upstream core scorer and schema-validation logic.

## Input validation without downloading the model

The schema-validation code used by the scorer depends only on the Python standard library, so input structure can be checked on CPU first. Run this from the current directory. It validates local input handling only; it does not load a model and cannot replace the GPU benchmark below.

```bash
python - <<'PY'
import importlib

schema_tools = importlib.import_module("metask-jev_schema")
schema = {
    "priority": {"type": "enum", "description": "ticket priority", "choices": ["low", "high"]},
    "urgent": {"type": "boolean", "description": "is urgent"},
}
schema_tools.validate_schema(schema)
assert schema_tools.choices_for(schema["urgent"]) == [False, True]
try:
    schema_tools.validate_schema({"priority": {"type": "enum", "description": "priority", "choices": []}})
except ValueError:
    print("valid schema accepted, empty choices rejected")
else:
    raise AssertionError("empty choices must fail")
PY
```

On 2026-11-13, this check ran in a newly created virtual environment on Linux ARM64 with Python 3.12.3 and exited with code 0: the valid schema passed and empty choices were rejected. No model dependencies were installed, and no model or performance benchmark was run.

## Deployment and reproduction

The script requires Linux, Bash, Python 3.12, network access, CUDA-enabled PyTorch, and about 8.5 GB of model storage; minimum memory is not verified. On 2026-11-15, installation and the GPU benchmark completed in a newly created environment on NVIDIA GB10 / Linux aarch64 with Python 3.12.3. By default, the installer creates or reuses a virtual environment in `~/metask-jev`; `METASK_DIR` can point it to an isolated location. The scorer is copied from the pinned source in this repository instead of downloading upstream `main`. The model is pinned to `wayfind/metask-jev-4b-policy-mix@ea20fe85b28733b1522721dec119bec50947a869`. Direct dependency versions are pinned in `requirements.txt`; install PyTorch separately for the target device, then a generated constraint prevents other packages from replacing it. `METASK_DIR` and `HF_HOME` control the installation and model-cache locations.

Run initialization and the benchmark in order from this directory:

```bash
export XL_DATA_DIR="$PWD/.work/data"
export XL_OUT_DIR="$PWD/.work/out/run-001"
mkdir -p "$XL_OUT_DIR"
export METASK_DIR="$XL_DATA_DIR/metask-jev"
export HF_HOME="$XL_DATA_DIR/huggingface"
export HF_ENDPOINT="https://hf-mirror.com"  # use https://huggingface.co if reachable directly
export METASK_MODEL_REVISION="ea20fe85b28733b1522721dec119bec50947a869"
export TORCH_REQUIREMENT="torch==2.15.0+cu132"
export TORCH_INDEX_URL="https://download.pytorch.org/whl/cu132"
bash metask-install.sh
"$METASK_DIR/.venv/bin/python" metask-bench.py | tee "$XL_OUT_DIR/benchmark.log"
```

The PyTorch command is for the NVIDIA GB10 / Linux aarch64 combination recorded here. For other devices, select an available build from the official PyTorch instructions and set `TORCH_REQUIREMENT` and `TORCH_INDEX_URL` accordingly. Set `XL_DATA_DIR`, `HF_HOME` and `METASK_DIR` to new isolated locations to avoid reusing another experiment's environment or cache. The installer activates its environment only in its child process, so the benchmark command explicitly uses that environment's interpreter. The installer creates a PyTorch constraint, checks dependency consistency, copies the pinned scorer code, downloads the specified model revision and runs a one-case GPU self-test; the benchmark output is saved to the log.

PyTorch `2.15.0+cu132` on Linux aarch64 pins NVIDIA `cuSPARSELt 0.8.1`. Its NVIDIA `manylinux2014_sbsa` wheel tag is reported by current pip's `pip check` as a platform mismatch. The installer accepts only this exact version, architecture, and error message, then verifies that the ARM library loads and CUDA is available. Other dependency conflicts still fail installation. The benchmark also warned that `causal_conv1d` and `flash-linear-attention` were not installed, so it used the slower reference PyTorch implementations; the latency results include this condition.

`requirements.txt` pins direct dependencies; pip resolves transitive dependencies. The clean installation, model-class import, one-case GPU self-test and benchmark were completed in this rerun.

## Measured metrics and data notes

Fresh-environment measurements from NVIDIA GB10 / Linux aarch64 with Python 3.12.3, PyTorch `2.15.0+cu132`, and Transformers `5.22.2`, on 2026-11-15:

- **Load and output:** Model loaded in 56.19 seconds; A–Z were each one token; the GPU self-test returned valid probabilities.
- **Single-call latency:** 120 English four-choice decisions had a 98.20 ms median, 99.33 ms P90, and 100.19 ms P99. Forty uncalibrated (`T=1`) calls had a 98.69 ms median.
- **English/Chinese classification:** 20 fixed examples per language; 19/20 English correct (95%) and 20/20 Chinese correct (100%). The respective P50 latencies were 98.9 ms and 97.7 ms.
- **GPU memory:** 8.55 GB allocated, 8.65 GB peak. Host available memory changed from 117.4 GB before loading to 107.3 GB after the test; minimum requirements remain unknown.
- **Test limits:** Only 20 examples per language. `causal_conv1d` and `flash-linear-attention` were absent, so the model used slower reference implementations. The temperature section shows confidence and entropy for four examples; it does not recompute the upstream ECE.

See the [public rerun summary](evidence/2026-11-15-gpu-rerun.json) for the reproducible run details.

**Upstream benchmark data:** We did not reproduce the ECE calibration error reported in the upstream README (0.114 to 0.040) or the JevBench ranking. The ranking is also an upstream interpolated self-evaluation, not an official evaluation. Use this repository's measurements for selection. Full trade-offs are in the [benchmark report](../../../docs/zh/decision-models/判别式决策模型基准实测.md).

[中文](README.md)
