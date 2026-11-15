# Jev Series Experiment: metask-jev (4B)

This directory contains an on-device evaluation and reproduction setup for `wayfind/metask-jev-4b-policy-mix`.

The model is fine-tuned from Qwen3.5-4B. Its core technique constrains each candidate answer to a single token (A–Z), reads the corresponding output-layer logits directly, and normalizes them into probabilities. It scored 20/20 on the Chinese classification test set, the only perfect score among the four approaches, with 8.58 GB resident memory.

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

The script requires Bash, Python 3, curl, network access, and enough storage for the model. Historical performance testing used CUDA. The install script creates or reuses a virtual environment in `~/metask-jev`, installs and upgrades packages, downloads the scorer from upstream `main`, downloads about 8.5 GB of model weights, and runs one inference. Direct dependencies other than PyTorch are listed in `requirements.txt`; most versions, as well as upstream code and model revisions, are not pinned, so the same environment cannot be guaranteed. On Linux, the installer first tries the CUDA 12.1 PyTorch download source; check compatibility with your device. Inspect an existing directory of the same name before using it. This installation path has not been verified end to end in a fresh environment.

Run initialization and the benchmark in order from this directory:

```bash
bash metask-install.sh
"$HOME/metask-jev/.venv/bin/python" metask-bench.py
```

**Runtime dependency:** `metask-bench.py` directly uses the scorer module initialized by the install script under `~/metask-jev`. The install script activates the virtual environment in its child process, which does not change the current shell; therefore the benchmark command explicitly uses that environment's interpreter. Run `metask-install.sh` completely to initialize the environment and weights before starting the benchmark.

## Measured metrics and data notes

Measurements from the tested local environment:

- **End-to-end latency:** Median single classification decision of 133.5 ms (English) and 133.7 ms (Chinese).
- **Task accuracy:** 19 of 20 English tickets correct; all 20 Chinese tickets correct.
- **Memory:** 8.58 GB resident.

**Upstream benchmark data:** We did not reproduce the ECE calibration error reported in the upstream README (0.114 to 0.040) or the JevBench ranking. The ranking is also an upstream interpolated self-evaluation, not an official evaluation. Use this repository's measurements for selection. Full trade-offs are in the [benchmark report](../../../docs/zh/decision-models/判别式决策模型基准实测.md).

[中文](README.md)
