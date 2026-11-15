# LLM Inference Cost: From Memory Budgets to Performance Estimates

This directory contains experiment code, configurations, and result evidence. See the [full experiment document](../../docs/zh/inference/推理内存预算与成本估算.md) for its purpose, principles, data, results, and boundaries.

[scripts](scripts/) · [records](records/) · [evidence](evidence/)

Run all commands below from this experiment directory. See the [Chinese README](README.md).

## Environment setup

The basic examples run on systems with Python 3.10 or later. They do not depend on a particular CPU architecture or GPU and require no third-party packages, models, or datasets. Enter this experiment directory and check the interpreter. A virtual environment is optional:

```bash
cd experiments/inference-cost-models
python --version
python -m venv .venv
source .venv/bin/activate  # Linux / macOS; on Windows use .venv\Scripts\Activate.ps1
```

HTTP timing requires a network-accessible inference service and the third-party `requests` package; installation is shown separately below. `requirements.txt` pins `requests==2.34.2`, while transitive dependencies are resolved by pip. Timing results depend on the service and runtime conditions.

## Runnable examples

Python 3.10+ is required. The example below computes only the theoretical operation count for the two matrix multiplications in dense, non-causal full attention and the size of an attention-score matrix under a single-layer materialization assumption. It excludes other layers in hybrid architectures and does not predict actual latency or peak memory.

```bash
python scripts/attention_model.py
```

The memory example can also be run with `python scripts/hybrid_architecture_model.py`. These two examples use only the Python standard library and repository modules; they need no model weights, dataset, or GPU. The memory example excludes other layer state and runtime overhead.

Request-level HTTP timing requires installing `requests` separately and preparing a model service that supports `/v1/completions`:

```bash
python -m pip install -r requirements.txt
python scripts/compare_engines.py --help
python -m unittest discover -s tests -v
```

Actual timing requires a service URL, model name, and prompt. Results include queueing, prompt processing, generation, and network time; they do not break out inference stages.

The basic examples need no third-party packages. The HTTP client and offline checks use `requests==2.34.2`, listed in [requirements.txt](requirements.txt). Network requests in tests are mocked, so no inference service is needed for tests.

On 2026-11-13, in a separate environment on Windows 11 / WSL2 Ubuntu 24.04 ARM64 with Python 3.12.3, `requests 2.34.2` was installed from the requirements file and `pip check` passed. Both basic examples, the HTTP script's `--help`, and five offline tests ran successfully. No inference service was connected, and GPU benchmarks and historical values were not verified.

On 2026-11-15, a new virtual environment on Linux aarch64 installed the pinned `requests==2.34.2`; `pip check` passed. Both basic examples and the HTTP script's `--help` completed. No inference service was queried and no GPU benchmark was run. See the [environment-check summary](evidence/clean-env-check-2026-11-15.json).

The historical `scripts/bench_embedding_spark.py` is a separate GPU/model benchmark draft. It requires CUDA PyTorch, NVIDIA drivers, `nvidia-smi`, `sentence-transformers` from [requirements-gpu.txt](requirements-gpu.txt), and the corresponding model from the script's `MODELS`. First load may download models. Install a device-compatible version according to the [PyTorch installation instructions](https://pytorch.org/get-started/locally/), then install the optional dependency with `python -m pip install -r requirements-gpu.txt`. The script uses synthetic repeated text and writes results to `emb_bench_*.json` in the current user's home directory, overwriting a same-named file. This is not one of the lightweight examples above. Dependency versions and measurement validity are not fully pinned; read [Usage Boundaries and Audit](records/使用边界与审计.md) before running it.

The experiment document linked above explains the principles. Other existing scripts are research drafts whose parameters and outputs still need individual audits; read [Usage Boundaries and Audit](records/使用边界与审计.md) before running them.

## Audit and historical records

Historical scripts and reports in this experiment contain useful derivation clues, but a code review on 2026-11-07 found that they are not yet reliable for performance prediction across hardware or workloads. Read the audit before citing any figures.

| Document | Contents |
|---|---|
| [Usage Boundaries and Audit](records/使用边界与审计.md) | Script scope and discovered issues; this page is authoritative for historical conclusions |
| [LLM-Viewer Cross-Validation](records/LLM-Viewer交叉验证.md) | Historical comparison from 2026-09-13, including deployment commands and pitfalls; conclusions have known issues |
| [Iteration Log](records/迭代日志.md) | Process record of model development |
| [vLLM deployment on GB10](records/vLLM在GB10上的部署.md) · [SGLang deployment on GB10](records/SGLang在GB10上的部署.md) | Deployment steps and troubleshooting for the two engines |
