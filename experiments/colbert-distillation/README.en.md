# ColBERT distillation: code and reproduction

Historical results must be interpreted under their original run conditions. The current scripts add a teacher-advantage gate on a separate development set: training proceeds only when the teacher's nDCG@10 exceeds the initial student's by at least 0.05 and the lower bound of the paired per-query bootstrap 95% interval is above zero. The test set is evaluated after both training runs. This is a setting for this experiment, not a general threshold. The updated training and test workflow has completed once with real models in an incomplete-dependency fallback environment; see below for its scope.

[中文](README.md)

This experiment tests whether candidate scores from a single-vector Qwen3-Embedding-8B teacher can improve the ranking of a multi-vector ColBERT student, against a label-supervised control with the same student, training examples, and budget. The teacher only generates training targets; the deployed student does not need it. Distillation is itself training, so teacher scores are not human ground truth. The [full experiment guide](../../docs/en/colbert/colbert-distillation.md) covers the motivation, method, data, and interpretation of results.

## Reproduction

### Environment

The historical run used Linux ARM64 and CUDA on DGX Spark. The current full workflow requires Linux `/proc/meminfo`, trains on CUDA, and uses BF16 for teacher scoring. Use an NVIDIA GPU with CUDA and BF16 support, compatible drivers and PyTorch, and sufficient GPU memory and system RAM. During teacher CUDA scoring and formal training, the scripts require at least 50 GiB of currently available system memory. This is a run-protection threshold, not a generally established hardware minimum. The 8B teacher weights, data, and outputs also require disk space.

In an isolated environment, use Python 3.12 or newer with this dependency set. The data-preparation script itself requires Python 3.11 or newer; the recorded NumPy and SciPy versions require Python 3.12 or newer. Install a PyTorch build compatible with the device and CUDA driver before other packages, and confirm `torch.cuda.is_available()` is true:

```bash
python3 -m venv .venv
source .venv/bin/activate
# Install CUDA PyTorch for this device: https://pytorch.org/get-started/locally/
python -c "import importlib.metadata as m; print('torch==' + m.version('torch'))" > torch-constraint.txt
python -m pip install -c torch-constraint.txt -r requirements.txt
python -m pip check
python -c "import torch, numpy, sentence_transformers, pylate; print('torch:', torch.__version__, 'CUDA:', torch.version.cuda, 'available:', torch.cuda.is_available()); print('numpy:', numpy.__version__)"
```

The constraint file pins the already installed PyTorch version so pip does not replace its CUDA build while resolving dependencies. On 2026-11-15 a from-scratch install was validated in a fresh Linux ARM64 virtual environment: with PyTorch 2.15.0+cu132, every pinned version in `requirements-base.txt` resolves and installs, and the previously unpinned pandas, sqlitedict, and voyager resolved to 3.1.1, 2.1.0, and 2.2.3, now pinned accordingly. The file is not a complete lockfile; remaining transitive dependencies still vary with resolution. Passing `pip check` and import checks does not establish that real-model execution works.

PyLate 1.4.0 declares `fast-plaid` as a required package, even though this experiment only reranks a fixed candidate set. A 2026-11-15 probe confirmed that the default index offers no fast-plaid distribution for Linux ARM64 (CPython 3.12): neither a wheel nor an sdist is installable (pip reports `from versions: none`). The versions also constrain PyTorch; for example, PyTorch 2.14.0+cu130 from the historical run does not meet those constraints. A normal install failure will not automatically downgrade PyTorch. This experiment does not build a PLAID index. On Linux ARM64, the following **incomplete dependency fallback** can be tried:

```bash
python -m pip install -c torch-constraint.txt -r requirements-base.txt
python -m pip install --no-deps pylate==1.4.0
python -m pip check
python -c "import torch, numpy, sentence_transformers, pylate; print(torch.__version__, torch.cuda.is_available())"
```

This fallback omits `fast-plaid`; `pip check` will report it missing and may report other dependency conflicts. Review all output. It is only for investigating this experiment's fixed-candidate path, does not satisfy PyLate's complete dependencies, and does not make index functionality available or establish a successful reproduction in a new environment. On 2026-11-15 this fallback was executed from scratch in a fresh Linux ARM64 virtual environment: the install succeeded, `pip check` reported only `pylate 1.4.0 requires fast-plaid, which is not installed`, and `torch`, `numpy`, `sentence_transformers`, and `pylate` imported with CUDA visible (see [env-validation-20261115.json](evidence/env-validation-20261115.json)). The historical device's PyLate path and the 2026-11-14 import check remain subject to their original conditions.

On 2026-11-14, the fallback was installed in an isolated copied Linux ARM64 environment. PyTorch 2.15.0+cu132, Transformers 4.56.2, Sentence Transformers 5.1.1 and PyLate 1.4.0 imported, and CUDA was visible; `pip check` reported the missing `fast-plaid`. This is an import check with incomplete dependencies, not a real-model training or clean-install validation.

On the same day, this fallback environment loaded the ColBERTv2 student at revision `0b9b1e380096a3b72b6b4877f68d59effb62e0f2` and encoded and reranked two candidates for one example query on CUDA. This checks only fixed-candidate reranking. `fast-plaid` is still missing, so it does not establish a complete dependency installation or formal training.

On 2026-11-14, a new SciFact run used pinned Qwen3-Embedding-8B teacher and ColBERTv2 student models. On 89 development queries, teacher nDCG@10 was 0.7171 and student nDCG@10 was 0.6434; the 0.0737 gap had a paired bootstrap 95% interval of [0.0232, 0.1318] and passed the gate. Supervised and distilled students each trained for 128 steps. On 300 test queries, baseline, supervised, and distilled nDCG@10 were 0.6536, 0.6686, and 0.6657; distilled minus supervised was -0.0029 with a paired bootstrap 95% interval of [-0.0111, 0.0051], showing no stable advantage. Test query IDs did not overlap with training or development IDs, but some test-positive documents appeared in training candidates. Results cover this fixed candidate pool, one training seed, and this device. The student ran in the fallback environment without fast-plaid, so this does not establish a complete PyLate installation or usable indexing.

### GPU training check with synthetic data

To check only whether the distillation training function can use a GPU, PyLate 1.2.0 can be tried in a separate environment. It does not have PyLate 1.4.0's required `fast-plaid` dependency, but its loss implementation differs between the two versions. This check cannot reproduce the 1.4.0 experiment results or resolve the 1.4.0 ARM64 dependency issue.

From the experiment directory, install CUDA PyTorch appropriate for the device first, then install the remaining dependencies. On Linux ARM64 with CUDA 13 the install command below is the verified one; on other devices choose the matching build from the [official PyTorch guide](https://pytorch.org/get-started/locally/) and constrain that version:

```bash
python3.12 -m venv .venv-smoke
source .venv-smoke/bin/activate
python -m pip install 'torch==2.15.0' --index-url https://download.pytorch.org/whl/cu132
python -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available()"
python -c "import importlib.metadata as m; print('torch==' + m.version('torch'))" > .venv-smoke/torch-constraint.txt
python -m pip install -c .venv-smoke/torch-constraint.txt -r requirements-smoke.txt
python -m pip check
python scripts/gpu_synthetic_smoke.py
```

If the device presets proxy environment variables, confirm the proxy is reachable or clear those variables first; otherwise pip fails immediately during connection.

The script initializes a tiny BERT in a temporary directory, creates four queries, candidate documents, and teacher scores, then calls the distillation training function from `run.py` for two CUDA training steps. It removes the temporary model afterward. The script retains the 50 GiB available-memory guard in that function. It does not download a model or data, score with the teacher, run the formal gate or supervised control, or evaluate the test set.

This synthetic training check ran on the Linux ARM64 DGX Spark GPU with Python 3.12, PyTorch 2.15.0+cu132, and PyLate 1.2.0 on 2026-11-14, reusing dependencies from an existing PyTorch environment in an isolated virtual environment, and `pip check` reported no dependency conflicts. On 2026-11-15 the install commands above were executed from scratch in a fresh virtual environment and the script was rerun: `pip check` reported `No broken requirements found`, and the script completed two CUDA training steps (first-step loss 1.922, 0.35 s). The resolved transitive versions are recorded in [env-validation-20261115.json](evidence/env-validation-20261115.json).

### Data and models

Run these commands from the experiment directory. The files go in a personal working directory; replace paths as needed. Downloading and extraction require `curl`, `unzip`, `tar`, and network access. The Python data-preparation script itself uses only the standard library. Pass the model directories shown below to the commands' `--model` options. Download each model in full; the student must include its `1_Dense/` projection layer because the scripts load it from local files:

```bash
mkdir -p output
curl -fL https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip -o output/scifact.zip
curl -fL https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz -o output/scifact-original.tar.gz
unzip -q output/scifact.zip -d output
tar -xOzf output/scifact-original.tar.gz data/claims_train.jsonl > output/claims_train.jsonl
huggingface-cli download Qwen/Qwen3-Embedding-8B --local-dir output/teacher
huggingface-cli download lightonai/colbertv2.0 --revision 0b9b1e380096a3b72b6b4877f68d59effb62e0f2 --local-dir output/student
```

The teacher is [Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B), and the student is [ColBERTv2](https://huggingface.co/lightonai/colbertv2.0). The student revision above is a snapshot used in a related experiment; the teacher command does not pin a revision. **The model revisions and weight hashes used in the original distillation run have not been confirmed.** Record the actual downloaded revisions and hashes when running the experiment. Data processing and licenses are described in the [experiment guide](../../docs/en/colbert/colbert-distillation.md).

After downloading, calculate source-file hashes and compare each matching entry with [source-hashes.json](configs/source-hashes.json). The original claims file corresponds to `original_claims_train.jsonl`. If a hash differs, check the source and version first. This file records hashes for the data and candidate pool, not model weights:

```bash
sha256sum output/scifact/corpus.jsonl output/scifact/queries.jsonl \
  output/scifact/qrels/train.tsv output/scifact/qrels/test.tsv output/claims_train.jsonl
# Compare the pools.json entry separately after preparing the candidate pool
sha256sum /path/to/prepared/pools.json
```

Copy the experiment directory to a working directory and run the following commands in that copy, replacing paths with your data, models, and output locations:

```bash
python scripts/prepare_pools.py --data /path/to/scifact --original-claims /path/to/claims_train.jsonl --config configs/pool_config.json --out /path/to/prepared
python scripts/prepare_teacher.py --data /path/to/scifact --pools /path/to/prepared/pools.json --model /path/to/Qwen3-Embedding-8B --device cuda --out /path/to/source --min-teacher-gap 0.05
python scripts/run.py --data /path/to/scifact --pools /path/to/prepared/pools.json --source /path/to/source --model /path/to/colbert --output /path/to/output --min-teacher-gap 0.05 --smoke
```

The first command prepares the candidate pool. The second generates teacher scores for training and development data only and records the predeclared threshold. The third checks inputs and the gate configuration. Remove `--smoke` to compare teacher and student on the development set. The supervised and distilled groups train only if the gate passes; the test set is evaluated after both runs. The teacher output directory must be empty, and the training threshold must match the one declared before teacher scoring. `prepare_teacher.py --synthetic-encoder` uses pseudo-vectors and is suitable only for checking the CPU interface; it cannot be used for formal training. The 50 GiB available-memory minimum in `run.py` is a guard from the original device, not a universal hardware requirement.

**Reproduction scope:** the candidate pool was regenerated on CPU, and its `pools.json` SHA-256 matches the original. The new gate changes the historical script path, so it is not a step-by-step reproduction of the historical results. A new run completed real 8B teacher scoring, training, and testing; the student environment lacks fast-plaid, and the exact environment of the original distillation run has not been confirmed. On 2026-11-15 both dependency installs and the synthetic-data GPU smoke check were additionally validated from scratch in fresh isolated environments, without downloading model weights or benchmark data; scope and limits are in [env-validation-20261115.json](evidence/env-validation-20261115.json).

## Files

- [prepare_pools.py](scripts/prepare_pools.py) and [pool_config.json](configs/pool_config.json): prepare and configure the candidate pool.
- [prepare_teacher.py](scripts/prepare_teacher.py): generate training candidate and teacher scores.
- [run.py](scripts/run.py): train the student and evaluate it.
- [gpu_synthetic_smoke.py](scripts/gpu_synthetic_smoke.py): check the CUDA distillation training function with synthetic data.
- [summary.json](evidence/summary.json) and [per-query-metrics.csv](evidence/per-query-metrics.csv): aggregate results and per-query metrics; [recompute_metrics.py](scripts/recompute_metrics.py) recomputes metrics.
- [requirements-smoke.txt](requirements-smoke.txt): dependency list for the synthetic-data GPU smoke check; [env-validation-20261115.json](evidence/env-validation-20261115.json): record of the 2026-11-15 from-scratch install and smoke validation.

Per-query metrics do not contain the original document rankings, so they are insufficient to re-evaluate each query's scores independently.
