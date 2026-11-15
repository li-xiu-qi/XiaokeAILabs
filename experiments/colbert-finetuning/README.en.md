# Fine-Tuning ColBERT for Scientific Literature Retrieval

This directory contains the experiment code, configuration, and evidence. See the [full experiment guide](../../docs/zh/retrieval/colbert/文本多向量检索模型微调.md) for the purpose, method, data, results, and reproduction steps. Unless the guide says otherwise, run its commands from this directory.

[scripts](scripts/) · [config](config/) · [evidence](evidence/) · [assets](assets/)

## Environment

The historical run used Linux ARM64 and CUDA on DGX Spark. It requires an NVIDIA GPU with a compatible CUDA driver and PyTorch build, plus enough disk for the model, data, and outputs. The data preparation script uses `hashlib.file_digest` and requires Python 3.11 or later; the recorded NumPy and SciPy requirements need Python 3.12 or later. No measured minimum VRAM is available. Dependencies include PyLate, sentence-transformers, Transformers, datasets, and accelerate; see [environment-freeze.txt](evidence/environment-freeze.txt) for the recorded environment. PyLate's full dependencies have installation limitations on Linux ARM64. This experiment uses model, training, and reranking functions only, so install and verify dependencies for the target platform.

Create an isolated environment and install the PyTorch build matching the device and CUDA driver using the [PyTorch installation guide](https://pytorch.org/get-started/locally/):

```bash
python3 -m venv .venv
source .venv/bin/activate
# Install matching CUDA-enabled PyTorch here
python -c "import torch; print('torch:', torch.__version__, 'CUDA:', torch.version.cuda, 'available:', torch.cuda.is_available())"
```

Train only if `available: True`. Do not replace matching CUDA PyTorch with a generic build. Then pin its version and install the full PyLate requirements:

```bash
python -c "import importlib.metadata as m; print('torch==' + m.version('torch'))" > torch-constraint.txt
python -m pip install -c torch-constraint.txt -r requirements.txt
python -m pip check
python -c "import torch, numpy, datasets, sentence_transformers, pylate"
```

The constraint prevents pip from replacing the installed PyTorch. These versions are not a full lock file; a clean install of PyLate 1.4.0 with its complete dependencies remains unverified. Passing checks does not mean the original experiment has been reproduced.

PyLate 1.4.0 requires `fast-plaid`, even though this experiment does not build a PLAID index. Supported PyPI versions provide no Linux ARM64 wheel or source package and constrain PyTorch; the historical PyTorch 2.14.0+cu130 does not meet those constraints. If full installation fails, fixed-candidate training and reranking may be tried with this **incomplete dependency fallback**:

```bash
python -m pip install -c torch-constraint.txt -r requirements-base.txt
python -m pip install --no-deps pylate==1.4.0
python -m pip check
python -c "import torch, numpy, datasets, sentence_transformers, pylate"
```

The fallback omits fast-plaid; `pip check` reports it missing and may report other conflicts. Inspect the complete output. This does not meet all package metadata dependencies and cannot establish PLAID support. PyLate 1.4.0 declares fast-plaid as required, but the supported PyPI version range has no Linux ARM64 distribution. On 2026-11-15, a fresh Linux ARM64 environment installed `requirements-base.txt` and PyLate 1.4.0 with `--no-deps`; `pip check` reported only fast-plaid missing. Imports for the model, loss, training utilities, and reranker passed, as did two CUDA training steps and reranking before and after training on synthetic data. The check did not build or query a PLAID index and does not reproduce the original result. See the [PyLate 1.4.0 ARM64 training smoke record](evidence/pylate-1.4-arm64-training-smoke-2026-11-15.json).

To repeat this synthetic check on Linux ARM64 with Python 3.12 and an NVIDIA GPU, set output locations and run the script. Its default installation matches the tested GB10 setup (PyTorch 2.15.0+cu132); set `XL_TORCH_SPEC` and `XL_TORCH_INDEX_URL` to use a PyTorch build suitable for another device.

```bash
export XL_RUN_ID="$(python3 -c 'import uuid; print(uuid.uuid4().hex)')"
export XL_DATA_DIR="$PWD/.work/data"
export XL_OUT_DIR="$PWD/.work/out/$XL_RUN_ID"
# Override these for a different CUDA/PyTorch build if needed.
# export XL_TORCH_SPEC='torch==<compatible-version>'
# export XL_TORCH_INDEX_URL='https://download.pytorch.org/whl/<cuda-version>'
bash scripts/verify_arm64_training_install.sh
```

The script creates an isolated environment, installs dependencies, checks imports and runs the synthetic GPU smoke. `pip check` must report only the missing fast-plaid dependency; the script stops if any other conflict appears. Do not use this incomplete dependency path for PLAID indexes.

### ARM64 synthetic GPU smoke test

PyLate 1.2.0 can check the training path if 1.4.0's full dependencies cannot be installed. It has no fast-plaid dependency, but its loss differs from 1.4.0, so it **cannot reproduce or compare the experiment's conclusions**. In Python 3.12, install device-compatible CUDA PyTorch and run:

```bash
python -c "import importlib.metadata as m; print('torch==' + m.version('torch'))" > torch-constraint.txt
python -m pip install -c torch-constraint.txt 'pylate==1.2.0' 'datasets==4.1.1' 'accelerate==1.10.1' attrs pillow
python -m pip check
python scripts/synthetic_gpu_smoke.py
```

The script creates a randomly initialized tiny BERT and synthetic data, then runs two CUDA training steps and reranks candidates before and after training. It downloads no original model or data, and its output says nothing about retrieval quality. On 2026-11-15, this path was installed from an empty Python 3.12 venv on Linux ARM64 / NVIDIA GB10: PyTorch 2.15.0+cu132, PyLate 1.2.0, Datasets 4.1.1, Accelerate 1.10.1, sentence-transformers 4.0.2, and Transformers 4.48.2. `pip check` passed, and the script completed both GPU training steps and reranking. See the [clean-environment GPU smoke record](evidence/clean-env-gpu-smoke-2026-11-15.json) for versions and scope. This validates the fallback version's installation and training path; it does not reproduce the original result. PyLate 1.4.0 also passed the same synthetic check with fast-plaid omitted; see above.

## Original data and training

Copy this directory to a working directory and run there. Download BEIR SciFact and the original SciFact claims file, checking the expected hashes:

```bash
mkdir -p output
curl -fL https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip -o output/scifact.zip
curl -fL https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz -o output/scifact-original.tar.gz
sha256sum -c <<'EOF'
536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165  output/scifact.zip
11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be  output/scifact-original.tar.gz
EOF
```

Expected SHA-256 values, in order: `536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165` and `11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be`.

```bash
unzip -q output/scifact.zip -d output
tar -xOzf output/scifact-original.tar.gz data/claims_train.jsonl > output/claims_train.jsonl
python scripts/prepare_scifact.py --data output/scifact --original-claims output/claims_train.jsonl --config config/data.json --out output/prepared
huggingface-cli download lightonai/colbertv2.0 --revision 0b9b1e380096a3b72b6b4877f68d59effb62e0f2 --local-dir output/model
sha256sum output/prepared/pools.json evidence/training-pairs.jsonl output/model/model.safetensors output/model/1_Dense/model.safetensors output/model/1_Dense/config.json
```

Compare hashes with [input checksums](evidence/input-sha256.txt). Training pairs contain query and document IDs, not source text. Download the complete model, including `1_Dense/`. The model is loaded offline by the scripts.

```bash
python scripts/run.py --data output/scifact --prepared output/prepared --pairs evidence/training-pairs.jsonl --model output/model --out output/smoke --smoke
python scripts/run.py --data output/scifact --prepared output/prepared --pairs evidence/training-pairs.jsonl --model output/model --out output/formal
python scripts/analyze.py --results output/formal --pairs evidence/training-pairs.jsonl
```

Do not add models, data, or training output to version control. Parameterized public scripts completed a two-step smoke run and full rerun on the original device; the rerun reproduced the same summary metrics. Check environment and input hashes when reproducing. Changed sampling, model, or parameters make results incomparable to this run.

## Recompute metrics and rerank

Recompute saved per-query metrics and bootstrap intervals without downloading a model. This uses a temporary directory to avoid overwriting repository evidence:

```bash
tmp_dir=$(mktemp -d)
cp evidence/{before,after}-per-query.json "$tmp_dir"/
python scripts/analyze.py --results "$tmp_dir" --pairs evidence/training-pairs.jsonl
```

This does not rerun the model, recompute candidate scores, or verify rankings. Ranking checks require the recorded candidate IDs, relevant IDs, and rankings.

Use the formally trained model to rerank candidates:

```python
from pylate import models, rank
model = models.ColBERT(model_name_or_path="output/formal/fine-tuned", device="cuda", query_length=32, document_length=180)
query = ["Does the treatment affect the outcome?"]
documents = [["A study of the treatment and outcome.", "An unrelated study."]]
query_vectors = model.encode(query, is_query=True, convert_to_numpy=True)
document_vectors = model.encode(documents[0], is_query=False, convert_to_numpy=True)
results = rank.rerank(documents_ids=[["doc-1", "doc-2"]], queries_embeddings=query_vectors, documents_embeddings=[document_vectors], device="cuda")
print(results[0])
```

The first training run saved weights despite a failed model-card network request; the weights were then reloaded offline for evaluation. After offline mode was added, training, saving, and evaluation completed in a full rerun. Conclusions apply only to this English scientific-literature task. Chinese, full-document retrieval, end-to-end index construction, and online latency have not been tested.

The separate `scripts/benchmark_residual_compression.py` uses synthetic vectors. Its inverted-index centroid assignments are random and do not correspond to the document vectors used for scoring; it extrapolates brute-force timing from 100 documents and computes at most 100 pruned candidates. Its output therefore does not establish actual retrieval recall or speedup. The 0.982 recall in older output was hard-coded rather than measured; it has been removed and recall is now marked as unmeasured. The script refuses to overwrite an existing `evidence/results_residual_compression.json`; run it in a copy to preserve existing evidence.

[中文说明](README.md)
