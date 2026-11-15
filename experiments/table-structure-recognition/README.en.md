# Recovering Table Rows and Columns from Images: SLANet+ Measurements

[中文](README.md)

This directory contains the experiment code, configuration and result evidence. See the [full experiment document](../../docs/zh/document-understanding/SLANet表格结构识别.md) for the purpose, method, data, results and limitations.

[scripts](scripts/) · [fixtures](fixtures/) · [results](results/) · [figures](figures/)

## Reproduction

### 1. Predict table structure

The historical CPU run used aarch64, two PyTorch threads, PyTorch 2.14.0+cu130, Transformers 5.17.0, and Pillow (exact version not recorded). CPU prediction does not require a GPU. The environment record is not a complete dependency lockfile.

[Transformers 5.22.2 package metadata](https://pypi.org/project/transformers/5.22.2/) requires Python 3.11 or later, and PyTorch 2.6 or later when using the PyTorch backend. These are upstream dependency ranges, not a validation of the complete combination.

In an isolated Python virtual environment, first install matching `torch` and `torchvision` for the operating system and CPU or CUDA using the [PyTorch installation guide](https://pytorch.org/get-started/locally/), then install the other dependencies. The script also needs `/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf` on Linux; on Ubuntu install it with `sudo apt install fonts-dejavu-core`. On other systems, provide that font path or adapt the script. Linux example:

```bash
python -m venv .venv
source .venv/bin/activate
# Install matching torch and torchvision according to the PyTorch guide first.
python -m pip install -r requirements.txt
```

After installation, `python -m pip check` and `python -c "import torch, torchvision, transformers; from PIL import Image; print(torch.__version__, torchvision.__version__, transformers.__version__)"` check dependencies and imports only; they do not load the model.

The [requirements.txt](requirements.txt) pins the direct Transformers 5.22.2 and Pillow 12.0.0 dependencies used by this experiment, but excludes `torch` and `torchvision`, which must be installed for the target platform. The assembly and robustness scripts below use only the standard library and do not require this dependency list. Transitive dependencies are not locked. The inference script reads a local model and does not download it automatically. Obtain the complete fixed snapshot from the [SLANet+ model page](https://huggingface.co/PaddlePaddle/SLANet_plus_safetensors), including weights, configuration and image processor files:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('PaddlePaddle/SLANet_plus_safetensors', revision='e44d12b3fe695792170dd46623075f63d8544461', local_dir='.work/slanet-model')"
```

On 2026-11-13, one snapshot download attempt hit read timeouts and HTTP 503 before inference; see the [record for that attempt](evidence/2026-11-13-model-download-failure.json). A clean-environment CUDA rerun on 2026-11-15 successfully downloaded the pinned revision and verified the weight hash before inference. The two network outcomes are recorded separately; the earlier failure does not describe the current download result.

The `model.safetensors` SHA-256 for this revision is `4455fa2a535b80738e0e0af2ca899f719efb8d6e1c7e7209cc5126f81607bf06`. Run from this experiment directory. Both `--out` paths must not already exist:

```bash
python scripts/slanet_table_smoke.py --model .work/slanet-model --out .work/slanet-uncropped
python scripts/slanet_table_smoke.py --model .work/slanet-model --out .work/slanet-cropped --crop-table
```

On a CUDA-capable device, add `--device cuda` and use a new output directory. The output directory contains the input image and `result.json`; inspect `device`, `model_sha256`, and `result.structure`. For example:

```bash
python scripts/slanet_table_smoke.py --model .work/slanet-model --out .work/slanet-gpu --crop-table --device cuda
python -c "import json; d=json.load(open('.work/slanet-gpu/result.json')); s=d['result']['structure']; print('device:', d['device'], 'rows:', s.count('<tr>'), 'cells:', s.count('<td></td>'))"
```

On 2026-11-15, a new virtual environment on Linux aarch64 with an NVIDIA GB10 installed PyTorch 2.15.0+cu132, torchvision 0.29.1+cu132, Transformers 5.22.2 and Pillow 12.0.0; `pip check`, the pinned model download and its weight hash check all passed. CUDA inference on one cropped synthetic table returned 4 rows and 12 cells, with a structure score of 0.99996 and first-forward time of about 0.430 seconds. This verifies the clean-environment installation, download and structure-prediction path; it does not measure real-table accuracy or reproduce the complete evaluation. Rerun with fresh external data and output directories:

```bash
export XL_DATA_DIR=/path/to/slanet-data/run-001
export XL_OUT_DIR=/path/to/slanet-output/run-001
export XL_RUN_ID=run-001
export XL_REQUIRE_EXTERNAL_ENV=1
bash scripts/clean_env_cuda_smoke.sh
```

The script creates the virtual environment, cache and model snapshot under `XL_DATA_DIR`, saves results under `XL_OUT_DIR`, and refuses to overwrite an existing environment or output directory. See the [clean-environment CUDA smoke evidence](evidence/2026-11-15-clean-env-gpu-smoke.json). The original CPU run used Transformers 5.17.0 in a separate historical environment.

### 2. Align text with the predicted structure and evaluate robustness

The assembly step fills empty HTML slots in the SLANet prediction with text polygons from `fixtures/ocr-recognition.json`, using spatial clustering. This small fixture is a copy of recognition output from this repository's `paddle-ocr-detection-recognition` experiment; it does not depend on that other experiment directory. This step uses only the Python standard library. PyTorch and the SLANet+ model are not required. Run from this experiment directory:

```bash
# Run complete spatial-grid assembly
python scripts/assemble_table.py \
  --ocr-json fixtures/ocr-recognition.json \
  --structure-json results/cropped.json \
  --out-dir .work/assembly

# Run the robustness stress test and cascading-error benchmark
python scripts/eval_table_reconstruction.py --output .work/assembly-robustness.json
```

On 2026-11-14, both standard-library scripts ran in a newly created isolated Python 3.12 environment. The sample assembly produced 4 rows and 12 cells; the corrected checks detected incomplete filling and structural mismatch for both a missing text item and an extra structure row. This is a single repository fixture, not an accuracy evaluation on real tables.

Historical results are in `results/`. These commands write new outputs under the Git-ignored `.work/`; they refuse to overwrite existing files, so use new output paths for reruns. The historical `assembly_robustness_benchmark.json` did not check remaining empty slots in `all_slots_filled`, nor structure predictions in `structure_exact`; it cannot determine whether text is missing or extra structure rows were produced. The corrected checks on a new result use row counts and empty-slot counts from this sample without merged cells. This is still not a general measure of table-structure equivalence.
## Evidence

- [2026-11-15 clean-environment CUDA smoke](evidence/2026-11-15-clean-env-gpu-smoke.json): pinned model and input hashes, package versions, structure score, and single-run forward time.
- [2026-11-13 GB10 synthetic-table CUDA smoke](evidence/2026-11-13-gpu-synthetic-smoke.json): earlier layered-environment record.
- [2026-11-13 model download failure](evidence/2026-11-13-model-download-failure.json): one fixed-snapshot retrieval stopped before inference after timeouts and HTTP 503; a later successful inference used files already available on the device.
