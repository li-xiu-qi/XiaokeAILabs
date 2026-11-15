# Detecting and Recognizing Table Text with PP-OCRv5

[中文](README.md)

This directory contains the experiment code, configuration, and result evidence. See the [full experiment guide](../../docs/zh/document-understanding/PaddleOCR文字检测与识别.md) for the purpose, method, data, results, and limitations.

[scripts](scripts/) · [results](results/) · [figures](figures/)

## Reproduction

The pipeline first detects text regions, then recognizes text in each detected box. It does not predict table rows or columns. CPU execution does not require a GPU. The original run recorded PyTorch 2.14.0+cu130, Transformers 5.17.0, OpenCV 5.0.0, and Pillow; this is not a complete dependency lock file.

### Set up the environment

The [Transformers 5.17.0 package page](https://pypi.org/project/transformers/5.17.0/) requires Python 3.10 or later and PyTorch 2.5 or later when using its PyTorch backend. These are upstream dependency ranges; they do not mean that the complete combination has been validated.

Create an isolated Python virtual environment. First install matching versions of `torch` and `torchvision` using the [PyTorch installation guide](https://pytorch.org/get-started/locally/), choosing the operating system and CPU or CUDA configuration, then install the remaining dependencies. For example, on Linux or macOS:

```bash
python -m venv .venv
source .venv/bin/activate
# Install matching torch and torchvision according to the PyTorch guide first:
python -m pip install -r requirements.txt
```

After installation, `python -m pip check` and `python -c "import torch, torchvision, transformers, cv2; from PIL import Image; print(torch.__version__, torchvision.__version__, transformers.__version__)"` check dependencies and imports only; they do not load a model.

The [requirements.txt](requirements.txt) pins the Transformers, Pillow and OpenCV distribution versions, but excludes platform-specific `torch` and `torchvision` and is not a complete lock file. The OpenCV Python distribution is pinned to `5.0.0.93`; after import, `cv2.__version__` reports `5.0.0`—these are different version fields ([PyPI release page](https://pypi.org/project/opencv-python/5.0.0.93/)). The first clean-install attempt on 2026-11-15 used the runtime `cv2` version as the distribution pin; after correcting it, a new empty venv completed installation and GPU inference. That run used existing pip wheel caches and verified model files on the device; it did not test downloads from an empty device cache.

To install from an empty environment and run the synthetic-table CUDA sample on Linux ARM64 / GB10, use this script. It creates a separate virtual environment, downloads fixed model revisions, checks `pip check` and CUDA, and requires all 12 recognized strings to match the expected sample output. Data, models and results go under `.work/`; use a new run ID each time to avoid overwriting earlier files:

```bash
export XL_DATA_DIR="$PWD/.work/data"
export XL_OUT_DIR="$PWD/.work/run-$(date +%Y%m%d-%H%M%S)"
export XL_RUN_ID="$(date +%Y%m%d-%H%M%S)"
bash scripts/run_clean_gpu_reproduction.sh
```

The script pins PyTorch `2.15.0+cu132` and torchvision `0.29.1+cu130` for the recorded GB10 Linux ARM64 setup. On another system or GPU, install builds matched to that device and use the general commands below. On success, it writes `ppocr_metrics.json` and a `summary.json` containing environment, model and input hashes. If model files already exist, set `PP_OCR_MODEL_SOURCE_DIR` to a directory containing `det/` and `rec/`; the script copies their configuration and weights and verifies the weight hashes. If unset, it downloads the fixed revisions from Hugging Face.

### Run

The script reads local models and does not download them automatically. Run the following from the experiment directory to download fixed revisions of the [detection model](https://huggingface.co/PaddlePaddle/PP-OCRv5_server_det_safetensors) and [recognition model](https://huggingface.co/PaddlePaddle/PP-OCRv5_server_rec_safetensors) into `models/`. The download library is a dependency of Transformers:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download('PaddlePaddle/PP-OCRv5_server_det_safetensors', revision='cbea9f3c3254c6ff7b0016cfbf90549e1ad4c5bb', local_dir='models/det')
snapshot_download('PaddlePaddle/PP-OCRv5_server_rec_safetensors', revision='542979d7cc3791732bb12af35313a6840952d79f', local_dir='models/rec')
PY
```

Run detection and recognition:

```bash
python scripts/run.py \
  --image figures/synthetic-table.png \
  --det-model models/det --rec-model models/rec \
  --det-device cpu --rec-device cpu --threads 2 \
  --output /tmp/paddle-ocr-result.json
```

The output JSON file must not already exist. For CUDA, change the device arguments to `--det-device cuda --rec-device cuda` and install PyTorch and torchvision versions that support the GPU. Inspect the recognized text with:

```bash
python -c "import json; x=json.load(open('/tmp/paddle-ocr-result.json', encoding='utf-8')); print(len(x), [r['prediction']['text'] for r in x])"
```

On 2026-11-13, a CUDA run on Linux aarch64 and NVIDIA GB10 with the fixed model revisions detected 12 text regions in the synthetic image; the recognized text matched `results/cpu-reproduction.json`. The run used Python 3.12, PyTorch 2.15.0+cu132, torchvision 0.29.1+cu130, Transformers 5.17.0, OpenCV 5.0.0 and Pillow 13.0.0. It used layered dependencies, so a complete clean installation remains unverified. The configuration and weight hashes matched the fixed revisions above, but the original run recorded neither revision nor weight hashes, so the original snapshot cannot be confirmed identical. One synthetic image does not establish accuracy on real documents.

On 2026-11-14, an isolated copied environment using the versions above passed `pip check` and processed a screenshot from the first page of a [fixed upstream PDF revision](https://github.com/RapidAI/RapidOCRPDF/blob/bb99e81e397c53fc9b6bba71d49408470c02f09e/tests/test_files/direct_and_image.pdf) on CUDA. It detected three text regions and recognized the paper title; character-level accuracy was not measured. The screenshot was rendered with PyMuPDF 2.1 from the top 200 PDF points of the first page at 2× resolution, then passed to `--image` above. The environment was copied from existing dependencies, not installed from scratch. A single screenshot does not establish overall document accuracy.

The initial clean-environment attempts on 2026-11-15 stopped at a broken inherited proxy and an incorrect OpenCV distribution pin; neither reached inference. See the [clean-install failure record](evidence/2026-11-15-clean-install-failure.json). After correcting the pin, a newly created venv passed dependency installation, `pip check`, CUDA checks and GPU inference. A Hub model-list request timed out, so this run reused device model files and matched their SHA-256 hashes against the fixed Hub revisions. The synthetic table produced all 12 expected text strings. See the [clean-environment GPU reproduction record](evidence/2026-11-15-clean-gpu-reproduction.json). This verifies fresh environment setup and GPU inference, but not online model downloads from an empty model cache.

To prepare that screenshot, use a separate Python environment with `pymupdf==2.1` and choose new output paths:

```bash
export PPOCR_TMP="$(mktemp -d)"
curl -fL https://raw.githubusercontent.com/RapidAI/RapidOCRPDF/bb99e81e397c53fc9b6bba71d49408470c02f09e/tests/test_files/direct_and_image.pdf -o "$PPOCR_TMP/source.pdf"
python - <<'PY'
import os, pymupdf
with pymupdf.open(os.path.join(os.environ['PPOCR_TMP'], 'source.pdf')) as doc:
    page = doc[0]
    page.get_pixmap(matrix=pymupdf.Matrix(2, 2), clip=pymupdf.Rect(0, 0, page.rect.width, min(200, page.rect.height))).save(os.path.join(os.environ['PPOCR_TMP'], 'page.png'))
PY
# In the OCR environment, run scripts/run.py above with --image "$PPOCR_TMP/page.png"
```
