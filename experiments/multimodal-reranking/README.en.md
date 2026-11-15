# Multimodal Reranking after Image-Text Retrieval

[中文](README.md)

This directory contains experiment code, configuration, run instructions, and result evidence. See the [image-text reranking guide](../../docs/zh/retrieval/reranking/多模态重排序.md) for the question, method, and results. Run the commands below from this directory.

[`scripts`](scripts/) · [`evidence`](evidence/) · [`figures`](figures/)

## Environment setup and reproduction

Recompute the public aggregate without images:

```bash
python scripts/summarize.py
```

Full reproduction requires matching annotations and images plus two models on Linux/CUDA. Verified environment: Python 3.12.3, Linux ARM64, NVIDIA GB10, PyTorch 2.14.0+cu130, torchvision 0.29.0+cu130, Transformers 5.16.1, sentence-transformers 6.0.1, qwen-vl-utils 0.0.14, and BF16. The CUDA 13.0 combination appears in the [official PyTorch version table](https://pytorch.org/get-started/previous-versions/); confirm support from the target driver, Python, CPU architecture, and GPU before installing.

```bash
python3.12 -m venv /path/to/venvs/retrieval
source /path/to/venvs/retrieval/bin/activate
python -m pip install pip==24.0
# CUDA 13.0 example; adjust for other devices using the official PyTorch guide
python -m pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
python -m pip check
python -c 'import torch, torchvision, qwen_vl_utils; print("torch:", torch.__version__, "torchvision:", torchvision.__version__)'
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA build:', torch.version.cuda, 'GPU:', torch.cuda.is_available()); assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(), 'A CUDA GPU with BF16 support is required'"
```

`qwen-vl-utils` 0.0.14 image processing directly depends on torchvision, although its package metadata does not declare that dependency; the import check above cannot be replaced by `pip check`. On ARM64, PyTorch 2.14.0 installs an `nvidia-cusparselt-cu13==0.8.1` wheel with an incorrect internal platform tag. pip 24.2 and later make `pip check` report that it is unsupported on this platform; using pip 24.0 allowed the dependency check, CUDA/BF16 checks, and inference to pass. See the [PyTorch issue](https://github.com/pytorch/pytorch/issues/202632). [`requirements.txt`](requirements.txt) pins the experiment's direct dependencies; install CUDA-specific torch and torchvision separately for the target device. `summarize.py` and `anonymize.py` use only the Python standard library. Keep outputs outside the repository: candidate and reranking files contain original descriptions and filenames and must not be published directly. Set the variables and prepare data and models as described below, then run:

```bash
python scripts/candidates.py --annotations "$ANNOTATIONS_CSV" --images "$IMAGES_DIR" --model "$BGE_MODEL" --output "$PRIVATE_DIR/candidates.json"
python scripts/rerank.py --candidates "$PRIVATE_DIR/candidates.json" --images "$IMAGES_DIR" --model "$QWEN_MODEL" --output "$PRIVATE_DIR/reranked.jsonl"
python scripts/anonymize.py --candidates "$PRIVATE_DIR/candidates.json" --reranked "$PRIVATE_DIR/reranked.jsonl" --output "$PRIVATE_DIR/anonymous.json"
python scripts/summarize.py "$PRIVATE_DIR/anonymous.json"
```

Generate five solid-color images and annotations for a self-check. First choose a new `PRIVATE_DIR` outside the repository, then run:

```bash
export PRIVATE_DIR=/path/to/new-work-directory
mkdir -p "$PRIVATE_DIR/synthetic-images"
python - <<'PY'
import csv, os
from pathlib import Path
from PIL import Image
root = Path(os.environ['PRIVATE_DIR'])
with (root / 'synthetic-annotations.csv').open('w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['split', 'raw', 'filename'])
    writer.writeheader()
    for i, color in enumerate(['red', 'green', 'blue', 'yellow', 'purple']):
        name = f'{i}.png'
        Image.new('RGB', (96, 96), color).save(root / 'synthetic-images' / name)
        writer.writerow({'split': 'test', 'raw': repr([f'a solid {color} square']), 'filename': name})
PY
ANNOTATIONS_CSV="$PRIVATE_DIR/synthetic-annotations.csv"
IMAGES_DIR="$PRIVATE_DIR/synthetic-images"
```

To check the pipeline with a smaller sample, obtain the models below and set `BGE_MODEL` and `QWEN_MODEL`, then run the following with a new set of output files. The annotations must contain at least five test rows. This smoke check does not reproduce the historical metrics:

```bash
python scripts/candidates.py --annotations "$ANNOTATIONS_CSV" --images "$IMAGES_DIR" --model "$BGE_MODEL" --sample-size 5 --output "$PRIVATE_DIR/smoke-candidates.json"
python scripts/rerank.py --candidates "$PRIVATE_DIR/smoke-candidates.json" --images "$IMAGES_DIR" --model "$QWEN_MODEL" --output "$PRIVATE_DIR/smoke-reranked.jsonl"
python scripts/anonymize.py --candidates "$PRIVATE_DIR/smoke-candidates.json" --reranked "$PRIVATE_DIR/smoke-reranked.jsonl" --expected-count 5 --output "$PRIVATE_DIR/smoke-anonymous.json"
python scripts/summarize.py "$PRIVATE_DIR/smoke-anonymous.json" --expected-count 5
```

Before running, set `ANNOTATIONS_CSV`, `IMAGES_DIR`, `BGE_MODEL`, `QWEN_MODEL`, and `PRIVATE_DIR` to the annotation CSV, image directory, two local model directories, and a new output directory outside the repository; create that output directory. Annotations and images are not distributed here. The historical run used the Flickr30k Karpathy test split. Obtain images and annotations through the [Flickr30k source](https://huggingface.co/datasets/WhereIsAI/flickr30k-v2) and verify their terms. The conversion script below exports images and CSV from the dataset; by default, it selects 512 training, 128 validation, and 100 test rows in source order, recording selected indices, seed, and data version. Verify image and annotation terms first. Data ordering may differ from the historical private annotations, so this path does not guarantee the original sample. `candidates.py` expects a UTF-8 CSV with `split`, `raw`, and `filename` columns; at least 100 rows with `split=test`; `raw` must be a nonempty Python list of strings, with its first caption used as the query; and `filename` is relative to the image directory. Example:

```csv
split,raw,filename
test,"['a red car', 'a vehicle']",example.jpg
```

Prepare data in a new, nonexistent directory by replacing `/path/to/new-flickr-export`. If known, pin the upstream commit with `--revision <commit>`; if omitted, record that it is unpinned. The first run downloads data and writes image files, so reserve disk space:

```bash
python scripts/prepare_flickr30k.py --out /path/to/new-flickr-export
```

Without dataset or model weights, run `python -m unittest discover -s tests -v` to check sampling, CSV and image export, overwrite protection, and the reranking, anonymization, summary, and resume pipeline with a fake model. Tests require only Pillow; the fake-model test does not validate real models or GPU inference.

Set `ANNOTATIONS_CSV` and `IMAGES_DIR` to the exported `annotations.csv` and `images/` paths. Obtain the models as follows. The historical Qwen revision is unknown; the pinned revision below has a weight file matching the historical SHA-256. Verify the hash after download. Reserve space for models and images. Custom model code is enabled for remote loading; inspect its source and license before use.

```bash
ANNOTATIONS_CSV=/path/to/annotations.csv
IMAGES_DIR=/path/to/images
PRIVATE_DIR=/path/to/new-work-directory
BGE_MODEL="$PRIVATE_DIR/bge"
QWEN_MODEL="$PRIVATE_DIR/qwen"
mkdir -p "$PRIVATE_DIR"
hf download BAAI/BGE-VL-base --revision cc4c733ed997dbee4ac70ccffb911e70c9c24b93 --local-dir "$BGE_MODEL"
hf download Qwen/Qwen3-VL-Reranker-2B --revision 4bd860ac4f15ad1897a214615cccc700f8f71818 --local-dir "$QWEN_MODEL"
sha256sum "$QWEN_MODEL/model.safetensors"
```

After preparing the models and data, run a five-image synthetic GPU smoke check in a new environment:

```bash
export XL_DATA_DIR=/path/to/external-data
export XL_OUT_DIR=/path/to/new-external-output
export XL_RUN_ID=rerank-clean-001
export BGE_MODEL="$PRIVATE_DIR/bge"
export QWEN_MODEL="$PRIVATE_DIR/qwen"
bash scripts/run_clean_gpu_reproduction.sh
```

The script creates a new environment at `XL_DATA_DIR/XL_RUN_ID/venv`, installs the pinned dependencies, checks packages and CUDA/BF16, then runs candidate generation, reranking, anonymization, and summary. It does not download or overwrite model weights. Use a unique `XL_RUN_ID` and an output directory outside the repository. The script refuses to reuse an existing environment or overwrite an existing summary. On success it writes `environment.json`, `metrics.json`, and `summary.json` without model paths. This checks installation and inference, not real-dataset retrieval quality.

The historical SHA-256 for Qwen's `model.safetensors` is `466ec01961061e9d7f804b4fb1625fb6f406106cd1567e026096d4736fa9d5b9`. The Qwen inference wrapper in the script is from the [official Qwen implementation](https://github.com/QwenLM/Qwen3-VL-Embedding), with a compatibility check added for converting multimodal token types to tensors in the Transformers 5.16.1 environment.

Reranking can resume from an existing JSONL file after checking each completed query, candidate list, and model path. Older records lack the information required for safe resume; choose a new output file for them. The path does not encode a weight hash, so ensure the model files at that path have not changed before resuming. Image or video preprocessing errors stop the run; fix the input before retrying.

The public reranking script sets `min_pixels=4096` and `max_pixels=65536` for image input and loads the model in BF16. These preprocessing settings affect reproduction and should be recorded if changed.

On 2026-11-13, the candidate, reranking, anonymization, and summary pipeline ran on Linux ARM64 with an NVIDIA GB10 CUDA GPU and BF16, using the two pinned model revisions above and five generated solid-color images. The run used layered dependencies: Python 3.12, PyTorch 2.15.0+cu132, torchvision 0.29.1+cu130, Transformers 5.16.1, sentence-transformers 6.0.1, qwen-vl-utils 0.0.14, huggingface-hub 1.33.0, and Pillow 13.0.0. The layered environment was not a fresh independent install, and its overall `pip check` was not verified. This synthetic sample checks execution, not retrieval quality or historical results.

On 2026-11-14, the experiment dependencies were installed in a copied and adjusted isolated environment with the same versions, and `pip check` passed. The public scripts completed candidate generation, reranking, anonymization, and summary using the pinned models and five synthetic images. This was not a fresh installation, and the synthetic sample cannot establish real retrieval quality.

On 2026-11-15, [`scripts/run_clean_gpu_reproduction.sh`](scripts/run_clean_gpu_reproduction.sh) installed the pinned dependencies into a fresh Python 3.12 venv on GB10. `pip check`, TorchVision/Qwen utility imports, CUDA, and BF16 checks passed; the public scripts completed candidate generation, reranking, anonymization, and summary using the two fixed-revision models already cached on the device and five generated images. Versions and metrics are in [`evidence/2026-11-15-clean-env-gpu-reproduction.json`](evidence/2026-11-15-clean-env-gpu-reproduction.json). Hit@1 remained 5/5, a ceiling result on a synthetic sample; this does not establish retrieval quality. The run reused cached model weights and did not verify downloading models or data from an empty device cache.

On 2026-11-14, the same separate environment ran the public candidate generation, reranking, anonymization, and summary scripts on 20 image-caption pairs sampled from the Flickr30k test split at revision `a2ac1f0bcee3132331324f1c435bcea4527c08f1`. All 20 corresponding images ranked first both before and after reranking. The candidate pool contained only 20 images and the result was at the ceiling, so it cannot establish an improvement from reranking or represent the full dataset or historical results.

The full public pipeline was then rerun with its default 100 test image-caption pairs from the same data revision and environment. With a 100-image candidate pool, Hit@1 was 97/100 before and 99/100 after reranking: three queries improved and one worsened. The paired per-query bootstrap 95% interval for the difference was [-0.02, 0.06]. It crosses zero; the small candidate pool and annotations identifying the paired image do not establish a stable improvement or represent all of Flickr30k. The data order also does not guarantee reconstruction of the historical private sample.

The historical measurement used equivalent private scripts, and the public summary script recomputed metrics from anonymized results. Dependency versions, image preprocessing, sampling, and model file differences can all change results. The annotation source and model-weight verification requirements are described in the guide's [data and experiment scope](../../docs/zh/retrieval/reranking/多模态重排序.md#二-实验设计与实测结果).
