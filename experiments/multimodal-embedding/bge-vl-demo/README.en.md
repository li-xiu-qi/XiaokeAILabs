# BGE-VL: Image and Composed Image-Text Queries

[中文](README.md)

Given a reference image and text describing a desired change, the model can encode the combined input as a query vector and compare it with candidate image vectors. This can be used to explore composed image retrieval, such as finding a photo with a darker background based on a reference image.

[`test_bge_vl.ipynb`](test_bge_vl.ipynb) follows the [official BGE-VL model example](https://huggingface.co/BAAI/BGE-VL-large) through the Transformers interface. It calls `set_processor`, encodes the query and candidate images separately, and computes scores with vector dot products. The notebook includes image-to-image and composed image-plus-text queries; it does not implement text-only image retrieval.

## Environment setup

The example uses `BAAI/BGE-VL-large`. Install compatible PyTorch, Transformers, Pillow, and FlagEmbedding versions. Model custom code is loaded with `trust_remote_code=True`. `requirements.txt` pins Transformers 4.46.3 and FlagEmbedding 1.4.2; other dependencies and the large model revision are not pinned. Consult the upstream model card and inspect the code source before running.

The commands below use Linux and Bash. The operating system, CPU architecture, and dependency combination have not been fully validated, and minimum memory and disk requirements have not been measured. The notebook does not explicitly move the model to a GPU, so its existing output must not be treated as GPU inference validation. The first run needs access to the model source and enough space for weights and cache.

Create an isolated Python environment, install the device-appropriate PyTorch version following the [PyTorch installation guide](https://pytorch.org/get-started/locally/), then install notebook and example dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
# Install a device-appropriate torch version following the PyTorch guide
python -m pip install -r requirements.txt
```

This is an installation entry point, not a full dependency lock; a fresh environment has not been verified end to end. FlagEmbedding 1.4.2 declares a Transformers requirement from 4.44.2 up to, but excluding, 6.0. In an environment with existing Python and PyTorch, `pip install --dry-run -r requirements.txt` resolved the dependencies, but the full combination was not installed or run, and the notebook mapping cell has not been tested. Check imports, then create an isolated run directory and copy the notebook there. Replace `RUNS_DIR` with a work directory and select the current environment as the Jupyter kernel:

```bash
python -c 'import torch, transformers, PIL, FlagEmbedding; print("torch", torch.__version__, "transformers", transformers.__version__)'
python -m pip check
export RUNS_DIR=/path/to/experiment-runs
mkdir -p "$RUNS_DIR"
BGE_RUN_DIR="$(mktemp -d "$RUNS_DIR/bge-vl-XXXXXXXX")" || exit 1
export BGE_RUN_DIR
cp test_bge_vl.ipynb "$BGE_RUN_DIR/"
mkdir -p "$BGE_RUN_DIR/.work"
# Put cat.png and dog.png in $BGE_RUN_DIR/.work before launching
cd "$BGE_RUN_DIR"
python -m jupyter lab test_bge_vl.ipynb
```

On first load, the notebook's `AutoModel.from_pretrained` and `set_processor` fetch model and processor files from [the model repository](https://huggingface.co/BAAI/BGE-VL-large), using the Hugging Face cache configuration. The model repository or dependency updates may change the interface. For offline use, prepare local files according to the model source instructions and change the notebook's `MODEL_NAME` to the local directory containing model and processor files; this offline path has not been verified for the large model.

Before running, put two images that you are allowed to use in the new run directory's `.work/`, named `cat.png` and `dog.png`, or edit the image paths in the notebook. This example demonstrates the interface and output shapes only; it is not a model quality evaluation.

## CUDA interface check

This separate smoke script uses the smaller `BAAI/BGE-VL-base` with the same `AutoModel`, `set_processor`, and `encode` interface to score two candidate images for an image query and an image-plus-text query. It generates two solid-color images and requires CUDA. It does not run the notebook cell that inspects FlagEmbedding model mappings. In an isolated environment, first install matching CUDA builds of PyTorch and torchvision following the PyTorch installation guide, then run:

```bash
python -m pip install 'transformers==4.46.3' pillow
export RUNS_DIR=/path/to/experiment-runs
mkdir -p "$RUNS_DIR"
BGE_RUN_DIR="$(mktemp -d "$RUNS_DIR/bge-vl-smoke-XXXXXXXX")" || exit 1
export BGE_RUN_DIR
export MODEL_DIR="$BGE_RUN_DIR/model"
# The first download needs disk space; do not point local_dir at existing files.
python -c 'import os; from huggingface_hub import snapshot_download; snapshot_download(repo_id="BAAI/BGE-VL-base", revision="cc4c733ed997dbee4ac70ccffb911e70c9c24b93", local_dir=os.environ["MODEL_DIR"])'
python smoke_bge_vl.py --model "$MODEL_DIR" --output-dir "$BGE_RUN_DIR/results"
```

`results/` contains the generated images and `result.json`. On 2026-11-13, offline loading and both CUDA query types passed on Linux ARM64 with NVIDIA GB10, Python 3.12.3, PyTorch 2.15.0+cu132, Transformers 4.46.3, Pillow 13.0.0, torchvision 0.29.1+cu130, and the model revision above; candidate vectors had shape `[2, 512]`. Dependencies came from an existing environment and isolated dependency directories; the installation commands have not been verified in a clean environment. The solid-color scores only check the interface, not retrieval quality; neither the large model nor the complete notebook was validated by this check.

For this model revision, the `AutoModel` custom code produced invalid position indices in our Transformers 5.16.1 and 5.22.2 environments, causing CUDA inference to fail. Prefer the tested 4.46.3 above and recheck when upgrading. Other model-loading interfaces may use different implementations, so their success does not establish that this path works.
