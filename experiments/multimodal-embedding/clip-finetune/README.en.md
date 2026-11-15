# CLIP Image-Text Fine-Tuning: Data, Loss, and Retrieval Evaluation

[中文](README.md)

When retrieval targets shift from everyday photos to product images or specialist charts, a general-purpose image-text model's similarity scores may not fit the target domain. Further training on domain images and descriptions can adjust the joint embedding space; held-out queries and images can then compare retrieval before and after fine-tuning.

This directory demonstrates bidirectional CLIP contrastive training with partial layer freezing. A fixed-version, real-data GPU small-sample run compared the fine-tuned model with its baseline; full-dataset training has not been verified.

## Data: multiple captions per image

The scripts read [WhereIsAI/flickr30k-v2](https://huggingface.co/datasets/WhereIsAI/flickr30k-v2), whose rows contain an image, a list of captions, an image ID, and a split field. The images come from Flickr30k, with multiple English captions per image. The dataset page does not provide complete data-use licensing details; verify the image and annotation sources and applicable terms before use.

One source record and its expanded training pairs look like this:

```text
Source:   image = image, img_id = 123, caption = [caption A, caption B, …]
Expanded: (image, caption A, 123), (image, caption B, 123), …
```

The data loader creates one image-text pair for each caption. During training, `CLIPProcessor` converts images into pixel tensors and text into token IDs and attention masks; the model outputs image-text similarity matrices. For retrieval, images and text are encoded separately, then ranked by vector similarity.

The dataset's outer structure contains only `train`; each row's `split` field indicates `train`, `val`, or `test`. Data preparation groups by that field before expanding captions. It also checks that image IDs exist, split names are valid, all three sets are nonempty, and no image ID crosses sets. If another source also uses the Flickr30k name, first confirm its version and split are the same.

## How the loss trains image-text matching

For a batch of `N` image-text pairs, let the model output normalized image vectors $\mathbf{v}_i$ and text vectors $\mathbf{t}_j$. A learnable scale $\alpha$ produces similarities:

$$
s_{ij}=\alpha\,\mathbf{v}_i^{\mathsf T}\mathbf{t}_j.
$$

The training script averages cross-entropy losses in both directions, using the batch diagonal as labels:

$$
\mathcal{L}=\frac{1}{2}\left[
-\frac{1}{N}\sum_{i=1}^{N}\log\frac{e^{s_{ii}}}{\sum_{j=1}^{N}e^{s_{ij}}}
-\frac{1}{N}\sum_{i=1}^{N}\log\frac{e^{s_{ii}}}{\sum_{j=1}^{N}e^{s_{ji}}}
\right].
$$

The first term brings each image closer to its caption; the second brings each caption closer to its image. By default, the script freezes other layers and unfreezes only the final 2 vision-encoder layers, final 4 text-encoder layers, both projection layers, and the scale parameter.

If two captions for the same image enter one batch, the diagonal loss incorrectly treats the other valid caption as a negative. The sampler therefore groups by image ID each epoch, randomly selects one caption per image, then builds batches. An image appears only once per epoch; a final batch with only one record is dropped. With a fixed seed, results are reproducible, while different epochs may select different captions. Different images may also be semantically related, leaving false negatives that still require data and annotation handling.

## Code and evaluation scope

- `finetune_clip_with_freeze.py`: builds batches, freezes parameters, trains, and selects checkpoints.
- `data_common.py`: loads data, expands image-text pairs, selects distinct images for training batches, and computes Recall@1/5/10 for image-to-text and text-to-image retrieval. For image-to-text, all captions for the same image count as relevant; for text-to-image, each caption maps to its annotated image.
- `get_dataset.py` and the exploration notebook: data loading and inspection. Data, weights, and training outputs are not included.

The configuration loads the model by default from `models/CLIP-ViT-B-32-laion2B-s34B-b79K` and uses large batches; edit `CONFIG` in the script to fit the available device. Training output defaults to the Git-ignored `.work/clip_output`; set `XL_OUT_DIR` to an external output directory to keep models and training results out of version control. The training loop currently accepts only `grad_accum_steps=1`. `requirements.txt` is the training dependency entry point; install PyTorch separately for the platform. The incomplete historical environment snapshot is in `requirements-historical.txt`. Model source: [LAION model card](https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K).

## Real-data small-sample rerun

On 2026-11-14, a Linux ARM64 NVIDIA GB10 run used `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` revision `1a25a446712ba5ee05982a381eed697ef9b435cf`, with weights SHA-256 `74813fbcdc750f235c9784c367ca1394d2a5c25eb0aac92761752ac239db7cff`. From Flickr30k revision `a2ac1f0bcee3132331324f1c435bcea4527c08f1`, the first 256 training, 64 validation, and 64 test images in their original splits produced 1280, 320, and 320 image-caption pairs. The separate copied environment used Python 3.12.3, PyTorch 2.15.0+cu132, Transformers 5.16.1, Datasets 5.2.0, and Matplotlib 3.11.2. Dependencies were supplemented from the requirements and passed `pip check`; this was not a clean installation.

BF16 training used batch size 16 for one epoch and 16 steps. On the 64 test images, image-to-text Recall@1 fell from 0.9844 to 0.9688, and text-to-image Recall@1 fell from 0.9375 to 0.8781. This uses a small candidate set and one seed, and model pretraining overlap has not been excluded. It does not establish an improvement or generalize to the full dataset.

The entry point below loads the pinned dataset revision and selects the first records within each split. It downloads data and requires an empty output directory. Set `CLIP_MODEL_DIR` and `RUNS_DIR` as described in the environment section:

```bash
CLIP_RUN="$(mktemp -d "$RUNS_DIR/clip-real-XXXXXXXX")"
python run_real_subset.py --model "$CLIP_MODEL_DIR" --out "$CLIP_RUN"
```

The actual run used a previously prepared local subset of the same revision and passed `--local-dataset <local-subset-path>`; this option does not independently verify the local data source. Direct Hub loading through this entry point has not been validated end to end. The output includes the model and configuration; summary metrics are printed to the log, so retain that log.

## Environment setup

Commands below use Linux and Bash. Operating system, CPU architecture, and dependency combinations have not been fully compatibility-tested. The script uses a GPU when CUDA is detected and otherwise falls back to CPU. Before fine-tuning, confirm the NVIDIA GPU supports the installed PyTorch CUDA version. The default configuration sets `bf16=True`, so the GPU must also support BF16; record the configuration if changing precision. Default training batch size is 300 and evaluation batch size is 640. Minimum VRAM, RAM, and disk requirements have not been measured; adjust for device capacity and reserve space for data cache, model, and outputs.

Create an isolated Python environment, install a CUDA build compatible with the device and driver following the [PyTorch installation guide](https://pytorch.org/get-started/locally/), then install the other direct dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
# Install a CUDA build suited to the device following the PyTorch guide
python -m pip install -r requirements.txt
```

This is an installation entry point, not a verified fresh-environment recipe. Versions are not fully pinned; `requirements.txt` omits platform-specific PyTorch, and `requirements-historical.txt` is an incomplete historical snapshot, not an installation file.

By default, the model loads from `./models/CLIP-ViT-B-32-laion2B-s34B-b79K`, relative to the current working directory when the script starts. Download the [model repository](https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K) into a new directory and set `CONFIG["model_name"]` to it. This download uses disk space; point `MODEL_ROOT` to a location with enough room:

```bash
export MODEL_ROOT=/path/to/model-assets
mkdir -p "$MODEL_ROOT"
CLIP_MODEL_DIR="$(mktemp -d "$MODEL_ROOT/clip-model-XXXXXXXX")" || exit 1
export CLIP_MODEL_DIR
hf download laion/CLIP-ViT-B-32-laion2B-s34B-b79K --revision 1a25a446712ba5ee05982a381eed697ef9b435cf --local-dir "$CLIP_MODEL_DIR"
```

The script uses Transformers' `CLIPModel` and `CLIPProcessor`; the local directory must contain the corresponding weights, configuration, and processor files. OpenCLIP-format weights alone are insufficient. Alternatively, set `model_name` to the Hub model ID so loading fetches it into cache. Data is loaded with `datasets.load_dataset` from the [Flickr30k dataset page](https://huggingface.co/datasets/WhereIsAI/flickr30k-v2) and defaults to the user's Hugging Face data cache. Check cache location, network access, and usage terms. The first load downloads the data. Before running, set `FLICKR_REVISION` to a verified full dataset-repository commit SHA. The loader requests that revision, and the training output's `train_config.json` records the requested value. If unset, it loads the default revision, records `null`, and does not automatically record the resolved commit, so the data cannot be pinned or reconstructed from that file. The inspection entry point `get_dataset.py` also accepts this variable but does not write a training configuration.

After setup, check CUDA, BF16, and imports in the same environment:

```bash
python -c 'import torch, transformers, datasets, matplotlib, PIL; print("torch", torch.__version__, "CUDA", torch.version.cuda, "available", torch.cuda.is_available()); assert torch.cuda.is_available(); print("GPU", torch.cuda.get_device_name(0), "BF16", torch.cuda.is_bf16_supported()); assert torch.cuda.is_bf16_supported()'
```

Passing this check does not mean a full training run has been verified. Once model and data access are ready, dependencies installed, and `CONFIG` edited, use a new output directory for each run. Before loading data or models, the script checks that the directory is empty; an existing empty directory is allowed. It creates a `.training-started` marker to prevent two runs from sharing a directory. After a failure, use a new directory for the next run. Set `RUNS_DIR` to a location with enough disk space:

```bash
export RUNS_DIR=/path/to/experiment-runs
mkdir -p "$RUNS_DIR"
XL_OUT_DIR="$(mktemp -d "$RUNS_DIR/clip-XXXXXXXX")" || exit 1
export XL_OUT_DIR
export FLICKR_REVISION="<verified full dataset commit SHA>"
python finetune_clip_with_freeze.py
```

The historical dataset revision and dependency versions are not pinned, and full training on the dataset has not been verified; reproducing historical effects is not guaranteed. In small evaluations, if there are fewer than K candidates, Recall@K uses all available candidates and should not be directly compared with the same metric over a full candidate set.

After installing PyTorch, Transformers, Datasets, and Matplotlib, run this CPU check from the experiment directory. It uses synthetic fields and a randomly initialized small model to cover split handling, cross-set ID conflicts, unique images within batches, and one forward/backward pass. It does not download Flickr30k or evaluate retrieval quality.

```bash
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -v
```

You can also check the actual training entry point on CUDA with a randomly initialized tiny CLIP and synthetic image-text data. This does not download pretrained weights or Flickr30k. It performs two training steps, validation, testing, and saving, and checks that the projection weights changed. Run from this directory and set `RUNS_DIR` to a writable location:

```bash
export RUNS_DIR=/path/to/experiment-runs
mkdir -p "$RUNS_DIR"
XL_OUT_DIR="$(mktemp -d "$RUNS_DIR/clip-smoke-XXXXXXXX")" || exit 1
export XL_OUT_DIR
python tests/smoke_cuda_train.py
```

Outputs, including configuration, checkpoint, plots, and final model, go to `$XL_OUT_DIR/synthetic-clip-training/`. The synthetic processor writes only a placeholder, so the final model directory is not a complete package for real inference. On 2026-11-13, this two-step CUDA check and 10 CPU tests passed on Linux ARM64 with an NVIDIA GB10 using Python 3.12.3, PyTorch 2.15.0+cu132, Transformers 5.22.2, Datasets 5.2.0, Matplotlib 3.11.2, Pillow 13.0.0, and torchvision 0.29.1+cu130. Dependencies came from an existing environment and additional dependency directories; a clean installation has not been verified. This check substitutes the data loader, processor, and model loading, and uses a random model, synthetic data, and FP32. It does not establish that real data processing, pretrained weight loading, BF16, or historical results are reproducible; its metrics do not measure model quality.

The script evaluates the fine-tuned and original models on the same test candidate set and computes Recall in both directions. This metric depends on the candidate set, split, and annotation coverage. Only an actual run with saved configuration, results, and per-item evidence can establish whether fine-tuning changes retrieval. This small-sample run retained its configuration and summary log, but did not save per-query evidence.
