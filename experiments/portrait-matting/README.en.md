# Evaluating Portrait Matting with RVM: Subject Regions, Hair Boundaries, and Inference Time

[中文](README.md)

This experiment runs RVM MobileNetV3 on 30 portrait photos to examine subject coverage, edges, alpha mattes, and model forward-pass time. Each photo is processed independently. The experiment does not evaluate temporal consistency in video or compare other models under the same conditions. See the [portrait matting guide](../../docs/zh/multimodal/人物抠像质量与推理速度.md) for motivation, metric interpretation, and results.

[scripts](scripts/) · [evidence](evidence/) · [assets](assets/)

For reproduction, first prepare the pinned model version and data, then run the evaluation. Per-image metrics in the output help locate failure cases hidden by averages. Original photos and outputs are for local use only; check the terms for each asset separately before publishing example images.

Run all commands below from this experiment directory.

## Installation and preparation

Prepare Python, a device-compatible version of PyTorch and torchvision, and install `numpy` and `Pillow`. Historical records used PyTorch 2.14.0 and CUDA 13.0; complete versions for the other dependencies were not pinned. The upstream RVM code is licensed under [GPL-3.0](https://github.com/PeterL1n/RobustVideoMatting/blob/master/LICENSE); check the applicable license before using its code or model. Select the platform and CUDA configuration using the [official PyTorch installation guide](https://pytorch.org/get-started/locally/). CPU execution is possible but slower. Set `WORK_DIR` to an existing working directory outside the repository. Retrieve the pinned RVM source and download `rvm_mobilenetv3.pth` from the [official repository's PyTorch download entry](https://github.com/PeterL1n/RobustVideoMatting#download) into `$WORK_DIR/weights/`. The weight SHA-256 expected by the code is `3c7c1d92033f7c38d6577c481d13a195d7d80a159b960f4f3119ac7b534cf4f8`.

```bash
: "${WORK_DIR:?Set a working directory outside the repository first}"
mkdir -p "$WORK_DIR/weights"
python3 -m venv "$WORK_DIR/.venv"
source "$WORK_DIR/.venv/bin/activate"
# Install torch and torchvision according to the official PyTorch guide
git clone https://github.com/PeterL1n/RobustVideoMatting.git "$WORK_DIR/rvm-src"
git -C "$WORK_DIR/rvm-src" checkout 53d74c6826735f01f4406b5ca9075eee27bec094
python -m pip install -r requirements.txt
curl -fL https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3.pth -o "$WORK_DIR/weights/rvm_mobilenetv3.pth"
echo '3c7c1d92033f7c38d6577c481d13a195d7d80a159b960f4f3119ac7b534cf4f8  '"$WORK_DIR"'/weights/rvm_mobilenetv3.pth' | sha256sum -c -
curl -fL https://paddleseg.bj.bcebos.com/matting/datasets/PPM-100.zip -o "$WORK_DIR/PPM-100.zip"
python scripts/prepare_ppm.py --archive "$WORK_DIR/PPM-100.zip" --output "$WORK_DIR/local-ppm-30"
```

The sampling script first checks archive SHA-256 `5a3d9029437954594c26ea773e927c4b7c4caaee7e94f6bbd8e96196c868254a`, then writes the images, annotations, and a manifest with per-file hashes. `$WORK_DIR/local-ppm-30` is local data and should not be committed to the public repository. If a filename or hash changes, recheck the source instead of disabling the validation.

After setup, `python -m pip check` and `python -c "import torch, torchvision, numpy, PIL; print('CUDA:', torch.cuda.is_available())"` check dependency imports and CUDA visibility only; they do not demonstrate that the model loads or that the evaluation has run.

On 2026-11-13, the command below processed the fixed sample of 30 PPM portrait photos and their alpha labels on CUDA using Linux ARM64, NVIDIA GB10, Python 3.12.3, PyTorch 2.15.0+cu132, torchvision 0.29.1+cu130, NumPy 2.5.4, and Pillow 13.0.0. The run produced 30 nonempty alpha images and result records. The inputs were dataset photos, not script-generated images; this run did not extract video frames or assess temporal consistency. It used layered dependencies, and `pip check` reported `nvidia-cusparselt-cu13` as unsupported on the platform in the base environment; a complete fresh installation has not been validated. This run checks the script and pinned weights, not every device, and does not replace verification of dataset and portrait-use terms.

On 2026-11-14, the same 30 photos were evaluated successfully on similar hardware with the dependency versions above in a copied and adjusted isolated environment; `pip check` passed. This was not a fresh installation and covered still images only, not video temporal consistency.

## Run single-frame evaluation

```bash
python scripts/run_rvm_ppm.py \
  --manifest "$WORK_DIR/local-ppm-30/manifest.jsonl" --output "$WORK_DIR/local-rvm-results" \
  --rvm-repo "$WORK_DIR/rvm-src" --weights "$WORK_DIR/weights/rvm_mobilenetv3.pth" \
  --device cuda --max-side 1024 --warmup 1 --repeats 3
```

Without CUDA, use `--device cpu`. The output directory must not already exist, to prevent overwriting. RVM's actual dependencies are determined by the dependency file at the pinned revision. Before the first run, confirm that the local PyTorch can load them.

Inputs are decoded as RGB, resized proportionally so the longest side is at most 1024, then converted to FP32 `1×3×H×W` in the 0–1 range. Labels are resized to the same dimensions with bilinear interpolation. The model's internal downsample ratio is set to `min(1,512/max(H,W))`. Each forward pass receives four empty states and returns an alpha matte at the same dimensions. Predictions are rounded and saved as 8-bit PNGs; metrics are calculated from those saved predictions. Source images and labels are not independently rotated according to EXIF, to avoid coordinate misalignment.

Each image gets one warmup and three timed runs; CUDA is synchronized before and after timing. Timing covers the model forward pass and synchronization, but excludes decoding, input upload, output download, and writing to disk. Image sizes can differ, so these timings are neither fixed-resolution throughput nor end-to-end latency. `results.json` contains model and data hashes, dimensions, parameters, per-run timings, per-image metrics, and macro averages. `alpha/` and `target/` contain predictions and resized annotations, including restricted asset information; keep them local.

## Checks

```bash
python scripts/test_score_mattes.py
python scripts/test_run_rvm_ppm.py
```

These tests use synthetic data and a mock predictor to check metrics, preprocessing integration, result writing, and validation failures. They do not run RVM and cannot replace a real-model or device test.
