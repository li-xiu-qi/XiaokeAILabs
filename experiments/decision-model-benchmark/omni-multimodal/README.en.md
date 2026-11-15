# Jev-Omni: Multimodal Decision Model Deployment and Evaluation

## About this experiment

Jev-Omni is an open-source, typed multimodal decision model fine-tuned from Gemma 4 12B IT. It supports text, images, audio, and video through one model. Its use follows the System 1 decision-model pattern: given the current state, a question, and candidate options, the model generates no text and directly returns a probability distribution over the options for classification, routing, or action selection.

This experiment checks the single-GPU private deployment process, actual inference latency across all four modalities, and whether decisions and confidence scores are usable.

## Evaluated model

| Item | Value |
|---|---|
| Model | akhilaaa3/Jev-Omni (Apache-2.0) |
| Base model | google/gemma-4-12B-it (about 24 GB, BF16) |
| Decision backbone | 12B; merged weights distributed in FP32 (about 48 GB, 13 shards) |
| Official benchmarks | JevBench 86.15%, DecisionBench Medium 87.57% |
| Runtime memory | About 27 GB resident in our measurement |
| Load time | About 8 minutes in our measurement (reading 48 GB of FP32 shards) |

Prepare over 60 GB of available memory or GPU memory. Unified-memory devices can run it. CUDA, `ffmpeg`, and Python 3.10 or later are required.

## Directory contents

| Path | Contents |
|---|---|
| `scripts/download_models.py` | Downloads all Jev-Omni weights and the Gemma 4 base; supports mirrors and a custom directory |
| `scripts/download_assets.py` | Downloads three separate CC0 test assets: a cat image, conversational audio, and a person in a video |
| `scripts/quickstart.py` | Minimal check that runs all four modalities once |
| `scripts/bench_latency.py` | Latency benchmark for four modalities (p50/p90/p99) |
| `scripts/multimodal_eval.py` | Batch evaluation: 10 image classes, 3 audio classes, and 1 video; reports accuracy and confidence |
| `scripts/test_suite.py` | Combined suite covering six groups: question types, option counts, Chinese business cases, text stability, multimodal understanding, and text length; loads the model once |
| `requirements.txt` | Runtime dependencies |
| `部署要点与工程边界.md` | Environment dependencies, runtime patches, and measured memory and loading details (Chinese source: `部署要点与工程边界.md`) |
| `../evidence/17-jevomni-多模态批量评估.excerpt.txt` | Excerpt of the batch evaluation with 10 images, 3 audio samples, and 1 video; the multimodal conclusions mainly come from this log |

Weights and assets are not stored in version control. Running the scripts creates `models/` (weights), `assets/` (three basic assets), and `results/` (benchmark output).

## Quick reproduction

On Linux, first install CUDA builds of PyTorch and torchvision compatible with your device, then install the requirements. Weights are provided by the model platform. If the base-model repository requires login or license acceptance, complete authorization on its model page and configure Hugging Face login. `ffmpeg` must be on `PATH`; check it with `ffmpeg -version`. The dependency list and download steps do not fully lock the historical environment.

```bash
# 1. An isolated virtual environment is recommended
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -c "import torch, torchvision; print('torch:', torch.__version__, 'torchvision:', torchvision.__version__, 'CUDA:', torch.cuda.is_available()); assert torch.cuda.is_available()"
ffmpeg -version

# 2. Download weights (add --mirror for networks in China)
python scripts/download_models.py --mirror

# 3. Download the three test assets (a GitHub acceleration prefix can be added if direct access fails)
python scripts/download_assets.py
#   python scripts/download_assets.py --gh-prefix https://ghfast.top/

# 4. Minimal check; ffmpeg must be on PATH
python scripts/quickstart.py

# 5. Latency benchmark
python scripts/bench_latency.py

# 6. Batch multimodal evaluation (requires the extended asset set)
python scripts/download_assets.py --full
python scripts/multimodal_eval.py

# 7. Combined test suite (question types, scale, Chinese, stability, multimodal understanding)
python scripts/test_suite.py
```

## Test assets

Media assets are excluded from Git and hosted in GitHub Release Assets. `download_assets.py` fetches them and verifies each file with SHA256. The asset release uses the tag `assets/jev-open-source`.

The basic set consists of three independent, commercially usable free assets, one for each non-text modality:

| File | Content | Source |
|---|---|---|
| `assets/cat.jpg` | Close-up of a tabby cat | Flickr (CC0) |
| `assets/conversation.mp3` | About 12 seconds of two people speaking English about a volleyball team | Freesound (CC0) |
| `assets/person-reading.mp4` | About 19 seconds showing a tired young person putting on glasses and looking down to read | Pexels (Pexels License) |

Use `--full` to download the extended evaluation set to `assets/eval/`: `images/` contains one image for each of 10 classes (cat, dog, car, person, bicycle, bird, boat, chair, building, and flower), while `audio/` contains one sample each of speech, music, and environmental sound. `multimodal_eval.py` classifies these assets using fixed candidate labels and reports accuracy and confidence.

The script downloads from the Release by default and verifies each file's SHA256; files with mismatched hashes are not used. If direct access fails, add `--gh-prefix https://ghfast.top/` for a GitHub acceleration prefix. To test only one asset class, use `--only image/audio/video`. The assets are hosted by this repository, so availability does not depend on third-party sites retaining the original files.

The Release page contains the complete asset source and license list:

https://github.com/li-xiu-qi/XiaokeAILabs/releases/tag/assets/jev-open-source

## References

- Model repository: https://huggingface.co/akhilaaa3/Jev-Omni
- Base-model repository: https://huggingface.co/google/gemma-4-12B-it

[中文](README.md)
