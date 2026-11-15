# Retrieving Document Pages with ColSmol: A Multi-Vector Matching Evaluation

[中文](README.md)

The historical visual-distillation script lacks a teacher-versus-student advantage gate on an independent development set, so it stops before training. A same-condition teacher/student evaluation and a predeclared advantage criterion must be added first. Completed historical results should still be interpreted under their original run conditions.

This directory contains experiment code, data preparation, run instructions, and result evidence. See [Visual Multi-Vector Retrieval](../../docs/zh/retrieval/colbert/视觉多向量检索.md) for the use case and method, [Supervised Fine-Tuning](../../docs/zh/retrieval/colbert/视觉多向量检索模型微调.md) for the fine-tuning objective, and [Distillation](../../docs/zh/retrieval/colbert/视觉多向量检索模型蒸馏.md) for the distillation objective. Run the commands below from this directory.

## Data and baseline

The [ViDoRe DocVQA test subset](https://huggingface.co/datasets/vidore/docvqa_test_subsampled) contains document-page images and questions about those pages, covering tables, reports, and forms. In dataset order, the first 100 distinct `image_filename` values are selected, using the first question for each. Each question ranks the same set of 100 pages, with its paired page as the only known positive.

| Metric | Result |
|---|---:|
| Recall@1 | 0.55 |
| Recall@5 | 0.78 |
| nDCG@5 | 0.6664 |

Other pages may also be relevant but unlabeled. The candidate set is limited to the selected 100 pages and does not represent full-corpus recall. The original model was run directly without fine-tuning for this evaluation. The model card says its training set did not use the same multi-page PDFs as ViDoRe, but overlap with the base model's pretraining data cannot be ruled out.

### Environment setup

The original run used Linux ARM64, NVIDIA GB10, BF16, PyTorch 2.14.0+cu130, Transformers 4.56.2, ColPali Engine 0.3.13, PEFT 0.17.1, and Datasets 4.1.1. The baseline and training scripts read Linux `/proc/meminfo` and load the model on CUDA in BF16, so they require an NVIDIA GPU supporting BF16, with compatible drivers and CUDA-enabled PyTorch. There is no CPU path in the current scripts. The scripts require at least 50 GiB of system available memory and exit if available memory falls below that value during a run. This is a guardrail from the original device, not a general minimum hardware requirement. Models, datasets, and run outputs require additional disk space.

The recorded PyTorch 2.14.0 does not match the `torch>=2.2,<2.9` requirement declared in the [ColPali Engine 0.3.13 release metadata](https://pypi.org/project/colpali-engine/0.3.13/). Directly installing the requirements below may replace PyTorch, so they must not be treated as steps to rebuild the historical environment. Why the historical combination ran is still unknown.

The following is a candidate combination that satisfies the package version constraint: Linux x86_64, CUDA 12.8 wheels, PyTorch 2.8.0, and torchvision 0.23.0, paired according to the [official PyTorch table](https://pytorch.org/get-started/previous-versions/). Confirm that the device architecture, Python, driver, and GPU support these wheels. As checked, the official PyTorch CUDA 12.8 index has no Linux aarch64 wheel for this version, so it cannot be used on the historical GB10; ARM64 requires another compatible build. This combination has not been installed from scratch or GPU-validated for this experiment, and it is not the historical environment. The dataset, base model, and adapter are downloaded from upstream and are not distributed in this repository:

```bash
WORK=/path/to/independent-workdir
mkdir -p "$WORK"
python3 -m venv "$WORK/venv"
source "$WORK/venv/bin/activate"
# Candidate for Linux x86_64 with CUDA 12.8; find a compatible build separately for ARM64
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
# Constrain the selected versions so the resolver does not replace the device-specific torch build
python -m pip install -r requirements.txt 'torch==2.8.0' 'torchvision==0.23.0'
python -c "import torch; print('torch:', torch.__version__, 'CUDA:', torch.version.cuda, 'available:', torch.cuda.is_available()); print('BF16:', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)"
python -m pip check
```

Proceed only after confirming CUDA and BF16 are both available. `prepare_data.py` defaults to dataset revision `49bf8f13e13c41dd8cdb0cae5314e31c1da1e0d6`; the first download requires network access.

```bash
hf download vidore/colSmol-256M --revision c79b633e17e060cc109b11aa2bf92a1517d3bd6f --local-dir "$WORK/model"
hf download vidore/ColSmolVLM-Instruct-256M-base --revision 99ca96f1f6b95b3a69e6abef74a2416cb738fed0 --local-dir "$WORK/base"
python scripts/prepare_data.py --output "$WORK/data"
python scripts/baseline.py --data "$WORK/data" --base "$WORK/base" --model "$WORK/model" --out "$WORK/smoke.json" --count 10
python scripts/baseline.py --data "$WORK/data" --base "$WORK/base" --model "$WORK/model" --out "$WORK/baseline100.json" --count 100
```

The released ColSmol weights are an adapter: load the specified base model first, then the adapter. The base model alone is not a ready-to-use retriever. The 10-page run is an environment check; the 100-page run corresponds to the baseline. The model card notes that the current ColPali Engine prompt format differs from the format used during model training. This run used version 0.3.13 and cannot be directly compared with the model-card score. Device and software differences may also change rankings.

The test took 78.12 seconds, including encoding and pairwise scoring but excluding model and data loading. Peak PyTorch allocation was about 2.81 GiB; this is not total system memory use. The script retains the 50 GiB available-memory guard from the original machine; it is not a general minimum requirement.

[Anonymous per-query ranks](evidence/baseline100.json) records the paired-page rank for 100 queries and does not include source images, questions, or dataset IDs. Run `python scripts/recompute.py` to recompute the aggregate metrics; this cannot validate model scores or labels.

## Supervised fine-tuning on synthetic reports

This experiment generated 160 fictional operational reports, with three fields per page, split by entity into 96 training, 32 validation, and 32 test pages. On 96 test queries, Recall@1 was 87/96 before training and 88/96 after training. See the [fine-tuning guide](../../docs/zh/retrieval/colbert/视觉多向量检索模型微调.md) for shared-template and single-training-seed limitations.

```bash
python scripts/generate_synthetic.py --output "$WORK/synthetic"
HF_HUB_OFFLINE=1 python scripts/train_synthetic.py --root "$WORK/synthetic" --base "$WORK/base" --adapter "$WORK/model" --out "$WORK/train-smoke" --smoke
HF_HUB_OFFLINE=1 python scripts/train_synthetic.py --root "$WORK/synthetic" --base "$WORK/base" --adapter "$WORK/model" --out "$WORK/train-formal" --steps 8
python scripts/recompute_training.py
```

Training uses the baseline model and software environment described above. The page-generation script reads fonts from fixed paths `/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf` and `/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf`; install the corresponding fonts and check that both files exist before running. Font versions may change page pixels. [synthetic-train-results.json](evidence/synthetic-train-results.json) stores per-query ranks and losses, not model weights. Recalculation checks record consistency; ranks alone cannot verify model outputs.

On 2026-11-13, a pipeline check on Linux ARM64 with NVIDIA GB10, Python 3.12, PyTorch 2.11.0+cu130, torchvision 0.26.0+cu130, and the pinned Transformers, ColPali Engine, and PEFT versions above loaded the pinned base model and adapter. It completed evaluation before training, two CUDA fine-tuning steps on synthetic pages, evaluation afterwards, and adapter saving. An additional check confirmed that all 420 parameter tensors with gradients changed in each optimizer step. This checks the training path only; it does not reproduce the 100-page baseline or the formal fine-tuning results.

The check layered dependencies over an existing environment and failed `pip check`: PyTorch 2.11 does not satisfy the `torch<2.9` requirement declared by ColPali Engine 0.3.13, SciPy was missing, and other package versions conflicted. It is not a clean, reproducible installation and does not establish compatibility for other tasks. The candidate installation above remains unverified.

## Distillation experiment record

The historical experiment used scores from a multimodal reranking teacher for training and compared the result with a supervised group using the same budget. Some test queries were inspected when selecting the teacher, so the results are not validation on a fully untouched test set. See the [distillation guide](../../docs/zh/retrieval/colbert/视觉多向量检索模型蒸馏.md) for details. The current historical training entry point stops until a teacher/student evaluation on an independent development set is added; do not use it to rerun formal distillation.

The trained adapter is not distributed in this repository. The [independent-page retest](independent-evaluation/README.md) describes the three required adapters, base model, old data, and recomputation method. `distill_synthetic.py` stops immediately at entry, so the current repository alone cannot produce a distillation adapter.

[qwen-train-scores.json](evidence/qwen-train-scores.json) stores synthetic training batches and teacher scores; [synthetic-distill-results.json](evidence/synthetic-distill-results.json) stores anonymous per-query ranks. Run `python scripts/recompute_distillation.py` to check historical metrics against batch records; this does not reproduce model inference. Teacher and student use separate environments. The original teacher used BF16 with a maximum image size of 65,536 pixels. Teacher and training weights are not distributed with the repository.

To regenerate teacher scores for training batches, install mutually compatible CUDA PyTorch and torchvision in a separate Python environment, then use [requirements-teacher.txt](requirements-teacher.txt) and the commands below. The teacher environment's Transformers version is based on a Qwen reranking run in the same series and has not been validated from scratch for this experiment. It differs from the student environment above; do not mix them. `scripts/qwen3_vl_reranker.py` documents the source of and modifications to the official Qwen implementation. The upstream Apache-2.0 license text is in [LICENSE-QWEN-APACHE-2.0](LICENSE-QWEN-APACHE-2.0). This step only generates candidate training scores; it does not implement the teacher-advantage gate or remove the stop condition for distillation training.

```bash
python3 -m venv "$WORK/teacher-venv"
source "$WORK/teacher-venv/bin/activate"
# CUDA 13.0 example using the corresponding official versions; adjust for other devices using the official guide
python -m pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements-teacher.txt
python -m pip check
python -c 'import torch, torchvision, qwen_vl_utils; print("torch:", torch.__version__, "torchvision:", torchvision.__version__, "CUDA:", torch.cuda.is_available())'
hf download Qwen/Qwen3-VL-Reranker-2B --revision 4bd860ac4f15ad1897a214615cccc700f8f71818 --local-dir "$WORK/teacher-model"
python scripts/qwen_train_scores.py --root "$WORK/synthetic" --model "$WORK/teacher-model" --out "$WORK/new-teacher-scores.json"
```

The CUDA 13.0 example comes from the [official PyTorch version table](https://pytorch.org/get-started/previous-versions/). Check that the device driver, Python version, and CPU architecture have a compatible wheel; this installation combination has not been validated from scratch for this experiment. The image-processing code in `qwen-vl-utils` 0.0.14 requires torchvision, but the package metadata does not declare it, so `pip check` alone is insufficient.

Verify the weight hashes after downloading. `--root` must contain the previously generated `records.json` and sibling image directory. Score output contains queries and image paths, so write it under the working directory.

On 2026-11-14, the pinned teacher above ran on Linux ARM64 and NVIDIA GB10 after generating 160 synthetic pages. It scored eight batches from the training split with CUDA BF16, each containing two queries and two candidate images, producing 32 finite scores. The separate copied and adjusted teacher environment passed `pip check` and import checks with Python 3.12.3, PyTorch 2.15.0+cu132, torchvision 0.29.1+cu130, Transformers 5.16.1, and qwen-vl-utils 0.0.14; it was not installed from scratch. This verifies training-batch teacher scoring only. It does not establish teacher superiority on a development set or lift the formal distillation stop.

## Licenses

The model and base-model repository metadata are marked MIT. The base-model card says its backbone is Apache-2.0 and its adapter is MIT; the model-card license text also contains ColQwen2 wording. Check the license for the exact files before redistributing weights. The ViDoRe dataset card is marked MIT, and the applicable terms for the original DocVQA data and source documents must still be followed. This repository does not distribute dataset pages, question text, or model weights.
