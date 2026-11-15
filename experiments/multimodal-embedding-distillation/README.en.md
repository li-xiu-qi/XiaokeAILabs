# Image-Text Retrieval Distillation: Jina v5 Teacher and BGE-VL Student

[中文](README.md)

The historical experiment did not set a development-set advantage threshold for choosing a teacher in advance. The current scripts add a teacher check on a separate validation set: before student training, the teacher must lead by at least the specified absolute margin in text-to-image R@1 on the same candidate pool, and the lower bound of the paired per-query bootstrap 95% interval must be above zero. The new workflow has run once on real image-text data; interpret historical results under their original run conditions.

This directory contains experiment code, configuration, run instructions, and result evidence. See the [image-text distillation guide](../../docs/zh/retrieval/embedding/多模态Embedding蒸馏.md) for the goal, method, and results. Run the commands below from this directory.

[`scripts`](scripts/) · [`evidence`](evidence/) · [`assets`](assets/)

## Reproduction and historical record

**Reproduction status:** the original device-side scripts completed the [historical experiment described in the guide](../../docs/zh/retrieval/embedding/多模态Embedding蒸馏.md). The public scripts remove device-private paths and the teacher wrapper and have completed one real-data run in a separately copied and adjusted environment. They require Linux with CUDA and about 50 GiB of available system memory. The image-text teacher depends on custom model code; inspect it before running. The original recorded environment used Python 3.12, PyTorch 2.14.0+cu130, Transformers 5.16.1, Sentence Transformers 6.0.1, NumPy, and Pillow. Changes in environment versions, annotation order, or hardware can affect results.

Install a device-compatible CUDA PyTorch build in an isolated Python environment, then install the other packages needed by the scripts and model downloads:

```bash
python3 -m venv /path/to/venvs/retrieval
source /path/to/venvs/retrieval/bin/activate
# Install CUDA PyTorch for the system architecture and GPU: https://pytorch.org/get-started/locally/
python -m pip install -r requirements.txt
python -m pip check
python -c "import peft; from peft import PeftMixedModel, PeftConfig; print('PEFT:', peft.__version__)"
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA build:', torch.version.cuda, 'GPU:', torch.cuda.is_available()); assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(), 'CUDA GPU with BF16 support is required'"
```

Pinned versions in [`requirements.txt`](requirements.txt) come from the original environment; other packages are not pinned. The Jina teacher code at the pinned revision imports PEFT directly, so the manifest includes an unpinned `peft` version. This combination has not been verified in a fresh environment. `verify.py` uses only the Python standard library.

On 2026-11-14, dependencies were installed from the manifest into a separately copied Linux ARM64 environment and `pip check` passed. Python 3.12.3, PyTorch 2.15.0+cu132, Transformers 5.16.1, Sentence Transformers 6.0.1, PEFT 0.21.3, and datasets 5.2.0 imported, with CUDA and BF16 available. Two data-export unit tests and the historical metric recomputation passed. This is not a clean installation; the environment check itself did not load the real image-text models or reproduce the full experiment.

Original images and annotations are not distributed here. The historical run used Flickr30k. Find the data through the [Flickr30k dataset page](https://huggingface.co/datasets/WhereIsAI/flickr30k-v2) and verify image and annotation terms, then convert it to the script's format. The script below exports the required CSV and images. The original data order is not included, so the historical sample cannot be guaranteed. The annotation CSV needs `filename`, `split`, and `raw` columns; `split` must be `train`, `val`, or `test`, and `raw` must be a Python list of five captions. See the guide's [data structure section](../../docs/zh/retrieval/embedding/多模态Embedding蒸馏.md#数据结构) for the format. Provide at least 512, 128, and 100 rows for the three splits. Filenames are relative to the image directory, and annotation row order affects sampling.

On 2026-11-14, 740 images were exported and prepared from Flickr30k revision `a2ac1f0bcee3132331324f1c435bcea4527c08f1`: 512 training, 128 validation, and 100 test images. The image decoding checks passed and no byte-identical images were found among the selected files. This does not exclude visually near-duplicate images or overlapping provenance.

On the same day, the real-data workflow completed with pinned Jina v5 teacher and BGE-VL student models. On 128 validation queries, teacher and initial-student R@1 were 0.9844 and 0.9141; the 0.0703 gap had a paired bootstrap 95% interval of [0.0156, 0.1250] and passed the preset gate. Supervised and distilled students each trained for two epochs, or 64 batches. In the 100-image test candidate pool, baseline R@1 was 0.96, while supervised, distilled, and teacher R@1 were each 0.99; the two students had identical R@1 outcomes on these 100 queries. The small pool and near-ceiling scores do not establish a distillation advantage or general equivalence. The run used one training seed; visually near-duplicate images and pretraining-data overlap have not been excluded.

For example, export the required sample counts to a new, nonexistent directory. Use `--revision <commit>` to pin a known dataset version; if omitted, the upstream version is not pinned. The first run downloads data and writes images, so allow disk space.

```bash
python scripts/prepare_flickr30k.py --out /path/to/new-flickr-export
```

Without data or model weights, you can run `python -m unittest discover -s tests -v` to check sampling, CSV and image export, and refusal to overwrite existing output. Tests require only Pillow.

You can also check the student training pipeline on a CUDA GPU, using a new, nonexistent output directory outside the repository:

```bash
python scripts/synthetic_gpu_smoke.py --out /path/to/new-synthetic-smoke
```

This creates a randomly initialized small CLIP, synthetic images and captions, and artificial teacher vectors. It then runs `train.py` to check the teacher gate, supervised and distillation training, validation, test, and completion marker. It downloads no dataset or weights and does not run the real teacher; its metrics do not measure model quality or reproduce the historical result. Use a fresh output directory for every run.

On 2026-11-13 this synthetic check passed on NVIDIA GB10, Linux, Python 3.12, PyTorch 2.15.0+cu132, Transformers 5.22.2, tokenizers 0.23.3, Pillow 12.0.0, and NumPy 2.5.4. It used 20 training, 16 validation, and 8 test examples, with four batches each for supervised and distillation training. This reused an existing environment with Pillow installed separately; it was not a fresh installation of the pinned versions above, and did not run the real Jina teacher, BGE-VL, or Flickr30k.

The export directory contains `annotations.csv`, `images/`, and `source.json`, which records source, sampling seed, and row numbers. Set `DATA` to the exported CSV, `IMAGES` to the image directory, and `WORK` to a new working directory outside the repository. Before evaluation, fix the teacher and student models, data, and teacher-margin threshold. Here `0.05` means the teacher must lead by at least five percentage points in R@1; this is the threshold for this design, not a universal standard for all datasets.

```bash
DATA=/path/to/annotations.csv
IMAGES=/path/to/images
WORK=/path/to/new-work-directory
python -c "from huggingface_hub import snapshot_download; snapshot_download('jinaai/jina-embeddings-v5-omni-small', revision='5c54692f22e186fc12ea38a9193e3ff9e1cda2a3', local_dir='/path/to/teacher')"
python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/BGE-VL-base', revision='cc4c733ed997dbee4ac70ccffb911e70c9c24b93', local_dir='/path/to/student')"
python scripts/prepare.py --annotations "$DATA" --images "$IMAGES" --out "$WORK"
python scripts/teacher.py --workdir "$WORK" --images "$IMAGES" --teacher /path/to/teacher --student /path/to/student --min-teacher-gap 0.05 --stage development
python scripts/train.py --workdir "$WORK" --images "$IMAGES" --student /path/to/student --min-teacher-gap 0.05
python scripts/teacher.py --workdir "$WORK" --images "$IMAGES" --teacher /path/to/teacher --stage test
python scripts/verify.py
```

Data preparation samples 512 training, 128 validation, and 100 test examples by image split; it checks that images can be decoded and are not byte-identical duplicates. Before encoding teacher vectors, the pipeline writes model paths, data hashes, and threshold to `teacher-plan.json`, then encodes only the training and validation sets. The training script checks the plan and teacher-vector hashes, compares teacher and initial-student rankings in the same validation candidate pool, and saves `teacher-gate.json`. A path alone does not prove that model weights have not changed; verify model versions separately. If the gate fails, training and test evaluation both stop. If it passes, the fixed-budget supervised and distillation groups run before test-set evaluation. Test-set teacher scoring requires markers showing both groups finished. Do not upload training outputs, image-text descriptions, or model weights to the public repository.

The validation set was not used for gradient updates by this script, but it was used for evaluation in the historical experiment. If you repeatedly adjust the model or threshold, prepare another independent development set and a final test set. Both groups use one training seed. Paired bootstrap intervals reflect sampling uncertainty for the fixed query set, not variation from retraining or dataset selection.

`verify.py` independently recomputes three student metrics and paired intervals from [per-query student rankings](evidence/student-test-ranks.json); this file contains no images, captions, or identities. Only aggregate teacher metrics were retained, so teacher results cannot be recomputed from the ranking file. This checks statistical calculations, not model reruns. Near-duplicate images, overlap with pretraining data, and other valid captions for the same image have not been ruled out.
