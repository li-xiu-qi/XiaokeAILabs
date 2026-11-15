# Whisper ASR Fine-Tuning Experiment

[中文](README.md)

## One small GPU run

On 2026-11-15, an NVIDIA GB10 run with Python 3.12, PyTorch 2.15.0+cu132, Transformers 5.16.1 and Datasets 5.2 completed preparation, baseline evaluation, bf16 fine-tuning and reevaluation of Whisper small on LibriSpeech clean. The model revision was `973afd24965f72e36ca33b3055d56a652f456b4d` and the dataset revision was `71cacbfb7e2354c4226d01e70d77d5fca3d04ba1`. The run used 64 training and 16 test samples, batch size 2, one epoch and 32 updates; after correcting the training labels, mean training loss was 0.4451.

After that correction, word error rate on the same test samples was **3.79% → 3.79%**, so no improvement was observed. The earlier implementation mistakenly masked the end token and produced 5.68%; that is not the final result. The test is small, and Whisper pretraining may have included LibriSpeech; it is not a new blind test and does not establish a general effect. Evaluation used batch size 2 and produced an attention-mask warning; interpret these numbers accordingly. The workflow and parameters below support reproduction, but identical numerical results are not guaranteed.

The training run used a dedicated virtual environment, but its original creation method was not recorded. A separate clean virtual environment was later installed and checked on an NVIDIA GB10 (Linux aarch64) system using the setup below: Python 3.12.3, PyTorch 2.15.0+cu132, Transformers 5.16.1, Datasets 5.2.0, SoundFile 0.14.0, jiwer 4.0.0, NumPy 2.5.4 and tqdm 4.70.1. `pip check` found no conflicts, imports succeeded and CUDA detected the NVIDIA GB10. This setup check did not download the model or dataset and did not run training.

A separate one-shot smoke run used 8 training and 4 test samples, completed 4 GPU updates, and reported WER of 0.87% both before and after fine-tuning. It checks the end-to-end entry point only; it is too small to assess model quality.

Fine-tune a public `openai/whisper-small` (or `whisper-tiny`) model on clean train/test shards of `openslr/librispeech_asr`, and compare word error rate (WER) before and after fine-tuning on the same test set. All scripts live in this directory and only touch this directory plus paths given via environment variables; nothing else in the repository is required.

A principles/motivation article is not written yet; this experiment is not linked from the repo navigation (this change was scoped to this directory only).

## Layout

```
asr-finetuning/
├── README.md / README.en.md   # this document and the Chinese version
├── requirements.txt           # pinned direct dependencies (PyTorch installed separately)
├── run.sh                     # one-shot entry: prepare / train / eval / all
├── scripts/
│   ├── asr_common.py          # run contract, normalization, WER, manifests, artifact protection
│   ├── prepare_librispeech.py # pinned-revision data prep (16 kHz, duration filter, split isolation)
│   ├── train_whisper.py       # fine-tuning loop (CUDA/CPU, bf16, checkpoint per epoch)
│   ├── eval_wer.py            # WER evaluation on the test manifest (baseline or fine-tuned model)
│   └── compare_wer.py         # compare two evaluation results
└── tests/                     # offline unit tests (no model loading, no downloads)
```

## Scope and architecture

- Data: LibriSpeech ASR `clean` config; train on `train.100`, test on `test` (splits are configurable).
- Input handling: all audio is resampled to 16 kHz mono PCM_16 WAV; transcripts are normalized (`normalize_text`: NFKC, lowercase, strip punctuation, collapse whitespace); duration filtering defaults to 0.5–30 s to match Whisper's 30-second window.
- Training: `WhisperForConditionalGeneration` with a custom PyTorch loop (AdamW, gradient clipping, bf16 autocast, checkpoint per epoch); `transformers.Trainer` is not used.
- Evaluation: greedy or beam-search transcription; references and hypotheses are normalized the same way and scored with jiwer WER. The baseline model and the fine-tuned model are evaluated on the same test manifest, and `compare_wer.py` reports the delta.
- Isolation and protection: sample ids must not overlap between train and test manifests (checked before training); the output directory must be a new or empty directory, otherwise the run is refused so existing artifacts are never overwritten.

## Hardware and environment

- Target environment: NVIDIA GB10 (Grace Blackwell, Linux aarch64) with a standalone CUDA virtual environment. A single GPU suffices; small-sample fine-tuning of whisper-small needs a few GB of VRAM, whisper-tiny less.
- Other Linux x86_64 + CUDA setups should work but are untested. CPU is smoke-test only.
- Dependencies: This run used Python 3.12.3. To reproduce the GB10 environment above, create a clean virtual environment, then install the same PyTorch CUDA 13.2 wheel and pinned direct dependencies. The official PyTorch index provides a Linux aarch64 wheel. For other GPUs/operating systems, choose a compatible build using the [official PyTorch instructions](https://pytorch.org/get-started/locally/). PyTorch is installed separately and is not in this experiment's requirements, so the manifest will not replace a platform-specific build.

```bash
python3 -m venv /path/to/venvs/asr
source /path/to/venvs/asr/bin/activate
# GB10 / Linux aarch64: reproduce the PyTorch version used in this run
python -m pip install --index-url https://download.pytorch.org/whl/cu132 "torch==2.15.0+cu132"
python -m pip install -r requirements.txt
python -m pip check
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
python -c "import transformers, datasets, soundfile, jiwer; print('imports ok')"
```

- Clean-install check: On 2026-11-15, a new isolated virtual environment on an NVIDIA GB10 (Linux aarch64) system was installed using the PyTorch index and this `requirements.txt`; `pip check`, imports and the CUDA check passed. This validation did not download the model or dataset or run training.
- Direct dependencies are pinned to the validated versions; transitive dependencies do not have a complete lock file yet. Whisper's generate arguments differ across transformers 4.x and 5.x; recheck output if upgrading.

## Model and data sources

- Model: `openai/whisper-small` (Hugging Face; weights under Apache-2.0, OpenAI Whisper code under MIT). A commit pin via `ASR_MODEL_REVISION` is required by default; a local downloaded directory also works. Read the model card for licensing before loading.
- Data: `openslr/librispeech_asr` (Hugging Face dataset copy of LibriSpeech, `clean` config). The LibriSpeech corpus is CC BY 4.0 (see <https://www.openslr.org/12/>); the dataset card on Hugging Face is authoritative. `ASR_DATASET_REVISION` is required by default.
- How to resolve revision pins (examples):

```bash
python -c "from huggingface_hub import HfApi; print(HfApi().model_info('openai/whisper-small').sha)"
python -c "from huggingface_hub import HfApi; print(HfApi().dataset_info('openslr/librispeech_asr').sha)"
```

- After weighing the reproducibility impact you may disable the pin with `ASR_REQUIRE_REVISION=0` or per-script `--allow-floating-revision`; this is not recommended.

## Commands

Commands below can run from any directory (scripts resolve their own location). Defaults are small-sample: 200 train / 50 test utterances, 3 epochs.

### One-shot entry

```bash
REPO=/path/to/XiaokeAILabs
EXP="$REPO/experiments/asr-finetuning"
WORK=/path/to/asr-work
mkdir -p "$WORK"

export XL_OUT_DIR="$WORK/out/run-001"
export XL_DATA_DIR="$WORK/data"
export XL_RUN_ID="run-001"
export ASR_MODEL="openai/whisper-small"
export ASR_MODEL_REVISION="<model commit from above>"
export ASR_DATASET="openslr/librispeech_asr"
export ASR_DATASET_REVISION="<dataset commit from above>"

bash "$EXP/run.sh" all
```

To use the model and data revisions and sample sizes from the GPU run above, set these before running the command:

```bash
export ASR_MODEL_REVISION=973afd24965f72e36ca33b3055d56a652f456b4d
export ASR_DATASET_REVISION=71cacbfb7e2354c4226d01e70d77d5fca3d04ba1
export ASR_MAX_TRAIN_SAMPLES=64 ASR_MAX_TEST_SAMPLES=16 ASR_SHUFFLE_BUFFER=64
export ASR_EPOCHS=1 ASR_BATCH_SIZE=2
```

### Step by step (rerun individual stages)

```bash
# 1) data prep: streaming download of the needed shards, 16 kHz WAV + manifests
python "$EXP/scripts/prepare_librispeech.py" \
  --dataset openslr/librispeech_asr --dataset-revision "<dataset commit>" \
  --train-split train.100 --test-split test \
  --max-train-samples 200 --max-test-samples 50 \
  --output-dir "$XL_DATA_DIR/manifests/$XL_RUN_ID"

# 2) training (output directory must be new)
python "$EXP/scripts/train_whisper.py" \
  --train-manifest "$XL_DATA_DIR/manifests/$XL_RUN_ID/train.jsonl" \
  --test-manifest "$XL_DATA_DIR/manifests/$XL_RUN_ID/test.jsonl" \
  --model openai/whisper-small --model-revision "<model commit>" \
  --output-dir "$XL_OUT_DIR/model" --epochs 3 --batch-size 8

# 3) baseline WER (before fine-tuning, same test set)
python "$EXP/scripts/eval_wer.py" \
  --model openai/whisper-small --model-revision "<model commit>" \
  --manifest "$XL_DATA_DIR/manifests/$XL_RUN_ID/test.jsonl" \
  --output "$XL_OUT_DIR/wer-baseline.json"

# 4) WER after fine-tuning (same test set)
python "$EXP/scripts/eval_wer.py" \
  --model "$XL_OUT_DIR/model/final-model" --model-revision "" --allow-floating-revision \
  --manifest "$XL_DATA_DIR/manifests/$XL_RUN_ID/test.jsonl" \
  --output "$XL_OUT_DIR/wer-finetuned.json"

# 5) before/after comparison
python "$EXP/scripts/compare_wer.py" \
  --baseline "$XL_OUT_DIR/wer-baseline.json" \
  --finetuned "$XL_OUT_DIR/wer-finetuned.json" \
  --output "$XL_OUT_DIR/wer-comparison.json"
```

### Smoke profile (minimal verification)

```bash
ASR_MODEL=openai/whisper-tiny ASR_MODEL_REVISION="$(python -c "from huggingface_hub import HfApi; print(HfApi().model_info('openai/whisper-tiny').sha)")" \
ASR_MAX_TRAIN_SAMPLES=8 ASR_MAX_TEST_SAMPLES=8 ASR_EPOCHS=1 ASR_BATCH_SIZE=2 \
bash "$EXP/run.sh" all
```

The command looks up the current tiny model revision; set `ASR_DATASET_REVISION` first. Tiny and small use different model revisions. The smoke profile downloads the model and some data; run it on the GPU machine.

## Environment variables

| Variable | Purpose | Default |
|---|---|---|
| `XL_OUT_DIR` | output root | `.work/out/$XL_RUN_ID` under the experiment directory |
| `XL_DATA_DIR` | data directory (manifests, WAV) | `.work/data` under the experiment directory |
| `XL_RUN_ID` | run id, used in manifest dir and metrics | `local` |
| `XL_REQUIRE_EXTERNAL_ENV` | `=1` forbids the `.work/` fallback and errors out when not injected | unset |
| `ASR_MODEL` | model id or local path | `openai/whisper-small` |
| `ASR_MODEL_REVISION` | model commit pin | none (required by default) |
| `ASR_DATASET` | dataset id or local path | `openslr/librispeech_asr` |
| `ASR_DATASET_REVISION` | dataset commit pin | none (required by default) |
| `ASR_REQUIRE_REVISION` | `=0` disables mandatory revision pins | `1` |
| `ASR_TRAIN_SPLIT` / `ASR_TEST_SPLIT` | train/test splits | `train.100` / `test` |
| `ASR_MAX_TRAIN_SAMPLES` / `ASR_MAX_TEST_SAMPLES` | sample caps, 0 = full split | `200` / `50` |
| `ASR_SHUFFLE_BUFFER` | Streaming shuffle buffer size | `1000` |
| `ASR_EPOCHS` / `ASR_BATCH_SIZE` | epochs / batch size; the one-shot flow also uses this batch size for evaluation | `3` / `8` |

In controlled environments (CI, compute nodes) set `XL_REQUIRE_EXTERNAL_ENV=1` and inject `XL_OUT_DIR` / `XL_DATA_DIR` / `XL_RUN_ID` explicitly to avoid accidental local writes.

## Verification

```bash
# offline unit tests (no GPU, no model or data downloads)
cd "$EXP" && python -m unittest discover -s tests -v

# artifact inspection
python -c "import json; print(json.load(open('$XL_OUT_DIR/wer-baseline.json'))['wer_percent'])"
python -c "import json; print(json.load(open('$XL_OUT_DIR/wer-comparison.json')))"
```

Key artifacts: `prepare_summary.json` (data and filter settings), `model/metrics.json` (loss curve, not WER), `wer-*.json` (WER plus worst examples), `wer-comparison.json` (before/after delta).

## Known limitations and unverified items

- Direct dependencies are pinned to the validated GB10 versions; select a suitable PyTorch build for other platforms. The clean-environment check only covers Linux aarch64 / Python 3.12.3 / CUDA 13.2; transitive packages remain unlocked.
- WER normalization is simplified (no number or abbreviation expansion); cross-paper numbers may use a different convention.
- `prepare` uses streaming shuffle with a bounded buffer; when the sample cap exceeds the buffer, randomness is weaker (the script warns).
- Train/test isolation is checked by sample id only; speaker leakage across custom splits is not detected. The official LibriSpeech splits are speaker-disjoint; keep that property when defining your own.
- Single GPU, single process; no fp16 GradScaler (use bf16), no gradient accumulation, no LR schedule or early stopping. `metrics.json` loss is a training-batch loss, not a generalization metric.
- Windows is untested; `run.sh` needs Bash (Git Bash works).
