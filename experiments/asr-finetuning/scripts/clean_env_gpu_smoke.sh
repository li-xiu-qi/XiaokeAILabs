#!/usr/bin/env bash
# Fresh-environment end-to-end GPU smoke run for the pinned Whisper experiment.
set -Eeuo pipefail

for name in XL_RUN_ID XL_DATA_DIR XL_OUT_DIR; do
  [[ -n "${!name:-}" ]] || { echo "FATAL: $name is required" >&2; exit 2; }
done
[[ "${XL_REQUIRE_EXTERNAL_ENV:-0}" == "1" ]] || {
  echo "FATAL: set XL_REQUIRE_EXTERNAL_ENV=1 and use external data/output directories" >&2; exit 2;
}
[[ "$XL_RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "FATAL: invalid XL_RUN_ID" >&2; exit 2; }

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXP_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3.12}
TORCH_INDEX_URL=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu132}
OUT_DIR=$XL_OUT_DIR/asr-clean-$XL_RUN_ID
VENV_DIR=${VENV_DIR:-$XL_DATA_DIR/venvs/asr-clean-$XL_RUN_ID}
HF_HOME=$XL_DATA_DIR/huggingface

mkdir -p "$XL_DATA_DIR" "$(dirname "$VENV_DIR")"
[[ ! -e "$OUT_DIR" && ! -e "$VENV_DIR" ]] || {
  echo "FATAL: refusing to reuse an existing environment or output path" >&2; exit 2;
}

"$PYTHON_BIN" -m venv "$VENV_DIR"
PY=$VENV_DIR/bin/python
"$PY" -m pip install --disable-pip-version-check pip==24.0
"$PY" -m pip install --disable-pip-version-check --index-url "$TORCH_INDEX_URL" "torch==2.15.0+cu132"
mkdir -p "$XL_DATA_DIR/constraints"
"$PY" -c 'import importlib.metadata as m; print("torch==" + m.version("torch"))' > "$XL_DATA_DIR/constraints/torch-$XL_RUN_ID.txt"
"$PY" -m pip install --disable-pip-version-check -c "$XL_DATA_DIR/constraints/torch-$XL_RUN_ID.txt" -r "$EXP_DIR/requirements.txt"
"$PY" -m pip check
"$PY" -c 'import torch; print("torch",torch.__version__,"CUDA",torch.version.cuda,"available",torch.cuda.is_available(),"BF16",torch.cuda.is_bf16_supported()); assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()'
"$PY" -c 'import transformers, datasets, soundfile, jiwer; print("ASR imports ok", transformers.__version__, datasets.__version__)'

export PYTHON=$PY XL_OUT_DIR=$OUT_DIR HF_HOME
export ASR_MODEL=openai/whisper-small
export ASR_MODEL_REVISION=973afd24965f72e36ca33b3055d56a652f456b4d
export ASR_DATASET=openslr/librispeech_asr
export ASR_DATASET_REVISION=71cacbfb7e2354c4226d01e70d77d5fca3d04ba1
export ASR_MAX_TRAIN_SAMPLES=64 ASR_MAX_TEST_SAMPLES=16 ASR_SHUFFLE_BUFFER=64
export ASR_EPOCHS=1 ASR_BATCH_SIZE=2
export XL_REQUIRE_EXTERNAL_ENV=1
bash "$EXP_DIR/run.sh" all

"$PY" - "$OUT_DIR" "$XL_DATA_DIR/manifests/$XL_RUN_ID" <<'PY'
import importlib.metadata as metadata
import json
import os
import platform
import sys
from pathlib import Path
import torch

out, manifest_dir = map(Path, sys.argv[1:])
train = json.loads((out / "model" / "metrics.json").read_text(encoding="utf-8"))
baseline = json.loads((out / "wer-baseline.json").read_text(encoding="utf-8"))
finetuned = json.loads((out / "wer-finetuned.json").read_text(encoding="utf-8"))
comparison = json.loads((out / "wer-comparison.json").read_text(encoding="utf-8"))
def rows(name):
    return sum(1 for line in (manifest_dir / name).open(encoding="utf-8") if line.strip())

summary = {
    "date": "2026-11-15",
    "purpose": "Fresh-environment end-to-end Whisper small fine-tuning smoke run on pinned LibriSpeech samples.",
    "hardware": torch.cuda.get_device_name(0),
    "environment": {
        "platform": platform.system(), "architecture": platform.machine(),
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
        "bfloat16_supported": torch.cuda.is_bf16_supported(),
        "transformers": metadata.version("transformers"), "datasets": metadata.version("datasets"),
        "soundfile": metadata.version("soundfile"), "jiwer": metadata.version("jiwer"),
        "pip_check": "passed"
    },
    "model_revision": "973afd24965f72e36ca33b3055d56a652f456b4d",
    "dataset_revision": "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1",
    "samples": {"train": rows("train.jsonl"), "test": rows("test.jsonl")},
    "training": {
        "epochs": train.get("epochs"), "batch_size": train.get("batch_size"),
        "updates": ((train["n_train_samples"] + train["batch_size"] - 1) // train["batch_size"]) * train["epochs"],
        "final_avg_loss": train.get("final_avg_loss"),
        "amp": train.get("amp"), "device": train.get("device"),
        "wall_seconds": train.get("wall_seconds")
    },
    "wer_percent": {
        "baseline": baseline["wer_percent"], "finetuned": finetuned["wer_percent"],
        "delta_percentage_points": comparison["delta_percent_points"],
        "improved": comparison["improved"],
        "same_test_manifest": comparison["manifest_match"],
        "test_samples": baseline.get("n_samples")
    },
    "limits": [
        "This is a small smoke run, not a general ASR quality estimate.",
        "Whisper pretraining overlap with LibriSpeech was not excluded.",
        "The tested sample is small and does not reproduce the full-data experiment."
    ]
}
(out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"result": "passed", "train": summary["samples"]["train"], "test": summary["samples"]["test"], "wer_percent": summary["wer_percent"]}, ensure_ascii=False))
PY
