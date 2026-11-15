#!/usr/bin/env bash
# run.sh — ASR 微调实验一键入口（prepare / train / eval / all）
#
# 运行契约（与 scripts/asr_common.py 一致）：
#   XL_OUT_DIR / XL_DATA_DIR / XL_RUN_ID 指定输出目录、数据目录与运行标识；
#   ASR_MODEL / ASR_MODEL_REVISION 指定模型与固定版本；
#   ASR_DATASET / ASR_DATASET_REVISION 指定数据集与固定版本；
#   未显式指定时回退到实验目录下被 .gitignore 忽略的 .work/；
#   XL_REQUIRE_EXTERNAL_ENV=1 时禁止本地回退并要求显式注入。
#
# 用法（任意目录执行，脚本按自身位置解析）：
#   bash run.sh all                 # prepare + train + 基线/微调 WER + 对比
#   bash run.sh prepare             # 只生成 16 kHz manifest
#   bash run.sh train               # 只训练（需先 prepare）
#   bash run.sh eval                # 只评估（基线 + 微调 + 对比，需先 train）
#   ASR_MODEL=openai/whisper-tiny bash run.sh all
#   XL_OUT_DIR=/path/out XL_DATA_DIR=/path/data XL_RUN_ID=run-001 bash run.sh all
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
STAGE="${1:-all}"

case "$STAGE" in
  prepare|train|eval|all) ;;
  *) echo "[run] 未知阶段: $STAGE（可选 prepare|train|eval|all）" >&2; exit 2 ;;
esac

if [[ "${XL_REQUIRE_EXTERNAL_ENV:-0}" == "1" ]]; then
  for name in XL_RUN_ID XL_DATA_DIR XL_OUT_DIR; do
    if [[ -z "${!name:-}" ]]; then
      echo "[run] $name 未设置（XL_REQUIRE_EXTERNAL_ENV=1 禁止本地回退）" >&2
      exit 1
    fi
  done
fi

export XL_RUN_ID="${XL_RUN_ID:-local}"
export XL_DATA_DIR="${XL_DATA_DIR:-$SCRIPT_DIR/.work/data}"
export XL_OUT_DIR="${XL_OUT_DIR:-$SCRIPT_DIR/.work/out/$XL_RUN_ID}"

MODEL="${ASR_MODEL:-openai/whisper-small}"
MODEL_REV="${ASR_MODEL_REVISION:-}"
DATASET="${ASR_DATASET:-openslr/librispeech_asr}"
DATA_REV="${ASR_DATASET_REVISION:-}"
TRAIN_SPLIT="${ASR_TRAIN_SPLIT:-train.100}"
TEST_SPLIT="${ASR_TEST_SPLIT:-test}"
MAX_TRAIN="${ASR_MAX_TRAIN_SAMPLES:-200}"
MAX_TEST="${ASR_MAX_TEST_SAMPLES:-50}"
SHUFFLE_BUFFER="${ASR_SHUFFLE_BUFFER:-1000}"
EPOCHS="${ASR_EPOCHS:-3}"
BATCH_SIZE="${ASR_BATCH_SIZE:-8}"

FLOAT_FLAG=""
if [[ "${ASR_ALLOW_FLOATING_REVISION:-0}" == "1" ]]; then
  FLOAT_FLAG="--allow-floating-revision"
fi

MANIFEST_DIR="${ASR_MANIFEST_DIR:-$XL_DATA_DIR/manifests/$XL_RUN_ID}"
TRAIN_MANIFEST="$MANIFEST_DIR/train.jsonl"
TEST_MANIFEST="$MANIFEST_DIR/test.jsonl"
MODEL_DIR="${ASR_MODEL_DIR:-$XL_OUT_DIR/model}"

MODEL_TAG="$MODEL@${MODEL_REV:-<floating>}"
DATA_TAG="$DATASET@${DATA_REV:-<floating>}"

if [[ "$STAGE" == "prepare" || "$STAGE" == "all" ]]; then
  echo "[run] [prepare] dataset=$DATA_TAG config=clean splits=$TRAIN_SPLIT/$TEST_SPLIT"
  echo "[run] [prepare] manifests -> $MANIFEST_DIR"
  "$PY" "$SCRIPT_DIR/scripts/prepare_librispeech.py" \
    --dataset "$DATASET" --dataset-revision "$DATA_REV" \
    --dataset-config clean \
    --train-split "$TRAIN_SPLIT" --test-split "$TEST_SPLIT" \
    --max-train-samples "$MAX_TRAIN" --max-test-samples "$MAX_TEST" \
    --shuffle-buffer "$SHUFFLE_BUFFER" \
    --output-dir "$MANIFEST_DIR" $FLOAT_FLAG
fi

if [[ "$STAGE" == "train" || "$STAGE" == "all" ]]; then
  if [[ ! -f "$TRAIN_MANIFEST" ]]; then
    echo "[run] 缺少 $TRAIN_MANIFEST，请先运行: bash run.sh prepare" >&2
    exit 1
  fi
  echo "[run] [train] model=$MODEL_TAG -> $MODEL_DIR"
  "$PY" "$SCRIPT_DIR/scripts/train_whisper.py" \
    --train-manifest "$TRAIN_MANIFEST" --test-manifest "$TEST_MANIFEST" \
    --model "$MODEL" --model-revision "$MODEL_REV" \
    --output-dir "$MODEL_DIR" \
    --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" $FLOAT_FLAG
fi

if [[ "$STAGE" == "eval" || "$STAGE" == "all" ]]; then
  if [[ ! -f "$TEST_MANIFEST" ]]; then
    echo "[run] 缺少 $TEST_MANIFEST，请先运行: bash run.sh prepare" >&2
    exit 1
  fi
  BASE_OUT="${ASR_BASELINE_OUTPUT:-$XL_OUT_DIR/wer-baseline.json}"
  FT_OUT="${ASR_FINETUNED_OUTPUT:-$XL_OUT_DIR/wer-finetuned.json}"
  COMPARE_OUT="${ASR_COMPARE_OUTPUT:-$XL_OUT_DIR/wer-comparison.json}"
  echo "[run] [eval] 基线 $MODEL_TAG on $TEST_MANIFEST"
  "$PY" "$SCRIPT_DIR/scripts/eval_wer.py" \
    --model "$MODEL" --model-revision "$MODEL_REV" \
    --manifest "$TEST_MANIFEST" --output "$BASE_OUT" --batch-size "$BATCH_SIZE" $FLOAT_FLAG
  echo "[run] [eval] 微调模型 $MODEL_DIR/final-model on $TEST_MANIFEST"
  "$PY" "$SCRIPT_DIR/scripts/eval_wer.py" \
    --model "$MODEL_DIR/final-model" --model-revision "" --allow-floating-revision \
    --manifest "$TEST_MANIFEST" --output "$FT_OUT" --batch-size "$BATCH_SIZE"
  echo "[run] [compare] -> $COMPARE_OUT"
  "$PY" "$SCRIPT_DIR/scripts/compare_wer.py" \
    --baseline "$BASE_OUT" --finetuned "$FT_OUT" --output "$COMPARE_OUT"
fi

if [[ "$STAGE" != "all" ]]; then
  echo "[run] 阶段 $STAGE 完成。产物见 XL_OUT_DIR=$XL_OUT_DIR"
fi
