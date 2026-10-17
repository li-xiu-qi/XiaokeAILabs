#!/usr/bin/env bash
# Wan2.2-TI2V-5B LoRA 学习训练启动脚本（ai-toolkit）
# 用法：bash scripts/run_train.sh [config 名，默认 rzqduck_wan22_ti2v_lora]
set -u

HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"

CFG="${1:-rzqduck_wan22_ti2v_lora}"
export HF_HUB_DISABLE_XET=1
export HF_ENDPOINT=https://hf-mirror.com

PY=/home/ke/jevomni-env/bin/python

mkdir -p output output_smoke
bash ~/preflight.sh "rzqduck-$CFG" 50 || exit 1

echo "=== 启动训练: $CFG ==="
$PY /home/ke/ai-toolkit-main/run.py "config/${CFG}.yaml" 2>&1 | tee "output/${CFG}.log"
