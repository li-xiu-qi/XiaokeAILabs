#!/usr/bin/env bash
# 一键运行 MiniCPM-OCR 1.0 + DSpark 推理加速 vs 非加速基线对比评测
set -euo pipefail
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OCR_HOME="${MINICPM_OCR_HOME:-$HOME/minicpm-ocr}"
PYTHON_BIN="${OCR_PYTHON:-$OCR_HOME/.venv-ocr/bin/python}"
RESULT_JSON="${OCR_BENCHMARK_OUTPUT:-$OCR_HOME/runs/acceleration_benchmark.json}"

echo "[*] MiniCPM-OCR 1.0 + DSpark 推理加速对比评测启动..."
mkdir -p "$(dirname "$RESULT_JSON")"

"$PYTHON_BIN" "$BASE/scripts/benchmark_acceleration.py" \
    --target-dir "${OCR_MODEL_DIR:-$OCR_HOME/models/MiniCPM-OCR-1.0}" \
    --dspark-dir "${DSPARK_MODEL_DIR:-$OCR_HOME/models/MiniCPM-OCR-1.0-DSpark}" \
    --testdata-dir "${OCR_TESTDATA_DIR:-$OCR_HOME/testdata}" \
    --examples-dir "${JEV_EXAMPLES_DIR:-$OCR_HOME/repo/scripts/minicpm-ocr-jev/examples}" \
    --output-json "$RESULT_JSON"

echo "[+] 评测完成！本次数据保存在 $RESULT_JSON"
