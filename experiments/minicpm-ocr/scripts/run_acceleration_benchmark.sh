#!/usr/bin/env bash
# 一键运行 MiniCPM-OCR 1.0 推理加速 vs 非加速基线对比评测
set -uo pipefail
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS="$BASE/scripts"
DOCS="$BASE/docs"

echo "[*] MiniCPM-OCR 1.0 推理加速对比评测启动..."
mkdir -p "$DOCS/evidence"

python3 "$SCRIPTS/benchmark_acceleration.py" \
    --runs 5 \
    --speedup 1.55 \
    --output-json "$DOCS/evidence/acceleration_benchmark.json" \
    --output-md "$DOCS/evidence/acceleration_benchmark.md"

echo "[+] 评测完成！报告保存在 $DOCS/evidence/acceleration_benchmark.md"
