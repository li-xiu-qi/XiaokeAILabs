#!/bin/bash
# 投机解码实验的统一服务启动脚本
#
# 用法：
#   bash scripts/start-server.sh base    # 无投机解码（基线）
#   bash scripts/start-server.sh mtp     # + MTP 草稿头
#   bash scripts/start-server.sh dflash  # + DFlash2 草稿模型
#
# 所有路径可用环境变量覆盖，默认值面向 GB10 统一内存设备的常见布局。
# 关键约束：--parallel 必须为 1（-np 1）。投机解码要求 single-batch 模式，
# 并发数大于 1 时引擎会强制 kv_unified 并分裂 KV，草稿接受率显著劣化。

set -euo pipefail

MODE="${1:-base}"
LLAMA_BIN="${LLAMA_BIN:-$HOME/llama.cpp/build/bin/llama-server}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/flashnext}"
MAIN_MODEL="${MAIN_MODEL:-Qwen3.8-Flash-Next-IQ4_XS-00001-of-00003.gguf}"
DRAFT_MODEL="${DRAFT_MODEL:-$HOME/models/flashnext-mtp/mtp-Qwen3.8-Flash-Next-Q8_0.gguf}"
PORT="${PORT:-8835}"
CTX="${CTX:-262144}"

SPEC_N_MAX="${SPEC_N_MAX:-2}"      # 每步最多草稿几个 token
SPEC_P_MIN="${SPEC_P_MIN:-0.75}"   # 草稿置信度下限，低于此值不投

cd "$MODEL_DIR"

COMMON=(
  -m "$MAIN_MODEL"
  -ngl 99
  -c "$CTX"
  -np 1
  -t 8
  --cache-type-k q8_0
  --cache-type-v q8_0
  --host 0.0.0.0
  --port "$PORT"
  --jinja
)

case "$MODE" in
  base)
    exec "$LLAMA_BIN" "${COMMON[@]}"
    ;;
  mtp)
    exec "$LLAMA_BIN" "${COMMON[@]}" \
      --spec-type draft-mtp \
      --spec-draft-n-max "$SPEC_N_MAX" \
      --spec-draft-p-min "$SPEC_P_MIN" \
      --spec-draft-model "$DRAFT_MODEL"
    ;;
  dflash)
    exec "$LLAMA_BIN" "${COMMON[@]}" \
      --spec-type draft-dflash \
      --spec-draft-n-max "$SPEC_N_MAX" \
      --spec-draft-p-min "$SPEC_P_MIN" \
      --spec-draft-model "$DRAFT_MODEL"
    ;;
  *)
    echo "unknown mode: $MODE (expected base|mtp|dflash)" >&2
    exit 1
    ;;
esac
