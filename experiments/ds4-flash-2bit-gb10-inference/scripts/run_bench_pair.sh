#!/usr/bin/env bash
# 在 DGX Spark（GB10）上跑完非投机与 MTP K=2 两组 bench。
# 前提：已按 README「复现」节装好环境并下载模型，在插件仓根目录执行。
#
#   bash run_bench_pair.sh                 # 两组都跑（默认）
#   bash run_bench_pair.sh nospec          # 只跑非投机
#   bash run_bench_pair.sh mtp2            # 只跑 MTP K=2
#
# 两个环境变量是必需的，原因见 README「已知限制」与 reports/01：
#   VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=0  解 MTP drafter capture 的 sm_121 跨流崩溃
#   GPU_UTIL=0.85                                 121 GiB 统一内存下避开 OOM
set -euo pipefail
cd "$(dirname "$0")/.."

export VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=0
export GPU_UTIL="${GPU_UTIL:-0.85}"
source .venv-vllm/bin/activate

run_group() {
  local name="$1"; shift
  echo "=== group: $name ==="
  python scripts/bench.py --max-model-len "${MAX_MODEL_LEN:-2048}" "$@"
}

case "${1:-both}" in
  nospec) run_group nospec --spec-mtp 0 --no-frspec ;;
  mtp2)   run_group mtp2 --spec-mtp 2 ;;
  *)      run_group nospec --spec-mtp 0 --no-frspec
          run_group mtp2 --spec-mtp 2 ;;
esac
echo "done"
