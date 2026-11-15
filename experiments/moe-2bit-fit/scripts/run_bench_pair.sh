#!/usr/bin/env bash
# 在 DGX Spark（GB10）上跑完非投机与 MTP K=2 两组 bench。
# 前提：已按 README「复现」节装好环境并下载模型，在插件仓根目录执行。
#
#   bash "$XL_REPO/experiments/moe-2bit-fit/scripts/run_bench_pair.sh"          # 两组
#   bash "$XL_REPO/experiments/moe-2bit-fit/scripts/run_bench_pair.sh" nospec   # 非投机
#   bash "$XL_REPO/experiments/moe-2bit-fit/scripts/run_bench_pair.sh" mtp2     # MTP K=2
# XL_REPO 指向 XiaokeAILabs 克隆目录；当前目录保持为插件仓根目录。
#
# 两个环境变量沿用历史实验配置，原因见 README「已知限制」与对应实测报告：
#   VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=0  解 MTP drafter capture 的 sm_121 跨流崩溃
#   GPU_UTIL=0.85                                 历史运行配额；并不保证其他环境不会 OOM
set -euo pipefail
if [[ ! -f scripts/bench.py || ! -f .venv-vllm/bin/activate ]]; then
  echo '请先进入包含 scripts/bench.py 和 .venv-vllm 的插件仓根目录。' >&2
  exit 1
fi

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
