#!/usr/bin/env bash
# 跨机流水线（PP=2）冒烟测试启动器。
#
# 两个节点各执行一次本脚本（role 不同），先起 stage1 再起 stage2。
# 环境变量：
#   PYTHON        Python 解释器（默认 python，需已安装带 CUDA 的 torch）
#   MASTER_ADDR   stage 1 节点地址，必填
#   MASTER_PORT   TCPStore 端口（默认 29500）
# 其余参数透传给 pipeline_stage.py。
#
# 示例：
#   # 节点 A
#   MASTER_ADDR=<stage1-ip> bash scripts/run_smoke.sh stage1
#   # 节点 B
#   MASTER_ADDR=<stage1-ip> bash scripts/run_smoke.sh stage2

set -euo pipefail

ROLE="${1:?用法: run_smoke.sh <stage1|stage2> [额外参数...]}"; shift || true
PYTHON="${PYTHON:-python}"
MASTER_ADDR="${MASTER_ADDR:?请设置 MASTER_ADDR 为 stage 1 的地址}"
MASTER_PORT="${MASTER_PORT:-29500}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec "$PYTHON" "$SCRIPT_DIR/pipeline_stage.py" \
  --role "$ROLE" \
  --master-addr "$MASTER_ADDR" \
  --master-port "$MASTER_PORT" \
  "$@"
