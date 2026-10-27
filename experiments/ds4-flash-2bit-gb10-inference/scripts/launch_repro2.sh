#!/usr/bin/env bash
# min_repro launcher with launch-blocking to catch the hanging kernel
cd "${PLUGIN_REPO:-$HOME/deepseek-v4-flash-2bit-gb10}"
pkill -9 -f min_repro.py 2>/dev/null
sleep 2
source .venv-vllm/bin/activate
export CUDA_LAUNCH_BLOCKING=1
export TRITON_PRINT_AUTOTUNING=1
export VLLM_LOGGING_LEVEL=DEBUG
nohup timeout 1200 python scripts/min_repro.py > logs/min_repro2.log 2>&1 &
echo "launched pid: $!"
