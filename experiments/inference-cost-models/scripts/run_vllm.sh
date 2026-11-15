#!/bin/bash
source $HOME/xinfer-env/bin/activate

echo "=== 启动 vLLM（enforce-eager 模式）==="
export PATH="$HOME/vllm-env/bin:$PATH"

cd $HOME/vllm-env
setsid ./bin/python -m vllm.entrypoints.openai.api_server \
    --model $HOME/models/Qwen3.8-27B-BF16 \
    --trust-remote-code \
    --dtype bfloat16 \
    --max-model-len 1024 \
    --gpu-memory-utilization 0.5 \
    --enforce-eager \
    --port 8000 \
    > /tmp/vllm_run.log 2>&1 < /dev/null &

sleep 2
echo "vLLM 启动命令已发出"
disown
