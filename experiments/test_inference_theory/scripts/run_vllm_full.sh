#!/bin/bash
source /home/ke/xinfer-env/bin/activate

export PATH="/home/ke/vllm-env/bin:$PATH"

echo "启动 vLLM（完整 torch.compile 模式）..."
cd /home/ke/vllm-env
setsid ./bin/python -m vllm.entrypoints.openai.api_server \
    --model /home/ke/models/Qwen3.8-27B-BF16 \
    --trust-remote-code \
    --dtype bfloat16 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.5 \
    --port 8000 \
    > /tmp/vllm_full.log 2>&1 < /dev/null &

sleep 2
echo "vLLM 完整模式启动命令已发出"
disown
