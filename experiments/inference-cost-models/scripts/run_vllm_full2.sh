#!/bin/bash
source /home/ke/xinfer-env/bin/activate

export PATH="/home/ke/vllm-env/bin:$PATH"

echo "启动 vLLM（完整编译 + 限制并发）..."
cd /home/ke/vllm-env
setsid ./bin/python -m vllm.entrypoints.openai.api_server \
    --model /home/ke/models/Qwen3.8-27B-BF16 \
    --trust-remote-code \
    --dtype bfloat16 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.5 \
    --max-num-seqs 64 \
    --port 8000 \
    > /tmp/vllm_full2.log 2>&1 < /dev/null &

sleep 2
echo "vLLM 已启动（max-num-seqs=64）"
disown
