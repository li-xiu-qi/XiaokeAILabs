#!/bin/bash
source /home/ke/xinfer-env/bin/activate

# 只杀掉之前测试的 vllm，不杀当前 shell
pgrep -f "vllm.entrypoints" | while read pid; do
    kill -9 $pid 2>/dev/null
done
sleep 2

echo "启动 vLLM..."
cd /home/ke/vllm-env
setsid ./bin/python -m vllm.entrypoints.openai.api_server \
    --model /home/ke/models/Qwen3.8-27B-BF16 \
    --trust-remote-code \
    --dtype bfloat16 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.6 \
    --port 8000 \
    > /tmp/vllm_new.log 2>&1 < /dev/null &

sleep 2
echo "vLLM 启动命令已发出，PID: $!"
disown
