#!/bin/bash
source /home/ke/xinfer-env/bin/activate

echo "=== 安装 ninja 到 vllm-env ==="
uv pip install --python /home/ke/vllm-env/bin/python ninja 2>&1 | tail -5

echo "=== 验证 ninja ==="
/home/ke/vllm-env/bin/python -c "import ninja; print('ninja:', ninja.__file__)"
which ninja || echo "ninja 不在 PATH（Python 包内的可用）"

echo "=== 重启 vLLM ==="
pgrep -f "vllm.entrypoints" | while read pid; do kill -9 $pid 2>/dev/null; done
sleep 3

cd /home/ke/vllm-env
setsid ./bin/python -m vllm.entrypoints.openai.api_server \
    --model /home/ke/models/Qwen3.8-27B-BF16 \
    --trust-remote-code \
    --dtype bfloat16 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.6 \
    --port 8000 \
    > /tmp/vllm_new2.log 2>&1 < /dev/null &

sleep 2
echo "vLLM 重启命令已发出"
disown
