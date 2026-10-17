#!/bin/bash
source /home/ke/xinfer-env/bin/activate

export PATH="/home/ke/sglang-env/bin:$PATH"

echo "启动 SGLang 服务器..."
cd /home/ke/sglang-env
setsid ./bin/python -m sglang.launch_server \
    --model-path /home/ke/models/Qwen3.8-27B-BF16 \
    --trust-remote-code \
    --dtype bfloat16 \
    --context-length 2048 \
    --mem-fraction-static 0.5 \
    --port 8001 \
    > /tmp/sglang.log 2>&1 < /dev/null &

sleep 2
echo "SGLang 启动命令已发出"
disown
