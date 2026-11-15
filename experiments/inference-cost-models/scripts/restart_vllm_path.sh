#!/bin/bash
source $HOME/xinfer-env/bin/activate

# 杀掉旧进程
pgrep -f "vllm.entrypoints" | while read pid; do kill -9 $pid 2>/dev/null; done
sleep 3

# 找到 ninja 可执行文件路径并加到 PATH
NINJA_BIN=$(find $HOME/vllm-env -name "ninja" -type f 2>/dev/null | head -1)
if [ -n "$NINJA_BIN" ]; then
    export PATH="$(dirname $NINJA_BIN):$PATH"
    echo "ninja 路径: $NINJA_BIN"
fi

# 也把 vllm-env/bin 加到 PATH
export PATH="$HOME/vllm-env/bin:$PATH"

cd $HOME/vllm-env
setsid ./bin/python -m vllm.entrypoints.openai.api_server \
    --model $HOME/models/Qwen3.8-27B-BF16 \
    --trust-remote-code \
    --dtype bfloat16 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.6 \
    --port 8000 \
    > /tmp/vllm_final.log 2>&1 < /dev/null &

sleep 2
echo "vLLM 已启动"
disown
