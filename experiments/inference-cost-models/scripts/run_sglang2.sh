#!/bin/bash
source $HOME/xinfer-env/bin/activate

export PATH="$HOME/sglang-env/bin:$PATH"

echo "启动 SGLang（提高显存利用率）..."
cd $HOME/sglang-env
setsid ./bin/python -m sglang.launch_server \
    --model-path $HOME/models/Qwen3.8-27B-BF16 \
    --trust-remote-code \
    --dtype bfloat16 \
    --context-length 2048 \
    --mem-fraction-static 0.7 \
    --max-running-requests 32 \
    --port 8001 \
    > /tmp/sglang2.log 2>&1 < /dev/null &

sleep 2
echo "SGLang 已启动（mem-fraction-static=0.7, max-running-requests=32）"
disown
