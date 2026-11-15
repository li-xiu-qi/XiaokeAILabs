#!/bin/bash
source $HOME/xinfer-env/bin/activate

echo "=== 安装 vLLM ==="
uv pip install --python $HOME/vllm-env/bin/python vllm 2>&1 | tail -30
