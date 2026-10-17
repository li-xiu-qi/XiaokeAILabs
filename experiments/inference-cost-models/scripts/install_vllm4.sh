#!/bin/bash
source /home/ke/xinfer-env/bin/activate

echo "=== 安装 vLLM ==="
uv pip install --python /home/ke/vllm-env/bin/python vllm 2>&1 | tail -30
