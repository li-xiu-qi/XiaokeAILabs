#!/bin/bash
source ~/xinfer-env/bin/activate

echo "=== 用 gguf 元数据工具读取 Q4_K_XL ==="
cd ~/llama.cpp
python3 convert_hf_to_gguf.py --verbose /home/ke/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf 2>&1 | grep -E "gguf|metadata|block_count|head_count|embedding_length|quantization|file size" | head -20

echo ""
echo "=== 直接从 config.json 读取架构参数 ==="
cat ~/models/Qwen3.8-27B-BF16/config.json
