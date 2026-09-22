#!/bin/bash
echo "=== 测试 1: Decode 吞吐（单次推理）==="
echo "Prompt: '你好' | 生成: 128 tokens"

START=$(date +%s.%N)
RESULT=$(curl -s http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "/home/ke/models/Qwen3.8-27B-BF16", "prompt": "你好", "max_tokens": 128, "temperature": 0.7}')
END=$(date +%s.%N)
ELAPSED=$(echo "$END - $START" | bc)

echo "$RESULT" | /home/ke/vllm-env/bin/python -c "
import sys, json
r = json.load(sys.stdin)
text = r['choices'][0]['text']
usage = r['usage']
print(f'生成文本: {text[:100]}...')
print(f'Prompt tokens: {usage[\"prompt_tokens\"]}')
print(f'Completion tokens: {usage[\"completion_tokens\"]}')
print(f'总耗时: $ELAPSED 秒')
print(f'Decode TPS: {usage[\"completion_tokens\"]/$ELAPSED:.2f} tok/s')
"

echo ""
echo "=== 测试 2: Prefill 吞吐（批处理）==="
echo "32 条 prompt，每条约 128 tokens，生成 1 token"

START=$(date +%s.%N)
RESULT2=$(curl -s http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "/home/ke/models/Qwen3.8-27B-BF16", "prompt": ["你好世界。这是一个测试。"] * 32, "max_tokens": 1, "temperature": 0.7}')
END=$(date +%s.%N)
ELAPSED=$(echo "$END - $START" | bc)

echo "$RESULT2" | /home/ke/vllm-env/bin/python -c "
import sys, json
r = json.load(sys.stdin)
usage = r['usage']
print(f'Prompt tokens 总数: {usage[\"prompt_tokens\"]}')
print(f'Completion tokens: {usage[\"completion_tokens\"]}')
print(f'总耗时: $ELAPSED 秒')
print(f'Prefill TPS: {usage[\"prompt_tokens\"]/$ELAPSED:.2f} tok/s')
"

echo ""
echo "=== 测试 3: GPU 状态 ==="
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader

echo ""
echo "=== 测试完成 ==="
