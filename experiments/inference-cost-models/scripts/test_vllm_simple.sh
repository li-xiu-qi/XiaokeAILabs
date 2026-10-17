#!/bin/bash
source /home/ke/xinfer-env/bin/activate

echo "=== Qwen3.8-27B BF16 vLLM 部署测试 ==="

# 先清理
pkill -9 -f vllm 2>/dev/null
sleep 3

# 启动 vLLM
echo "启动 vLLM 服务器..."
cd /home/ke/vllm-env
nohup ./bin/python -m vllm.entrypoints.openai.api_server \
    --model /home/ke/models/Qwen3.8-27B-BF16 \
    --trust-remote-code \
    --dtype bfloat16 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.6 \
    --port 8000 \
    > /tmp/vllm_bf16.log 2>&1 &

VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

# 等待启动
echo "等待服务器启动..."
for i in $(seq 1 60); do
    sleep 5
    if curl -s http://localhost:8000/health 2>/dev/null | grep -q "ok"; then
        echo "✅ vLLM 服务器已就绪 (${i}0 秒)"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "❌ 启动超时"
        tail -50 /tmp/vllm_bf16.log
        exit 1
    fi
done

echo ""
echo "=== 测试 1: Decode 吞吐 ==="
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
print(f'文本: {text[:80]}...')
print(f'Prompt: {usage[\"prompt_tokens\"]} | Completion: {usage[\"completion_tokens\"]}')
print(f'耗时: $ELAPSED 秒')
print(f'Decode TPS: {usage[\"completion_tokens\"]/$ELAPSED:.2f} tok/s')
"

echo ""
echo "=== 测试 2: Prefill 吞吐 ==="
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
print(f'Completion: {usage[\"completion_tokens\"]}')
print(f'耗时: $ELAPSED 秒')
print(f'Prefill TPS: {usage[\"prompt_tokens\"]/$ELAPSED:.2f} tok/s')
"

echo ""
echo "=== GPU 状态 ==="
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader

echo ""
echo "=== 完成 ==="
echo "vLLM PID: $VLLM_PID"
