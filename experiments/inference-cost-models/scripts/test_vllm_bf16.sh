#!/bin/bash
source /home/ke/xinfer-env/bin/activate

echo "=== Qwen3.8-27B BF16 vLLM 部署测试（降低显存利用率）==="

# 先清理之前的进程
pkill -9 -f vllm 2>/dev/null
sleep 2

# 启动 vLLM 服务器（降低显存利用率到 0.6）
echo "启动 vLLM 服务器（gpu-memory-utilization=0.6）..."
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
echo "vLLM 进程 PID: $VLLM_PID"

# 等待服务器启动（最多等 5 分钟）
echo "等待服务器启动..."
for i in $(seq 1 60); do
    sleep 5
    if curl -s http://localhost:8000/health 2>/dev/null | grep -q "ok"; then
        echo "✅ vLLM 服务器已就绪 (${i}0 秒)"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "❌ 服务器启动超时，查看错误:"
        tail -80 /tmp/vllm_bf16.log
        exit 1
    fi
done

echo ""
echo "=== 测试 1: 单次推理（decode 吞吐）==="
echo "Prompt: '你好' | 生成: 128 tokens"
START_TIME=$(date +%s.%N)
RESULT=$(curl -s http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "/home/ke/models/Qwen3.8-27B-BF16",
        "prompt": "你好",
        "max_tokens": 128,
        "temperature": 0.7
    }')
END_TIME=$(date +%s.%N)
ELAPSED=$(echo "$END_TIME - $START_TIME" | bc)

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
echo "=== 测试 2: 批处理（prefill 吞吐）==="
echo "32 条 prompt，每条约 128 tokens，生成 1 token"
START_TIME=$(date +%s.%N)
RESULT2=$(curl -s http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "/home/ke/models/Qwen3.8-27B-BF16",
        "prompt": ["你好世界。这是一个测试。"] * 32,
        "max_tokens": 1,
        "temperature": 0.7
    }')
END_TIME=$(date +%s.%N)
ELAPSED=$(echo "$END_TIME - $START_TIME" | bc)

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
echo "vLLM 服务器仍在运行 (PID: $VLLM_PID)"
echo "停止: kill $VLLM_PID"
