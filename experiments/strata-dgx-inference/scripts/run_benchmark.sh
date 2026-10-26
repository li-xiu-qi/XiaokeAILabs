#!/bin/bash
# Run Strata benchmark on DGX Spark
# Prerequisite: Strata server running on 127.0.0.1:8890
set -euo pipefail

OUTDIR="${STRATA_WORK_ROOT:-$HOME/strata-dgx-inference}/artifacts/benchmark"
mkdir -p "$OUTDIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS="$OUTDIR/strata-results-${TIMESTAMP}.jsonl"

run_test() {
    local name="$1"
    local prompt="$2"
    local max_tokens="$3"
    local result
    result=$(curl -s -X POST http://127.0.0.1:8890/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"qwen\",\"messages\":[{\"role\":\"user\",\"content\":\"$prompt\"}],\"max_tokens\":$max_tokens,\"temperature\":0.7}" \
        --max-time 180)
    echo "{\"test\":\"$name\",\"timestamp\":\"$(date -Is)\",\"result\":$result}" >> "$RESULTS"
    echo "$name: $(echo "$result" | python3 -c "import sys,json; d=json.load(sys.stdin); t=d.get(\"timings\",{}); print(f\"prompt={t.get(\"prompt_per_second\",\"N/A\")} tok/s, gen={t.get(\"predicted_per_second\",\"N/A\")} tok/s, draft_acc={t.get(\"draft_n_accepted\",0)}/{t.get(\"draft_n\",0)}\")" 2>/dev/null || echo "parse error")"
}

echo "Strata benchmark started at $(date -Is)"

run_test "warmup" "你好" 20
run_test "short_short" "1+1等于多少？" 30
run_test "short_short_2" "中国的首都是哪里？" 30
run_test "short_long" "请写一首关于春天的诗。" 200
run_test "medium_medium" "请解释一下什么是机器学习，并给出三个应用场景。" 100
LONG_PROMPT="请总结：人工智能是计算机科学的重要分支，旨在让机器模拟人类感知、学习、推理和决策能力。它通过算法和数据完成图像识别、自然语言处理、智能推荐等任务，广泛应用于医疗、交通、金融和教育等领域。人工智能正改变生产生活方式，也带来伦理与就业挑战。随着深度学习的发展，AI模型规模不断扩大，对计算资源的需求也日益增长。"
run_test "long_short" "$LONG_PROMPT" 50
run_test "repeat_1" "你好，请用一句话介绍你自己。" 50
run_test "repeat_2" "你好，请用一句话介绍你自己。" 50
run_test "repeat_3" "你好，请用一句话介绍你自己。" 50

echo "Strata benchmark completed at $(date -Is)"
echo "Results saved to $RESULTS"

# Semantic smoke tests
echo ""
echo "=== Semantic smoke: arithmetic ==="
curl -s -X POST http://127.0.0.1:8890/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"qwen","messages":[{"role":"user","content":"17乘以23等于多少？请直接给出数字答案。"}],"max_tokens":100,"temperature":0}' \
    --max-time 120
echo ""
echo "=== Semantic smoke: general ==="
curl -s -X POST http://127.0.0.1:8890/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"qwen","messages":[{"role":"user","content":"中国的首都是哪里？请直接回答城市名。"}],"max_tokens":100,"temperature":0}' \
    --max-time 120
echo ""
