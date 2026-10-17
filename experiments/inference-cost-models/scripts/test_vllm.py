import requests
import json
import time

url = "http://localhost:8000/v1/completions"
model = "/home/ke/models/Qwen3.8-27B-BF16"

print("=== 测试 1: Decode 吞吐（单次推理）===")
print("Prompt: '你好' | 生成: 128 tokens")
print()

payload = {
    "model": model,
    "prompt": "你好",
    "max_tokens": 128,
    "temperature": 0.7
}

start = time.time()
r = requests.post(url, json=payload, timeout=300)
elapsed = time.time() - start

result = r.json()
text = result['choices'][0]['text']
usage = result['usage']

print(f"生成文本: {text[:150]}...")
print(f"Prompt tokens: {usage['prompt_tokens']}")
print(f"Completion tokens: {usage['completion_tokens']}")
print(f"总耗时: {elapsed:.2f} 秒")
print(f"Decode TPS: {usage['completion_tokens']/elapsed:.2f} tok/s")
print()

print("=== 测试 2: Prefill 吞吐（批处理）===")
print("32 条 prompt，每条约 128 tokens，生成 1 token")
print()

payload2 = {
    "model": model,
    "prompt": ["你好世界。这是一个测试。"] * 32,
    "max_tokens": 1,
    "temperature": 0.7
}

start = time.time()
r2 = requests.post(url, json=payload2, timeout=300)
elapsed2 = time.time() - start

result2 = r2.json()
usage2 = result2['usage']

print(f"Prompt tokens 总数: {usage2['prompt_tokens']}")
print(f"Completion tokens: {usage2['completion_tokens']}")
print(f"总耗时: {elapsed2:.2f} 秒")
print(f"Prefill TPS: {usage2['prompt_tokens']/elapsed2:.2f} tok/s")
print()

print("=== 总结 ===")
print(f"Decode (bs=1, 128 tokens): {usage['completion_tokens']/elapsed:.2f} tok/s")
print(f"Prefill (bs=32, ~128 tokens): {usage2['prompt_tokens']/elapsed2:.2f} tok/s")
