import requests
import time

url = "http://localhost:8001/v1/completions"
model = "/home/ke/models/Qwen3.8-27B-BF16"

print("=== SGLang 测试 ===")
print()

# Decode
print("--- Decode (bs=1, 128 tokens) ---")
payload = {"model": model, "prompt": "你好", "max_tokens": 128, "temperature": 0.7}
start = time.time()
r = requests.post(url, json=payload, timeout=600)
elapsed = time.time() - start
result = r.json()
usage = result['usage']
print(f"文本: {result['choices'][0]['text'][:100]}...")
print(f"Completion: {usage['completion_tokens']}")
print(f"耗时: {elapsed:.2f}s")
print(f"Decode TPS: {usage['completion_tokens']/elapsed:.2f} tok/s")
print()

# Prefill
print("--- Prefill (bs=32, ~128 tokens) ---")
payload2 = {"model": model, "prompt": ["你好世界。这是一个测试。"] * 32, "max_tokens": 1, "temperature": 0.7}
start = time.time()
r2 = requests.post(url, json=payload2, timeout=600)
elapsed2 = time.time() - start
result2 = r2.json()
usage2 = result2['usage']
print(f"Prompt tokens: {usage2['prompt_tokens']}")
print(f"Completion: {usage2['completion_tokens']}")
print(f"耗时: {elapsed2:.2f}s")
print(f"Prefill TPS: {usage2['prompt_tokens']/elapsed2:.2f} tok/s")
print()

print("=== 总结 ===")
print(f"Decode: {usage['completion_tokens']/elapsed:.2f} tok/s")
print(f"Prefill: {usage2['prompt_tokens']/elapsed2:.2f} tok/s")
