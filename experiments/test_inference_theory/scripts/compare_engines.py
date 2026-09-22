import requests
import time
import sys

def test_engine(name, url, model, prompt_tokens=128, batch_size=1):
    """测试单个引擎的 decode 和 prefill 性能。"""
    print(f"\n{'='*60}")
    print(f"测试 {name} @ {url}")
    print(f"{'='*60}")

    # Decode 测试
    print(f"\n--- Decode (bs=1, {prompt_tokens} tokens) ---")
    payload = {
        "model": model,
        "prompt": "你好",
        "max_tokens": prompt_tokens,
        "temperature": 0.7
    }
    try:
        start = time.time()
        r = requests.post(f"{url}/v1/completions", json=payload, timeout=600)
        elapsed = time.time() - start
        result = r.json()
        usage = result.get('usage', {})
        completion_tokens = usage.get('completion_tokens', 0)
        decode_tps = completion_tokens / elapsed if elapsed > 0 else 0
        print(f"  Completion tokens: {completion_tokens}")
        print(f"  耗时: {elapsed:.2f}s")
        print(f"  Decode TPS: {decode_tps:.2f} tok/s")
    except Exception as e:
        print(f"  ERROR: {e}")
        decode_tps = 0

    # Prefill 测试
    print(f"\n--- Prefill (bs={batch_size}, ~128 tokens) ---")
    payload2 = {
        "model": model,
        "prompt": ["你好世界。这是一个测试。"] * batch_size,
        "max_tokens": 1,
        "temperature": 0.7
    }
    try:
        start = time.time()
        r2 = requests.post(f"{url}/v1/completions", json=payload2, timeout=600)
        elapsed2 = time.time() - start
        result2 = r2.json()
        usage2 = result2.get('usage', {})
        prompt_tokens_total = usage2.get('prompt_tokens', 0)
        prefill_tps = prompt_tokens_total / elapsed2 if elapsed2 > 0 else 0
        print(f"  Prompt tokens: {prompt_tokens_total}")
        print(f"  耗时: {elapsed2:.2f}s")
        print(f"  Prefill TPS: {prefill_tps:.2f} tok/s")
    except Exception as e:
        print(f"  ERROR: {e}")
        prefill_tps = 0

    return decode_tps, prefill_tps


def main():
    model = "/home/ke/models/Qwen3.8-27B-BF16"

    results = {}

    # 测试 vLLM（端口 8000）
    if "--vllm" in sys.argv or "--all" in sys.argv:
        try:
            requests.get("http://localhost:8000/health", timeout=5)
            dec, pre = test_engine("vLLM 0.28.0", "http://localhost:8000", model)
            results["vLLM"] = {"decode": dec, "prefill": pre}
        except:
            print("vLLM 未就绪，跳过")

    # 测试 SGLang（端口 8001）
    if "--sglang" in sys.argv or "--all" in sys.argv:
        try:
            requests.get("http://localhost:8001/health", timeout=5)
            dec, pre = test_engine("SGLang 0.5.19", "http://localhost:8001", model)
            results["SGLang"] = {"decode": dec, "prefill": pre}
        except:
            print("SGLang 未就绪，跳过")

    # 汇总
    if results:
        print(f"\n{'='*60}")
        print("汇总对比")
        print(f"{'='*60}")
        print(f"{'引擎':<20} {'Decode tok/s':>15} {'Prefill tok/s':>15}")
        print("-" * 52)
        for name, r in results.items():
            print(f"{name:<20} {r['decode']:>13.2f}  {r['prefill']:>13.2f}")

        # 与 llama.cpp 对比
        print(f"\n{'llama.cpp':<20} {4.30:>13.2f}  {25.80:>13.2f}")


if __name__ == "__main__":
    main()
