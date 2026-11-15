"""Request-level HTTP timing; does not measure prefill or decode phases."""
import argparse
import json
import time
import requests


def test_engine(url, model, prompt, max_tokens=128, timeout=600):
    payload = {"model": model, "prompt": prompt, "max_tokens": max_tokens,
               "temperature": 0}
    start = time.perf_counter()
    response = requests.post(f"{url.rstrip('/')}/v1/completions", json=payload, timeout=timeout)
    elapsed = time.perf_counter() - start
    response.raise_for_status()
    result = response.json()
    usage = result.get("usage") or {}
    return {"elapsed_seconds": elapsed, "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "completion_tokens_per_request_second": (
                usage["completion_tokens"] / elapsed
                if elapsed > 0 and usage.get("completion_tokens") is not None else None)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    print("Request-level timing includes queue, prefill, generation and network.")
    print(json.dumps(test_engine(args.url, args.model, args.prompt, args.max_tokens), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
