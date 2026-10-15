#!/usr/bin/env python3
"""投机解码性能测试：发统一请求，提取 prefill / 生成吞吐 / 草稿接受率。

只依赖标准库。默认走 /completion 原始补全端点，因为它让生成长度可控
（chat 端点下 thinking 模型会自主决定思维链长度，两次运行长度不一致，
平均吞吐就不可直接比）。

用法：
    python scripts/bench.py --port 8835 --n-predict 400
    python scripts/bench.py --port 8835 --mode chat --n-predict 400
"""

import argparse
import json
import urllib.request


DEFAULT_PROMPT = (
    "请用中文写一段约300字的介绍，主题是：统一内存架构对本地大模型推理的影响。"
    "要求分三点说明。"
)


def call(base_url, endpoint, payload, timeout):
    url = f"{base_url}/{endpoint}"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


# llama.cpp 的原始补全端点在 /completion，chat 模板端点在 /v1/chat/completions
ENDPOINTS = {"completion": "completion", "chat": "v1/chat/completions"}


def build_payload(mode, prompt, n_predict, temperature, ignore_eos=False):
    if mode == "completion":
        payload = {"prompt": prompt, "n_predict": n_predict, "temperature": temperature}
        # 固定生成长度：模型自然输出 EOS 会提前结束，导致平均吞吐不可比
        if ignore_eos:
            payload["ignore_eos"] = True
        return payload
    return {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": n_predict,
        "temperature": temperature,
    }


def extract(result, mode):
    timings = result.get("timings", {})
    usage = result.get("usage", {})
    out = {
        "prefill_tps": round(timings.get("prompt_per_second", 0.0), 2),
        "gen_tps": round(timings.get("predicted_per_second", 0.0), 2),
        "prefill_tokens": timings.get("prompt_n", 0),
        "gen_tokens": timings.get("predicted_n", 0) or usage.get("completion_tokens", 0),
    }
    # 投机解码专有：草稿生成数与接受数，接受率 = 接受 / 生成
    if "draft_n" in timings:
        draft_n = timings.get("draft_n", 0)
        accepted = timings.get("draft_n_accepted", 0)
        out["draft_n"] = draft_n
        out["draft_accepted"] = accepted
        out["accept_rate"] = round(accepted / draft_n, 4) if draft_n else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8835)
    ap.add_argument("--mode", choices=["completion", "chat"], default="completion")
    ap.add_argument("--n-predict", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--prompt-file", help="从文件读 prompt（覆盖 --prompt）")
    ap.add_argument("--ignore-eos", action="store_true", help="固定生成长度，忽略 EOS")
    ap.add_argument("--timeout", type=int, default=300)
    args = ap.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    prompt = args.prompt
    if args.prompt_file:
        with open(args.prompt_file, encoding="utf-8") as f:
            prompt = f.read().strip()
    payload = build_payload(
        args.mode, prompt, args.n_predict, args.temperature, args.ignore_eos
    )
    result = call(base_url, ENDPOINTS[args.mode], payload, args.timeout)
    metrics = extract(result, args.mode)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
