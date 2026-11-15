#!/usr/bin/env python3
"""Run the experiment's small deterministic probe, not a full accuracy benchmark."""
import argparse
import ast
import json
import re
import time
import urllib.request
from pathlib import Path

CASES = [
    {
        "id": "arithmetic",
        "kind": "regex",
        "expected": r"(?<!\d)421(?!\d)",
        "prompt": "计算 (37×46−18)÷4。最终只输出数字。",
    },
    {
        "id": "algebra",
        "kind": "regex",
        "expected": r"(?<!\d)6(?!\d)",
        "prompt": "解方程 3x+7=25。最终只输出 x 的数值。",
    },
    {
        "id": "logic_color",
        "kind": "contains",
        "expected": ["蓝"],
        "prompt": "所有 Zorp 都是蓝色的。小明有一个 Zorp。它是什么颜色？最终只回答颜色。",
    },
    {
        "id": "sheep",
        "kind": "regex",
        "expected": r"(?<!\d)9(?!\d)",
        "prompt": "农场有17只羊，除了9只以外都跑了。还剩几只？最终只输出数字。",
    },
    {
        "id": "unit",
        "kind": "regex",
        "expected": r"(?<!\d)2500(?!\d)",
        "prompt": "2.5 千克等于多少克？最终只输出数值，不带单位。",
    },
    {
        "id": "json",
        "kind": "json",
        "expected": {"city": "Paris", "country": "France"},
        "prompt": "返回一个 JSON 对象，字段 city 为 Paris，country 为 France。不要输出其他文字。",
    },
    {
        "id": "leap_date",
        "kind": "contains",
        "expected": ["2月29日"],
        "prompt": "2024年2月28日后一天是几月几日？最终只按 M月D日 的格式回答。",
    },
    {
        "id": "exact_list",
        "kind": "exact",
        "expected": "apple, banana, cherry",
        "prompt": "严格按这个格式输出三个词：apple, banana, cherry。不要输出其他内容。",
    },
    {
        "id": "photosynthesis",
        "kind": "contains",
        "expected": ["水", "二氧化碳"],
        "prompt": "用一句中文回答：光合作用的两种主要无机原料是什么？只需写出两种原料。",
    },
    {
        "id": "syllogism",
        "kind": "regex",
        "expected": "^否[。.!！]?$",
        "prompt": "如果所有 A 都是 B，且没有 B 是 C，那么是否可能有 A 是 C？只回答 是 或 否。",
    },
    {
        "id": "code_palindrome",
        "kind": "python_syntax",
        "expected": "is_palindrome",
        "prompt": "写一个 Python 函数 is_palindrome(s)，忽略非字母数字字符并忽略大小写。只输出代码，不要解释。",
    },
    {
        "id": "code_binary_search",
        "kind": "python_syntax",
        "expected": "binary_search",
        "prompt": (
            "写一个 Python 函数 binary_search(sorted_items, target)，返回目标下标，" "不存在时返回 -1。只输出代码，不要解释。"
        ),
    },
]


def post(base_url, model, case, max_tokens, timeout):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": case["prompt"]}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_p": 1,
        "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "xhigh"},
        "separate_reasoning": True,
        "return_spec_tokens_details": True,
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.load(response)
    latency = time.perf_counter() - started
    choice = body["choices"][0]
    message = choice["message"]
    return {
        "content": message.get("content") or "",
        "reasoning_content": message.get("reasoning_content") or "",
        "finish_reason": choice.get("finish_reason"),
        "usage": body.get("usage"),
        "sglext": body.get("sglext"),
        "latency_s": latency,
    }


def strip_code_fence(text):
    stripped = text.strip()
    match = re.fullmatch(r"```(?:python)?\s*\n?(.*?)```", stripped, re.S | re.I)
    return match.group(1).strip() if match else stripped


def score(case, content):
    kind = case["kind"]
    text = content.strip()
    if kind == "regex":
        return bool(re.fullmatch(case["expected"], text))
    if kind == "contains":
        compact = re.sub(r"\s+", "", text)
        return all(re.sub(r"\s+", "", token) in compact for token in case["expected"])
    if kind == "exact":
        return re.sub(r"\s+", " ", text) == case["expected"]
    if kind == "json":
        try:
            value = json.loads(strip_code_fence(text))
        except (json.JSONDecodeError, TypeError):
            return False
        return value == case["expected"]
    if kind == "python_syntax":
        code = strip_code_fence(text)
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return False
        return any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == case["expected"]
            for node in tree.body
        )
    raise ValueError(kind)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:19410/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    result = {
        "model": args.model,
        "scope": "small deterministic probe; not GPQA/MMLU/LCB and not a global accuracy estimate",
        "automated_case_count": 10,
        "code_cases_note": (
            "code cases check syntax and required function name; "
            "use verify_code_functional.py for behavior"
        ),
        "cases": [],
    }
    for index, case in enumerate(CASES):
        response = post(args.base_url, args.model, case, args.max_tokens, args.timeout)
        passed = score(case, response["content"])
        record = {**case, "passed": passed, "response": response}
        result["cases"].append(record)
        print(
            f"{index + 1}/{len(CASES)} {case['id']} passed={passed} "
            f"finish={response['finish_reason']} latency={response['latency_s']:.2f}s",
            flush=True,
        )

    automated = [case for case in result["cases"] if case["kind"] != "python_syntax"]
    code = [case for case in result["cases"] if case["kind"] == "python_syntax"]
    result["summary"] = {
        "automated_passed": sum(case["passed"] for case in automated),
        "automated_total": len(automated),
        "automated_truncated": sum(
            case["response"]["finish_reason"] == "length" for case in automated
        ),
        "automated_empty": sum(not case["response"]["content"].strip() for case in automated),
        "code_syntax_passed": sum(case["passed"] for case in code),
        "code_total": len(code),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"saved {output}")


if __name__ == "__main__":
    main()
