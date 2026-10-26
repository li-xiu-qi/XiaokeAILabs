#!/usr/bin/env python3
"""Standard OpenAI text and image smoke for the NInfer GB10 server."""
import argparse
import base64
import json
import urllib.request
from pathlib import Path


def post(base_url, payload):
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


def payload(model, messages, max_tokens=64):
    return {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:19410/v1")
    parser.add_argument("--model", default="qwen3.8-27b-nvfp4-dflash2")
    parser.add_argument(
        "--image",
        default=str(Path(__file__).resolve().parents[1] / "images" / "ninfer-vision-smoke.png"),
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    text_response = post(
        args.base_url,
        payload(
            args.model,
            [
                {
                    "role": "user",
                    "content": "先用一句中文解释投机解码，再单独一行给出 17×23 的数值答案。",
                }
            ],
            args.max_tokens,
        ),
    )
    image_data = base64.b64encode(Path(args.image).read_bytes()).decode()
    image_response = post(
        args.base_url,
        payload(
            args.model,
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "List the two shapes, their colors, and the text "
                                "shown in this image. Answer in English."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64," + image_data},
                        },
                    ],
                }
            ],
            args.max_tokens,
        ),
    )
    text_content = text_response["choices"][0]["message"].get("content") or ""
    image_content = image_response["choices"][0]["message"].get("content") or ""
    checks = {
        "text_has_391": "391" in text_content,
        "text_finished": text_response["choices"][0].get("finish_reason") == "stop",
        "image_finished": image_response["choices"][0].get("finish_reason") == "stop",
        "image_has_red_circle": (
            "red" in image_content.lower() and "circle" in image_content.lower()
        ),
        "image_has_blue_square": (
            "blue" in image_content.lower() and "square" in image_content.lower()
        ),
        "image_has_text": "dgx" in image_content.lower() and "19410" in image_content.lower(),
    }
    result = {
        "model": args.model,
        "text_content": text_content,
        "image_content": image_content,
        "text_response": text_response,
        "image_response": image_response,
        "checks": checks,
        "passed": all(checks.values()),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("TEXT:\n" + text_content)
    print("IMAGE:\n" + image_content)
    print(json.dumps({"passed": result["passed"], "checks": checks}, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
