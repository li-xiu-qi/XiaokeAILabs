#!/usr/bin/env python3
"""闭环节点：1.0 识别一个块 → Jev 给三个候选打分排序。

三个候选分别是：人工确认的原文、1.0 的识别结果、人为破坏的原文（删掉末行）。
期望的排序是 原文 > 1.0 输出 > 破坏版；若 1.0 输出与原文一字不差，两者应接近。

    python jev_loop_block.py \
        --block $MINICPM_OCR_HOME/repo/scripts/minicpm-ocr-jev/examples/01_text_putnam/block.png \
        --truth $MINICPM_OCR_HOME/runs/jevloop_block/truth.txt
"""

from __future__ import annotations

import os
import argparse
import base64
import json
import time
import urllib.request
from pathlib import Path

import requests

OCR_URL = "http://127.0.0.1:18499/v1/chat/completions"
OCR_MODEL = os.environ.get("OCR_MODEL_DIR", str(Path.home() / "minicpm-ocr/models/MiniCPM-OCR-1.0"))
RM_URL = "http://127.0.0.1:18501"


def ocr_block(image: Path, prompt: str, max_tokens: int) -> tuple[str, float]:
    b64 = base64.b64encode(image.read_bytes()).decode()
    payload = {
        "model": OCR_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": prompt},
            ],
        }],
        "max_tokens": max_tokens,
        "stream": False,
        "temperature": 0.01,
        "top_p": 0.001,
        "generation_config": json.dumps(
            {"top_k": 1, "repetition_penalty": 1.0, "no_repeat_ngram_size": 100}
        ),
    }
    req = urllib.request.Request(
        OCR_URL, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as resp:
        body = json.loads(resp.read())
    return body["choices"][0]["message"]["content"], time.time() - t0


def corrupt(text: str) -> str:
    """删掉最后一行非空内容，制造一个「看起来像但内容缺一块」的候选。"""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines[:-1]) if len(lines) > 1 else text + " [错]"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type=Path, required=True)
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--category", default="text")
    ap.add_argument("--prompt", default="Text Recognition:")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--out-dir", type=Path, default=Path(os.environ.get("JEV_RUN_DIR", str(Path.home() / "minicpm-ocr/runs/jevloop_block"))))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    truth = args.truth.read_text(encoding="utf-8")

    ocr_text, ocr_sec = ocr_block(args.block, args.prompt, args.max_tokens)
    (args.out_dir / "ocr.txt").write_text(ocr_text, encoding="utf-8")
    print(f"1.0 识别完成 {ocr_sec:.1f}s，{len(ocr_text)} 字符")
    print("--- 1.0 输出 ---")
    print(ocr_text)
    print("--- 与原文逐字相同:", ocr_text.strip() == truth.strip(), "---")

    candidates = [("原文", truth), ("1.0输出", ocr_text), ("破坏版", corrupt(truth))]
    payload = {
        "input_form": "binary",
        "image": base64.b64encode(args.block.read_bytes()).decode(),
        "candidates": [c[1] for c in candidates],
        "category": args.category,
    }
    t0 = time.time()
    resp = requests.post(f"{RM_URL}/v1/rm/score_best", json=payload, timeout=900)
    resp.raise_for_status()
    body = resp.json()
    total = time.time() - t0

    print(f"\n{'候选':<10}{'prob_yes':>10}{'verdict':>12}")
    print("-" * 32)
    for (name, _), score, verdict in zip(candidates, body["scores"], body["verdicts"]):
        print(f"{name:<10}{score:>10.5f}{verdict:>12}")
    print(f"\nbest = {candidates[body['best_index']][0]}   "
          f"（Jev 端到端 {total:.1f}s，stage={body.get('stage_sec')}）")
    (args.out_dir / "jev_scores.json").write_text(
        json.dumps({"candidates": [c[0] for c in candidates], **body}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
