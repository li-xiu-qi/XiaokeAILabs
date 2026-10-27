#!/usr/bin/env python3
"""往 Jev 打分服务投一条请求，打印每个候选的 prob_yes。

两种用法：
    # 官方样例：候选从 case.json 取，同时给出「自带参考渲染图」与「本机重渲染」两组分数
    python jev_probe.py --case $MINICPM_OCR_HOME/repo/scripts/minicpm-ocr-jev/examples/01_text_putnam

    # 自由对比：一张原图 + 若干候选文本文件
    python jev_probe.py --image page.jpg --text a.md --text b.md --category text
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import requests


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def post(url: str, image: Path, candidates: list[str], category: str,
         render_images: list[str] | None = None) -> dict:
    payload = {
        "input_form": "binary",
        "image": b64(image),
        "candidates": candidates,
        "category": category,
    }
    if render_images:
        payload["render_images"] = render_images
    resp = requests.post(f"{url}/v1/rm/score_best", json=payload, timeout=600)
    resp.raise_for_status()
    return resp.json()


def show(body: dict, names: list[str], observed: dict[str, float] | None = None) -> None:
    print(f"{'#':<3}{'candidate':<26}{'prob_yes':>9}{'verdict':>16}"
          + (f"{'官方值':>9}{'偏差':>8}" if observed else ""))
    print("-" * (54 + (17 if observed else 0)))
    for i, (score, verdict) in enumerate(zip(body["scores"], body["verdicts"])):
        name = names[i]
        line = f"{i:<3}{name:<26}{score:>9.5f}{verdict:>16}"
        if observed and name in observed:
            line += f"{observed[name]:>9.5f}{score - observed[name]:>+8.4f}"
        print(line)
    print(f"best_index = {body['best_index']}  ({names[body['best_index']]})")
    print(f"stage_sec = {body.get('stage_sec')}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:18501")
    ap.add_argument("--case", type=Path)
    ap.add_argument("--image", type=Path)
    ap.add_argument("--text", type=Path, action="append", default=[])
    ap.add_argument("--category", default="text")
    ap.add_argument("--extra-text", type=Path, action="append", default=[],
                    help="额外候选（如 1.0 的解析结果），--case 模式下追加")
    args = ap.parse_args()

    if args.case:
        case = json.loads((args.case / "case.json").read_text(encoding="utf-8"))
        image = args.case / case["block_image"]
        names = [f"{c.get('role')}" for c in case["candidates"]]
        texts = [c["text"] for c in case["candidates"]]
        observed = case.get("observed_prob_yes") or {}
        render_of = case.get("reference_render_of")

        print("=== A. 官方样例自带渲染图（只评它对应的那条候选，与官方同图）===")
        ref = args.case / case["reference_render"]
        ref_idx = next((i for i, n in enumerate(names) if n == render_of), 0)
        provided = [None] * len(texts)
        provided[ref_idx] = b64(ref)
        show(post(args.url, image, [texts[ref_idx]], case["category"], [provided[ref_idx]]),
             [names[ref_idx]], observed)

        print("\n=== B. 本机重新渲染（检验渲染器保真度）===")
        show(post(args.url, image, texts, case["category"]), names, observed)

        for extra in args.extra_text:
            texts.append(extra.read_text(encoding="utf-8"))
            names.append(extra.stem)
        if args.extra_text:
            print("\n=== C. 追加候选后重排 ===")
            show(post(args.url, image, texts, case["category"]), names)
        return 0

    if not args.image or not args.text:
        ap.error("需要 --case，或 --image 加至少一个 --text")
    texts = [p.read_text(encoding="utf-8") for p in args.text]
    show(post(args.url, args.image, texts, args.category), [p.stem for p in args.text])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
