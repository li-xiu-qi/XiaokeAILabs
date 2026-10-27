#!/usr/bin/env python3
"""MiniCPM-OCR-Jev 打分器（transformers 后端，不依赖 vLLM）。

官方只给了 vLLM 的 `runner="pooling"` + `convert="classify"` 路径；本机 aarch64 上
vLLM 装不上，所以这里用 transformers 的 Qwen3_5 主干手算两分类 logits：

    hidden = Qwen3_5Model(...).last_hidden_state[:, -1, :]     # 最后一个 token
    logits = hidden @ score.weight.T                          # [2, 2048] 头
    prob_yes = softmax(logits)[1]                             # 1 = 一致

prompt 模板与追加的 assistant 空轮逐字节复刻官方 `infer_vllm.py`，导入时用 md5
断言校验。模板被改动过，分数就不能和公布值比较。

用法：
    # 官方自带样例（只打有参考渲染图的那条候选）
    python jev_score.py --examples $MINICPM_OCR_HOME/repo/scripts/minicpm-ocr-jev/examples

    # 单条：原图 + 渲染图 + OCR 文本
    python jev_score.py --original block.png --render render.png --text-file ocr.md
"""

from __future__ import annotations

import os
import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

# ---------------------------------------------------------------------------
# 与官方 infer_vllm.py 逐字节一致的 prompt
# ---------------------------------------------------------------------------

IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"

INSTRUCTION = (
    "第一张图是原图，第二张图是上述 OCR 文本渲染得到的图片。\n"
    "请判断该 OCR 文本及其对应渲染图是否与原图内容一致。\n"
    "这是一个二分类任务：一致标为 1，不一致标为 0。"
)

#: 训练框架给序列分类器追加的空 assistant 轮。value head 读的就是它的最后一个 token，
#: 把它截掉会去池化一个训练时从未池化过的位置，分数看着正常其实已经偏了。
ASSISTANT_SUFFIX = (
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
    "<think>\n"
    "\n"
    "</think>\n"
    "\n"
)

PROMPT_TEMPLATE_MD5 = "9aab68e8ba8ab4da17ba9752bddb199c"

#: 每条 prompt 两张图：原图在前，渲染图在后。
IMAGES_PER_PROMPT = 2

#: Qwen3.5-VL 不接受极端长宽比；文档裁切块常是又宽又矮，超限按白底 padding 而不是缩放
#: （缩放会改变字形比例，padding 不会）。
MAX_ASPECT_RATIO = 199.0


def build_prompt(text: str) -> str:
    return (
        "<|im_start|>user\n"
        f"{IMAGE_PLACEHOLDER}\n"
        f"OCR文本：\n{text}\n"
        f"{IMAGE_PLACEHOLDER}\n"
        f"{INSTRUCTION}"
        f"{ASSISTANT_SUFFIX}"
    )


def _assert_prompt_unchanged() -> str:
    got = hashlib.md5(build_prompt("{text}").encode("utf-8")).hexdigest()
    if got != PROMPT_TEMPLATE_MD5:
        raise RuntimeError(
            "prompt 模板与官方不一致，分数不可比。\n"
            f"  期望 md5 {PROMPT_TEMPLATE_MD5}\n  实际 md5 {got}"
        )
    return got


def load_image(path: str | Path, max_ratio: float = MAX_ASPECT_RATIO) -> Image.Image:
    img = Image.open(path).convert("RGB")
    width, height = img.size
    longest, shortest = max(width, height), max(1, min(width, height))
    if longest / shortest <= max_ratio:
        return img
    if width >= height:
        new_w, new_h = width, max(height, int(width / max_ratio + 0.999999))
    else:
        new_h, new_w = height, max(width, int(height / max_ratio + 0.999999))
    canvas = Image.new("RGB", (new_w, new_h), "white")
    canvas.paste(img, ((new_w - width) // 2, (new_h - height) // 2))
    return canvas


# ---------------------------------------------------------------------------
# 打分器
# ---------------------------------------------------------------------------

@dataclass
class Scored:
    case_id: str
    role: str
    label: str | None
    observed: float | None
    prob_yes: float
    seconds: float


class JevScorer:
    def __init__(self, model_dir: str | Path, device: str = "cuda", dtype=torch.bfloat16):
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

        self.device = torch.device(device)
        self.dtype = dtype
        self.processor = AutoProcessor.from_pretrained(str(model_dir))
        t0 = time.time()
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            str(model_dir), dtype=dtype, device_map=device, attn_implementation="sdpa"
        )
        self.model.eval()
        # 权重里没有 lm_head 之外的分类头参数位置，score.weight 是独立键，单独读。
        from safetensors import safe_open

        with safe_open(str(Path(model_dir) / "model.safetensors"), framework="pt") as f:
            head = f.get_tensor("score.weight")
        if head.shape[0] != 2:
            raise RuntimeError(f"score.weight 不是两分类头：{tuple(head.shape)}")
        self.head = head.to(self.device, torch.float32)
        self.load_seconds = time.time() - t0

    def _forward_one(self, original: str | Path, render: str | Path, text: str) -> tuple[float, float]:
        prompt = build_prompt(text)
        images = [(load_image(original)), load_image(render)]
        inputs = self.processor(text=[prompt], images=[images], return_tensors="pt")
        inputs = {
            k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in inputs.items()
        }
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
        t0 = time.time()
        with torch.no_grad():
            out = self.model.model(**inputs)
        hidden = out.last_hidden_state[:, -1, :].to(torch.float32)
        logits = hidden @ self.head.T
        prob = torch.softmax(logits, dim=-1)[0, 1].item()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        return prob, time.time() - t0

# ---------------------------------------------------------------------------
# 样例与 CLI
# ---------------------------------------------------------------------------

def run_examples(scorer: JevScorer, examples_dir: Path) -> list[Scored]:
    results: list[Scored] = []
    observed_all: dict[str, float] = {}
    for case_dir in sorted(p for p in examples_dir.iterdir() if p.is_dir()):
        case_json = case_dir / "case.json"
        if not case_json.is_file():
            continue
        case = json.loads(case_json.read_text(encoding="utf-8"))
        original = case_dir / case["block_image"]
        observed_map = case.get("observed_prob_yes") or {}
        render_of = case.get("reference_render_of")
        render_path = case_dir / case["reference_render"] if case.get("reference_render") else None
        for cand in case["candidates"]:
            role = cand.get("role", "")
            # 参考渲染图只对应一条候选，其余候选需要渲染器另出图（本轮先不做）。
            if render_path is None or role != render_of:
                continue
            key = f"{case['case_id']}#{role}"
            observed_all[key] = observed_map.get(role)
            prob, secs = scorer._forward_one(original, render_path, cand["text"])
            results.append(Scored(
                case_id=case["case_id"], role=role, label=cand.get("label"),
                observed=observed_map.get(role), prob_yes=prob, seconds=secs,
            ))
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description="MiniCPM-OCR-Jev 打分（transformers 后端）")
    ap.add_argument("--model", default=os.environ.get("JEV_MODEL_DIR", str(Path.home() / "minicpm-ocr/models/MiniCPM-OCR-Jev")))
    ap.add_argument("--examples", type=Path,
                    default=Path(os.environ.get("JEV_EXAMPLES", str(Path.home() / "minicpm-ocr/repo/scripts/minicpm-ocr-jev/examples"))))
    ap.add_argument("--original", type=Path)
    ap.add_argument("--render", type=Path)
    ap.add_argument("--text", type=str)
    ap.add_argument("--text-file", type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    md5 = _assert_prompt_unchanged()
    print(f"prompt 模板 md5 校验通过: {md5}")

    scorer = JevScorer(args.model, device=args.device)
    print(f"模型加载完成 {scorer.load_seconds:.1f}s，设备 {scorer.device}，dtype {scorer.dtype}")

    if args.original and args.render:
        text = args.text
        if text is None and args.text_file:
            text = args.text_file.read_text(encoding="utf-8")
        if text is None:
            print("需要 --text 或 --text-file", file=sys.stderr)
            return 2
        prob, secs = scorer._forward_one(args.original, args.render, text)
        print(f"\nprob_yes = {prob:.5f}  ({secs:.2f}s)")
        if args.json_out:
            args.json_out.write_text(json.dumps(
                {"prob_yes": prob, "seconds": secs, "prompt_md5": md5}, ensure_ascii=False, indent=2
            ), encoding="utf-8")
        return 0

    results = run_examples(scorer, args.examples)
    print(f"\n{'case':<34}{'role':<14}{'label':<14}{'prob_yes':>9}{'官方值':>9}{'偏差':>8}{'秒':>7}")
    print("-" * 96)
    for r in results:
        obs = f"{r.observed:.5f}" if r.observed is not None else "-"
        delta = f"{r.prob_yes - r.observed:+.4f}" if r.observed is not None else "-"
        print(f"{r.case_id:<34}{r.role:<14}{r.label or '-':<14}{r.prob_yes:>9.5f}{obs:>9}{delta:>8}{r.seconds:>7.2f}")

    if torch.cuda.is_available():
        print(f"\n显存峰值 {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    if args.json_out:
        args.json_out.write_text(json.dumps(
            [r.__dict__ for r in results], ensure_ascii=False, indent=2
        ), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
