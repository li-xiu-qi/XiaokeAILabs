# -*- coding: utf-8 -*-
"""eval_wer.py — 在测试 manifest 上计算 WER（基础模型与微调模型同一测试集）。

贪心或集束解码生成转写，参考与假设都经 asr_common.normalize_text 归一化后
用 jiwer 计算词错误率，结果写 JSON（含最差的若干样本），供 compare_wer.py
做训练前后对比。测试 manifest 只读，绝不参与训练。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

import asr_common as common


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Whisper WER 评估")
    parser.add_argument("--model", default=os.environ.get("ASR_MODEL", common.DEFAULT_MODEL))
    parser.add_argument("--model-revision", default=os.environ.get("ASR_MODEL_REVISION", ""))
    parser.add_argument("--manifest", required=True, help="测试 manifest（prepare 生成的 test.jsonl）")
    parser.add_argument("--output", default=None, help="结果 JSON 路径，默认 XL_OUT_DIR/wer-<tag>.json")
    parser.add_argument("--tag", default=None, help="结果文件名标签，默认由模型名生成")
    parser.add_argument("--max-samples", type=int, default=0, help="调试用：只评前 N 条，0 为全部")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--language", default="en")
    parser.add_argument("--task", default="transcribe")
    parser.add_argument("--num-beams", type=int, default=1, help="1 为贪心解码")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--allow-floating-revision", action="store_true")
    return parser.parse_args(argv)


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def sanitize_tag(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def load_audio(path):
    array, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if array.ndim > 1:
        array = array.mean(axis=1)
    if int(sr) != common.TARGET_SAMPLE_RATE:
        raise ValueError(f"{path} 采样率为 {sr}，应为 {common.TARGET_SAMPLE_RATE}")
    return array


def generate_batch(model, processor, features, args):
    """优先用 language/task 参数，旧版 transformers 回退 forced_decoder_ids。"""
    try:
        return model.generate(
            features,
            language=args.language,
            task=args.task,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
        )
    except TypeError:
        prompt_ids = processor.get_decoder_prompt_ids(language=args.language, task=args.task)
        return model.generate(
            features,
            forced_decoder_ids=prompt_ids,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
        )


def main(argv=None):
    args = parse_args(argv)
    manifest = Path(args.manifest)
    if not manifest.is_file():
        sys.exit(f"测试 manifest 不存在：{manifest}")
    if args.batch_size < 1 or args.max_samples < 0:
        sys.exit("--batch-size 至少为 1，--max-samples 不能为负")

    revision = args.model_revision.strip() or None
    if revision is None and not args.allow_floating_revision and common.require_revision():
        sys.exit(
            "未提供 --model-revision / ASR_MODEL_REVISION；默认要求固定模型 revision。\n"
            "评估本地微调产物（目录路径）时可用 --allow-floating-revision 跳过。"
        )

    device = resolve_device(args.device)
    tag = args.tag or sanitize_tag(Path(args.model).name or "model")
    output = Path(args.output) if args.output else common.out_dir() / f"wer-{tag}.json"
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = common.read_manifest(manifest)
    if args.max_samples and len(rows) > args.max_samples:
        rows = rows[: args.max_samples]
    base = manifest.resolve().parent

    print(f"[eval] model={args.model} revision={revision or '<floating>'} device={device} n={len(rows)}")
    processor = WhisperProcessor.from_pretrained(args.model, revision=revision)
    model = WhisperForConditionalGeneration.from_pretrained(args.model, revision=revision)
    model.to(device)
    model.eval()

    try:
        import jiwer
    except ImportError as exc:
        sys.exit(f"计算 WER 需要 jiwer：pip install jiwer（{exc}）")

    ids, references, hypotheses, skipped_empty = [], [], [], 0
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        arrays = [load_audio(base / row["audio"]) for row in batch]
        features = processor(
            audio=arrays, sampling_rate=common.TARGET_SAMPLE_RATE, return_tensors="pt"
        )["input_features"].to(device)
        with torch.no_grad():
            sequences = generate_batch(model, processor, features, args)
        texts = processor.batch_decode(sequences, skip_special_tokens=True)
        for row, text in zip(batch, texts):
            reference = common.normalize_text(row.get("text", ""))
            if not reference:
                skipped_empty += 1
                continue
            ids.append(str(row["id"]))
            references.append(reference)
            hypotheses.append(common.normalize_text(text))

    if not references:
        sys.exit("没有可评估样本（参考文本均为空？）")
    wer_value = jiwer.wer(references, hypotheses) * 100.0

    per_example = []
    for sample_id, ref, hyp in zip(ids, references, hypotheses):
        if not ref.split():
            continue
        per_example.append(
            {"id": sample_id, "wer": jiwer.wer([ref], [hyp]) * 100.0, "ref": ref, "hyp": hyp}
        )
    worst = sorted(per_example, key=lambda item: item["wer"], reverse=True)[:5]

    result = {
        "model": args.model,
        "model_revision": revision,
        "manifest": str(manifest),
        "n_samples": len(references),
        "skipped_empty_reference": skipped_empty,
        "language": args.language,
        "task": args.task,
        "num_beams": args.num_beams,
        "wer_percent": round(wer_value, 4),
        "normalization": "asr_common.normalize_text（NFKC/小写/去标点/折叠空白）",
        "worst_examples": worst,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[eval] WER {wer_value:.2f}%（{len(references)} 条，跳过空参考 {skipped_empty} 条）")
    print(f"[eval] 结果：{output}")


if __name__ == "__main__":
    main()
