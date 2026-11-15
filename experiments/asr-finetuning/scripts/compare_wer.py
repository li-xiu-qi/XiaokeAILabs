# -*- coding: utf-8 -*-
"""compare_wer.py — 对比同一测试集上两次 eval_wer.py 的结果。

读取基线（微调前）与微调后的 WER JSON，输出差值、相对改善与样本量核对；
测试 manifest 不一致时给出警告，避免拿不同测试集的数字下结论。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_result(path):
    path = Path(path)
    if not path.is_file():
        sys.exit(f"结果文件不存在：{path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        sys.exit(f"{path} 不是合法 JSON：{exc}")
    if "wer_percent" not in data:
        sys.exit(f"{path} 缺少 wer_percent 字段（确认由 eval_wer.py 生成）")
    return data


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="对比两次 WER 评估结果")
    parser.add_argument("--baseline", required=True, help="微调前的 WER JSON")
    parser.add_argument("--finetuned", required=True, help="微调后的 WER JSON")
    parser.add_argument("--output", default=None, help="对比结果 JSON 输出路径（可选）")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    baseline = load_result(args.baseline)
    finetuned = load_result(args.finetuned)

    base_wer = float(baseline["wer_percent"])
    fine_wer = float(finetuned["wer_percent"])
    manifest_match = baseline.get("manifest") == finetuned.get("manifest")
    if not manifest_match:
        print(
            "[compare] 警告：两次评估的 manifest 不一致，"
            f"{baseline.get('manifest')} vs {finetuned.get('manifest')}",
            file=sys.stderr,
        )
    n_match = baseline.get("n_samples") == finetuned.get("n_samples")
    if not n_match:
        print(
            "[compare] 警告：样本量不一致，"
            f"{baseline.get('n_samples')} vs {finetuned.get('n_samples')}",
            file=sys.stderr,
        )

    result = {
        "baseline_wer_percent": base_wer,
        "finetuned_wer_percent": fine_wer,
        "delta_percent_points": round(fine_wer - base_wer, 4),
        "relative_change_percent": (
            round((fine_wer - base_wer) / base_wer * 100.0, 4) if base_wer > 0 else None
        ),
        "improved": fine_wer < base_wer,
        "manifest_match": manifest_match,
        "n_samples_match": n_match,
        "baseline_model": baseline.get("model"),
        "finetuned_model": finetuned.get("model"),
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        f"[compare] 基线 WER {base_wer:.2f}% -> 微调后 {fine_wer:.2f}%"
        f"（变化 {result['delta_percent_points']:+.2f} 个百分点，"
        f"{'有改善' if result['improved'] else '无改善或退化'}）"
    )
    if args.output:
        print(f"[compare] 结果：{args.output}")


if __name__ == "__main__":
    main()
