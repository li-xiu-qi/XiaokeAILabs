"""Validate the tutorial's inputs before downloading a model or training."""
import json
import math
from pathlib import Path


def load_pairs(path):
    samples = []
    with open(path, encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"第 {line_number} 行不是有效 JSON: {exc.msg}") from exc
            if not isinstance(sample, dict) or any(
                not isinstance(sample.get(key), str) or not sample[key].strip()
                for key in ("query", "pos_doc")
            ):
                raise ValueError(f"第 {line_number} 行的 query 和 pos_doc 必须是非空字符串")
            samples.append(sample)
    if len(samples) < 2:
        raise ValueError("批内对比学习至少需要两条训练数据")
    return samples


def validate_training_settings(batch_size, epochs, lr, query_max_len, passage_max_len):
    if batch_size < 2:
        raise ValueError("批内对比学习的 batch_size 必须至少为 2")
    if epochs < 1:
        raise ValueError("epochs 必须至少为 1")
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("学习率必须是大于 0 的有限数")
    if query_max_len < 1 or passage_max_len < 1:
        raise ValueError("文本最大长度必须至少为 1")


def prepare_output_dir(path):
    output = Path(path)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"输出目录必须不存在或为空，以免覆盖已有结果: {path}")
    output.mkdir(parents=True, exist_ok=True)
