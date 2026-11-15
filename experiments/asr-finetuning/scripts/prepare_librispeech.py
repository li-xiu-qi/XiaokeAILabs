# -*- coding: utf-8 -*-
"""prepare_librispeech.py — 生成 16 kHz Whisper 训练/测试 manifest（小样本友好）。

从固定 revision 的 openslr/librispeech_asr（clean 配置）读取指定 split，
按秒数过滤、归一化文本、重采样为 16 kHz 单声道 PCM_16 WAV，写出互相隔离的
train/test manifest（JSONL）与音频文件。使用 streaming 模式，只下载需要的
样本，不落全量数据集。

用法与参数含义见 README.md；训练与评估脚本只消费本脚本的产物。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import asr_common as common


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="生成 16 kHz 训练/测试 manifest")
    parser.add_argument("--dataset", default=os.environ.get("ASR_DATASET", common.DEFAULT_DATASET))
    parser.add_argument("--dataset-revision", default=os.environ.get("ASR_DATASET_REVISION", ""))
    parser.add_argument("--dataset-config", default=common.DEFAULT_DATASET_CONFIG)
    parser.add_argument("--train-split", default="train.100")
    parser.add_argument("--test-split", default="test")
    parser.add_argument(
        "--output-dir", default=str(common.data_dir() / "manifests" / common.run_id())
    )
    parser.add_argument("--max-train-samples", type=int, default=200)
    parser.add_argument("--max-test-samples", type=int, default=50)
    parser.add_argument("--min-duration", type=float, default=0.5)
    parser.add_argument("--max-duration", type=float, default=30.0)
    parser.add_argument("--shuffle-buffer", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="允许覆盖已存在的 manifest")
    parser.add_argument("--allow-floating-revision", action="store_true")
    return parser.parse_args(argv)


def sanitize_id(value):
    """把样本 id 转成安全文件名片段。"""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def resample_linear(array, src_sr, dst_sr):
    """无第三方依赖的线性插值重采样（仅在数据集未按 16 kHz 解码时兜底）。"""
    import numpy as np

    if int(src_sr) == int(dst_sr):
        return array, dst_sr
    n = int(round(len(array) * float(dst_sr) / float(src_sr)))
    if n <= 0 or len(array) == 0:
        return array[:0], dst_sr
    pos = np.arange(n, dtype=np.float64) * (len(array) - 1) / max(n - 1, 1)
    return np.interp(pos, np.arange(len(array)), array).astype("float32"), dst_sr


def prepare_split(dataset, revision, config, split, tag, out_dir, args):
    """流式读取一个 split，落 WAV 与 manifest，返回 (manifest 路径, 统计)。"""
    import io
    import numpy as np
    from datasets import Audio, load_dataset
    import soundfile as sf

    ds = load_dataset(dataset, config, split=split, streaming=True, revision=revision or None)
    try:
        ds = ds.cast_column("audio", Audio(decode=False))
    except Exception as exc:
        raise RuntimeError(f"无法把 audio 列转换为 {common.TARGET_SAMPLE_RATE} Hz：{exc}") from exc
    ds = ds.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    audio_dir = out_dir / "audio" / tag
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / f"{tag}.jsonl"
    limit = args.max_train_samples if tag == "train" else args.max_test_samples
    used_names = set()
    stats = {"split": split, "scanned": 0, "kept": 0, "skipped": 0, "limit": limit}

    with open(manifest_path, "w", encoding="utf-8") as fh:
        for example in ds:
            if limit and stats["kept"] >= limit:
                break
            stats["scanned"] += 1
            audio = example.get("audio") or {}
            source = io.BytesIO(audio["bytes"]) if audio.get("bytes") is not None else audio.get("path")
            if not source:
                stats["skipped"] += 1
                continue
            array, sr = sf.read(source, dtype="float32")
            if array.ndim > 1:
                array = np.mean(array, axis=1)
            if array is None or len(array) == 0 or sr <= 0:
                stats["skipped"] += 1
                continue
            duration = len(array) / float(sr)
            if not (args.min_duration <= duration <= args.max_duration):
                stats["skipped"] += 1
                continue
            if sr != common.TARGET_SAMPLE_RATE:
                array, sr = resample_linear(array, sr, common.TARGET_SAMPLE_RATE)
            text = common.normalize_text(example.get("text", ""))
            if not text:
                stats["skipped"] += 1
                continue
            sample_id = str(example.get("id") or example.get("file") or f"{tag}-{stats['scanned']}")
            name = sanitize_id(sample_id)
            while name in used_names:
                name = f"{name}-dup"
            used_names.add(name)
            rel = f"audio/{tag}/{name}.wav"
            sf.write(out_dir / rel, array.astype("float32"), sr, subtype="PCM_16")
            row = {
                "id": sample_id,
                "audio": rel,
                "text": text,
                "duration": round(duration, 3),
                "sampling_rate": common.TARGET_SAMPLE_RATE,
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            stats["kept"] += 1
    return manifest_path, stats


def main(argv=None):
    args = parse_args(argv)
    if args.train_split == args.test_split:
        sys.exit("--train-split 与 --test-split 不能相同：训练集与测试集必须隔离")
    if args.min_duration <= 0 or args.max_duration <= args.min_duration:
        sys.exit("时长过滤参数不合法：需要 0 < min-duration < max-duration")
    if args.max_train_samples < 0 or args.max_test_samples < 0:
        sys.exit("样本上限不能为负（0 表示该 split 全量，体积大，慎用）")
    if args.shuffle_buffer < 1:
        sys.exit("--shuffle-buffer 至少为 1")
    if args.shuffle_buffer < max(args.max_train_samples, args.max_test_samples):
        print(
            "[prepare] 警告：shuffle-buffer 小于样本上限，streaming 随机性偏弱",
            file=sys.stderr,
        )

    revision = args.dataset_revision.strip()
    if not revision and not args.allow_floating_revision and common.require_revision():
        sys.exit(
            "未提供 --dataset-revision / ASR_DATASET_REVISION；默认要求固定数据集 revision。\n"
            "可用 huggingface_hub 查询当前 commit，例如：\n"
            "  python -c \"from huggingface_hub import HfApi; "
            "print(HfApi().model_info('openslr/librispeech_asr', repo_type='dataset').sha)\""
        )
    if not revision:
        print("[prepare] 警告：未固定数据集 revision，结果不可复现", file=sys.stderr)

    out_dir = Path(args.output_dir)
    existing = [p for p in (out_dir / "train.jsonl", out_dir / "test.jsonl") if p.exists()]
    if existing and not args.force:
        sys.exit(
            f"输出目录已包含 manifest：{[str(p) for p in existing]}；"
            "请更换 --output-dir / XL_RUN_ID，或显式 --force"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[prepare] dataset={args.dataset} revision={revision or '<floating>'} "
        f"config={args.dataset_config} splits={args.train_split}/{args.test_split}"
    )
    train_manifest, train_stats = prepare_split(
        args.dataset, revision, args.dataset_config, args.train_split, "train", out_dir, args
    )
    test_manifest, test_stats = prepare_split(
        args.dataset, revision, args.dataset_config, args.test_split, "test", out_dir, args
    )
    common.check_split_isolation(train_manifest, test_manifest)

    summary = {
        "dataset": args.dataset,
        "dataset_revision": revision or None,
        "dataset_config": args.dataset_config,
        "train_split": args.train_split,
        "test_split": args.test_split,
        "seed": args.seed,
        "min_duration": args.min_duration,
        "max_duration": args.max_duration,
        "sample_rate": common.TARGET_SAMPLE_RATE,
        "normalization": "asr_common.normalize_text（NFKC/小写/去标点/折叠空白）",
        "train": train_stats,
        "test": test_stats,
        "train_manifest": str(train_manifest),
        "test_manifest": str(test_manifest),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "prepare_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[prepare] train: {train_stats}")
    print(f"[prepare] test:  {test_stats}")
    print(f"[prepare] manifest 目录：{out_dir}")
    print("[prepare] 完成。下一步：train_whisper.py 或 run.sh train")


if __name__ == "__main__":
    main()
