# -*- coding: utf-8 -*-
"""train_whisper.py — 用 manifest 在本地微调 Whisper（小样本、固定 revision）。

自定义训练循环（不依赖 transformers.Trainer），输入为
prepare_librispeech.py 生成的 16 kHz WAV + 归一化文本 manifest；支持
CUDA / CPU、bf16 / fp32、每轮保存检查点，训练前校验训练/测试集隔离，
输出目录必须是不存在的新目录或空目录，避免覆盖已有产物。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor

import asr_common as common

MAX_LABEL_TOKENS = 448  # Whisper 解码器最大长度


class ManifestAsrDataset(Dataset):
    """读取 manifest 与对应 16 kHz WAV，产出 Whisper 输入特征与标签。"""

    def __init__(self, manifest_path, processor, max_duration=30.0, limit=0):
        rows = common.read_manifest(manifest_path)
        base = Path(manifest_path).resolve().parent
        self.base = base
        self.processor = processor
        self.rows = []
        dropped = 0
        for row in rows:
            try:
                duration = float(row.get("duration"))
            except (TypeError, ValueError):
                dropped += 1
                continue
            if not (0.0 < duration <= max_duration):
                dropped += 1
                continue
            text = common.normalize_text(row.get("text", ""))
            if not text:
                dropped += 1
                continue
            wav = (base / row["audio"]).resolve()
            if not wav.is_file():
                dropped += 1
                continue
            self.rows.append({"id": str(row["id"]), "audio": wav, "text": text})
        if limit and len(self.rows) > limit:
            self.rows = self.rows[:limit]
        self.dropped = dropped

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        array, sr = sf.read(str(row["audio"]), dtype="float32", always_2d=False)
        if array.ndim > 1:
            array = array.mean(axis=1)
        if int(sr) != common.TARGET_SAMPLE_RATE:
            raise ValueError(
                f"{row['audio']} 采样率为 {sr}，应为 {common.TARGET_SAMPLE_RATE}；"
                "请用 prepare_librispeech.py 重新生成 16 kHz 数据"
            )
        features = self.processor(
            audio=array, sampling_rate=common.TARGET_SAMPLE_RATE, return_tensors="pt"
        )
        labels = self.processor.tokenizer(row["text"]).input_ids
        if len(labels) > MAX_LABEL_TOKENS:
            labels = labels[:MAX_LABEL_TOKENS]
        return {
            "input_features": features["input_features"][0],
            "labels": labels,
            "id": row["id"],
        }


def make_collate(tokenizer):
    """把样本堆叠为 input_features 与 labels（pad 位置置 -100 不计损失）。"""

    def collate(batch):
        input_features = torch.stack(
            [torch.as_tensor(item["input_features"], dtype=torch.float32) for item in batch]
        )
        padded = tokenizer.pad(
            {"input_ids": [item["labels"] for item in batch]},
            padding=True,
            return_tensors="pt",
        )
        labels = padded["input_ids"].clone()
        # Whisper shares its padding and end-of-text token ID; mask only padding,
        # otherwise the model never learns the end-of-transcription token.
        labels[padded["attention_mask"] == 0] = -100
        decoder_start = tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        if labels.shape[1] and (labels[:, 0] == decoder_start).all():
            labels = labels[:, 1:]
        return {"input_features": input_features, "labels": labels}

    return collate


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Whisper 小样本微调")
    parser.add_argument("--train-manifest", required=True, help="prepare 生成的 train.jsonl")
    parser.add_argument("--test-manifest", default=None, help="可选，用于训练前校验集合隔离")
    parser.add_argument("--model", default=os.environ.get("ASR_MODEL", common.DEFAULT_MODEL))
    parser.add_argument("--model-revision", default=os.environ.get("ASR_MODEL_REVISION", ""))
    parser.add_argument("--output-dir", default=str(common.out_dir() / "model"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-duration", type=float, default=30.0)
    parser.add_argument("--limit-samples", type=int, default=0, help="调试用：只取前 N 条")
    parser.add_argument("--amp", choices=["auto", "bf16", "none"], default="auto")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--allow-floating-revision", action="store_true")
    return parser.parse_args(argv)


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def save_checkpoint(model, processor, path):
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(path))
    processor.save_pretrained(str(path))


def main(argv=None):
    args = parse_args(argv)
    started = time.time()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    train_manifest = Path(args.train_manifest)
    if not train_manifest.is_file():
        sys.exit(f"训练 manifest 不存在：{train_manifest}（先运行 prepare_librispeech.py）")
    if args.test_manifest:
        common.check_split_isolation(train_manifest, args.test_manifest)
    if args.epochs < 1 or args.batch_size < 1:
        sys.exit("--epochs 与 --batch-size 至少为 1")

    revision = args.model_revision.strip() or None
    if revision is None and not args.allow_floating_revision and common.require_revision():
        sys.exit(
            "未提供 --model-revision / ASR_MODEL_REVISION；默认要求固定模型 revision。\n"
            "可用 huggingface_hub 查询当前 commit，例如：\n"
            "  python -c \"from huggingface_hub import HfApi; "
            "print(HfApi().model_info('openai/whisper-small').sha)\""
        )

    out_dir = common.ensure_fresh_dir(args.output_dir)
    common.ensure_paths_outside(out_dir, [train_manifest, args.test_manifest])

    device = resolve_device(args.device)
    if device.type == "cpu":
        print("[train] 警告：未检测到 CUDA，使用 CPU 训练，仅适合冒烟验证", file=sys.stderr)
    if args.amp == "auto":
        amp_name = "bf16" if device.type == "cuda" else "none"
    else:
        amp_name = args.amp
    use_amp = amp_name == "bf16" and device.type == "cuda"
    if amp_name == "bf16" and device.type != "cuda":
        print("[train] 警告：CPU 上不用 bf16 autocast，按 fp32 运行", file=sys.stderr)

    print(f"[train] model={args.model} revision={revision or '<floating>'} device={device} amp={amp_name}")
    processor = WhisperProcessor.from_pretrained(args.model, revision=revision)
    model = WhisperForConditionalGeneration.from_pretrained(args.model, revision=revision)
    model.config.use_cache = False
    model.to(device)
    model.train()

    dataset = ManifestAsrDataset(
        train_manifest, processor, max_duration=args.max_duration, limit=args.limit_samples
    )
    if len(dataset) == 0:
        sys.exit(f"训练集为空（丢弃 {dataset.dropped} 条）：{train_manifest}")
    print(f"[train] 训练样本 {len(dataset)} 条（丢弃 {dataset.dropped} 条）")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=make_collate(processor.tokenizer),
        num_workers=args.num_workers,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    amp_dtype = torch.bfloat16 if use_amp else None

    epoch_losses = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                outputs = model(input_features=batch["input_features"], labels=batch["labels"])
                loss = outputs.loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"epoch {epoch} 出现非有限 loss：{loss.item()}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += float(loss.detach())
            steps += 1
        avg_loss = total_loss / max(steps, 1)
        epoch_losses.append(avg_loss)
        print(f"[train] epoch {epoch}/{args.epochs} 平均 loss {avg_loss:.4f}（{steps} 步）")
        save_checkpoint(model, processor, out_dir / f"checkpoint-epoch-{epoch}")

    final_dir = out_dir / "final-model"
    save_checkpoint(model, processor, final_dir)

    metrics = {
        "run_id": common.run_id(),
        "model": args.model,
        "model_revision": revision,
        "train_manifest": str(train_manifest),
        "test_manifest": str(args.test_manifest) if args.test_manifest else None,
        "n_train_samples": len(dataset),
        "dropped_samples": dataset.dropped,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "amp": amp_name,
        "device": str(device),
        "seed": args.seed,
        "epoch_loss": epoch_losses,
        "final_avg_loss": epoch_losses[-1] if epoch_losses else None,
        "wall_seconds": round(time.time() - started, 2),
        "started_at": datetime.fromtimestamp(started, tz=timezone.utc).isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "note": "loss 为训练集批次损失，不是 WER；效果对比用 eval_wer.py + compare_wer.py",
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[train] 最终模型：{final_dir}")
    print(f"[train] 指标：{out_dir / 'metrics.json'}")
    print("[train] 完成。下一步：eval_wer.py 在测试集上对比基线")


if __name__ == "__main__":
    main()
