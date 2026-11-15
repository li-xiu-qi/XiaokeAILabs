# -*- coding: utf-8 -*-
"""
无监督 SimCSE 训练脚本。

同一批句子做两次独立前向，靠 dropout 的随机性得到两个略有差异的视图，
把它们当作正例对，批内其余样本作为负例，用 InfoNCE 损失训练。

用法示例：
    python train_unsup_simcse.py --model bert-base-uncased --data sentences.jsonl --out runs/unsup
"""

import argparse
import json
import os
import time
from typing import List

import torch
from torch.utils.data import DataLoader

from simcse_lib import (
    TrainConfig,
    info_nce,
    load_encoder,
    mlm_loss,
    read_sentences,
    resolve_device,
    set_seed,
    steps_per_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="无监督 SimCSE（dropout 正例）训练")
    parser.add_argument("--model", required=True, help="基座模型名或本地路径，如 bert-base-uncased")
    parser.add_argument("--data", required=True, help="训练句子，JSON Lines，每行含 sentence 字段")
    parser.add_argument("--out", required=True, help="输出目录，保存模型与训练指标")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--temperature", type=float, default=0.05, help="InfoNCE 温度，常用 0.05")
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--pooling", choices=["cls", "mean"], default="cls")
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--mlm-weight", type=float, default=0.0, help="MLM 辅助损失权重，0 表示关闭"
    )
    parser.add_argument("--mlm-probability", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda / cuda:0")
    parser.add_argument("--limit", type=int, default=0, help="只取前 N 条句子，用于冒烟测试")
    parser.add_argument("--log-every", type=int, default=20)
    return parser.parse_args()


def build_loader(sentences: List[str], batch_size: int) -> DataLoader:
    """按 batch 切分句子。collate 直接返回字符串列表，分词在模型内完成。"""

    def collate(batch: List[str]) -> List[str]:
        return batch

    return DataLoader(sentences, batch_size=batch_size, shuffle=True, collate_fn=collate)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    sentences = read_sentences(args.data)
    if args.limit:
        sentences = sentences[: args.limit]

    config = TrainConfig(
        model_name_or_path=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        temperature=args.temperature,
        max_length=args.max_length,
        pooling=args.pooling,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        mlm_weight=args.mlm_weight,
        mlm_probability=args.mlm_probability,
        extra={"sentences": str(len(sentences)), "device": str(device)},
    )

    model = load_encoder(args.model, device=args.device)
    model.train()
    loader = build_loader(sentences, config.batch_size)

    total_steps = steps_per_epoch(len(sentences), config.batch_size) * config.epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, (step + 1) / max(1, warmup_steps)) if warmup_steps else 1.0,
    )

    print(f"设备 {device} · 句子 {len(sentences)} 条 · 总步数 {total_steps} · 预热 {warmup_steps}")
    history = []
    global_step = 0
    started = time.time()
    for epoch in range(config.epochs):
        for batch in loader:
            if len(batch) < 2:
                # InfoNCE 需要批内负例，单样本批次跳过
                continue
            # 两次前向：train 模式下 dropout 生效，得到两个不同视图
            z1 = model(batch)
            z2 = model(batch)
            loss = info_nce(z1, z2, config.temperature)
            if config.mlm_weight > 0:
                loss = loss + config.mlm_weight * mlm_loss(
                    model, batch, probability=config.mlm_probability
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            global_step += 1
            history.append({"step": global_step, "loss": float(loss.item())})
            if global_step % args.log_every == 0 or global_step == 1:
                print(f"epoch {epoch + 1} step {global_step}/{total_steps} loss {loss.item():.4f}")

    elapsed = time.time() - started
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    metrics = {
        "config": {
            **{k: v for k, v in vars(config).items() if k != "extra"},
            **config.extra,
        },
        "steps": global_step,
        "elapsed_seconds": round(elapsed, 2),
        "final_loss": history[-1]["loss"] if history else None,
        "mean_loss_last_10pct": (
            sum(item["loss"] for item in history[-max(1, len(history) // 10) :])
            / max(1, len(history) // 10)
            if history
            else None
        ),
        "loss_curve": history,
    }
    with open(os.path.join(args.out, "train_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    print(f"完成：{global_step} 步，用时 {elapsed:.1f}s，模型与指标已写入 {args.out}")


if __name__ == "__main__":
    main()
