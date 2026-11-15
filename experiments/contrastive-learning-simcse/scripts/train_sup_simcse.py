# -*- coding: utf-8 -*-
"""
监督版 SimCSE 训练脚本。

数据是 (anchor, positive, hard negative) 三元组，来自 NLI 等带标注语料。
每个 anchor 的负例有两类：与它配对的 hard negative，以及批内其他样本的
positive；损失在两个方向（anchor→positive、positive→anchor）上取平均。

用法示例：
    python train_sup_simcse.py --model bert-base-uncased --data nli_triplets.jsonl --out runs/sup
"""

import argparse
import json
import os
import time
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader

from simcse_lib import (
    TrainConfig,
    load_encoder,
    read_triplets,
    resolve_device,
    set_seed,
    steps_per_epoch,
    supervised_contrastive,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="监督版 SimCSE（NLI 三元组）训练")
    parser.add_argument("--model", required=True, help="基座模型名或本地路径，如 bert-base-uncased")
    parser.add_argument(
        "--data", required=True, help="三元组，JSON Lines，含 anchor/positive/negative"
    )
    parser.add_argument("--out", required=True, help="输出目录，保存模型与训练指标")
    parser.add_argument("--epochs", type=int, default=3, help="NLI 数据常用 1-3 个 epoch")
    parser.add_argument("--batch-size", type=int, default=512, help="监督版靠大批量提供负例")
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--pooling", choices=["cls", "mean"], default="cls")
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda / cuda:0")
    parser.add_argument("--limit", type=int, default=0, help="只取前 N 条三元组，用于冒烟测试")
    parser.add_argument("--log-every", type=int, default=20)
    return parser.parse_args()


def build_loader(triplets: List[Tuple[str, str, str]], batch_size: int) -> DataLoader:
    """按 batch 切分三元组，collate 返回三个字符串列表。"""

    def collate(batch: List[Tuple[str, str, str]]):
        anchors = [item[0] for item in batch]
        positives = [item[1] for item in batch]
        negatives = [item[2] for item in batch]
        return anchors, positives, negatives

    return DataLoader(triplets, batch_size=batch_size, shuffle=True, collate_fn=collate)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    triplets = read_triplets(args.data)
    if args.limit:
        triplets = triplets[: args.limit]

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
        extra={"triplets": str(len(triplets)), "device": str(device)},
    )

    model = load_encoder(args.model, device=args.device)
    model.train()
    loader = build_loader(triplets, config.batch_size)

    total_steps = steps_per_epoch(len(triplets), config.batch_size) * config.epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, (step + 1) / max(1, warmup_steps)) if warmup_steps else 1.0,
    )

    print(f"设备 {device} · 三元组 {len(triplets)} 条 · 总步数 {total_steps} · 预热 {warmup_steps}")
    history = []
    global_step = 0
    started = time.time()
    for epoch in range(config.epochs):
        for anchors, positives, negatives in loader:
            z_a = model(anchors)
            z_p = model(positives)
            z_n = model(negatives)
            loss, accuracy = supervised_contrastive(z_a, z_p, z_n, config.temperature)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            global_step += 1
            history.append(
                {"step": global_step, "loss": float(loss.item()), "batch_accuracy": accuracy}
            )
            if global_step % args.log_every == 0 or global_step == 1:
                print(
                    f"epoch {epoch + 1} step {global_step}/{total_steps} "
                    f"loss {loss.item():.4f} 批内准确率 {accuracy:.3f}"
                )

    elapsed = time.time() - started
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tail = history[-max(1, len(history) // 10) :] if history else []
    metrics = {
        "config": {
            **{k: v for k, v in vars(config).items() if k != "extra"},
            **config.extra,
        },
        "steps": global_step,
        "elapsed_seconds": round(elapsed, 2),
        "final_loss": history[-1]["loss"] if history else None,
        "final_batch_accuracy": history[-1]["batch_accuracy"] if history else None,
        "mean_loss_last_10pct": (sum(item["loss"] for item in tail) / len(tail) if tail else None),
        "mean_accuracy_last_10pct": (
            sum(item["batch_accuracy"] for item in tail) / len(tail) if tail else None
        ),
        "loss_curve": history,
    }
    with open(os.path.join(args.out, "train_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    print(f"完成：{global_step} 步，用时 {elapsed:.1f}s，模型与指标已写入 {args.out}")


if __name__ == "__main__":
    main()
