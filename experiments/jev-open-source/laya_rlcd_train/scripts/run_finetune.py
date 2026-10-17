"""在单卡上复现 laya 的 RLCD 微调（2xT4 notebook 配方，去 DDP）。

配方与 notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb 对齐：
- 底座 convaiinnovations/laya 的 English checkpoint（max_len 512 / head_max_len 192）
- micro_batch=8 x grad_accum=8 = 有效 batch 64（train.py 注释确认的单卡等效）
- loss="rlcd"（默认）：GRPO 风格高斯噪声采样 + proper scoring rule 奖励 + soft CE
- epochs=4，rl_samples=4，sigma 0.4 -> 0.1，w_sph=0.75，w_rps=1.0
- train split 训练，10% 案例切校准，test split 做 held-out 评估

用法（DGX Spark 上，laya-rlcd-env）：
    HF_ENDPOINT=https://hf-mirror.com python run_rlcd_train.py \
        --data data/typed_decisions_train.jsonl \
        --eval-data data/typed_decisions_test.jsonl \
        --model-dir models/laya \
        --output-dir runs/agentjev_v1
"""
import argparse
import os

from huggingface_hub import snapshot_download

from laya.agent import _fix_tokenizer_config
from laya.train import TrainConfig, finetune


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--eval-data", default=None)
    ap.add_argument("--model-dir", default=None,
                    help="本地 checkpoint 目录；缺省从 HF 拉 convaiinnovations/laya 根目录")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--loss", default="rlcd", choices=["rlcd", "soft-ce"],
                    help="rlcd = GRPO 项 + soft CE（notebook 配方）；soft-ce = 纯蒸馏对照臂")
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--rl-samples", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = ap.parse_args()

    model_dir = args.model_dir
    if model_dir is None:
        model_dir = snapshot_download("convaiinnovations/laya")
    _fix_tokenizer_config(model_dir)

    # 单卡等效 notebook 配方（见 train.py TrainConfig 注释）：
    # notebook 有效 batch 64 = 8 per micro-batch x 2 GPU x 4 accum；单卡 = micro_batch 8 x grad_accum 8
    config = TrainConfig(
        epochs=args.epochs,
        micro_batch=args.micro_batch,
        grad_accum=args.grad_accum,
        loss=args.loss,
        rl_samples=args.rl_samples,
        seed=args.seed,
        calib_seed=args.seed,
        eval_data=args.eval_data,
        max_len=512,
        head_max_len=192,
    )
    summary = finetune(
        data=args.data,
        model_dir=model_dir,
        output_dir=args.output_dir,
        config=config,
        device=args.device,
    )
    print("\n=== SUMMARY ===")
    print(f"train_items: {summary['train_items']}, calibration_items: {summary['calibration_items']}, "
          f"eval_items: {summary['eval_items']}, skipped: {summary['skipped']}")
    print(f"epoch_loss: {summary['epoch_loss']}")
    tr = summary["train_report"]
    print(f"eval_mode: {tr['eval_mode']} (held_out={tr['is_held_out']}) — {tr['note']}")
    print(f"before: {tr['before']}")
    print(f"after:  {tr['after']}")
    print(f"delta:  {tr['comparison']}")
    print(f"temperature: {summary['temperature']}")
    print("saved:", summary.get("output_dir"))


if __name__ == "__main__":
    main()
