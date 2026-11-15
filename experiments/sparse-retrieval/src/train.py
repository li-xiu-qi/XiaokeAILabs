import os
import argparse
import json
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from model import SpladeModel
from loss import SpladeLoss
from dataset import TripletDataset

def count_non_zeros(sparse_tensor: torch.Tensor, threshold: float = 1e-4) -> float:
    """统计批次内每个样本的非零激活词项数均值"""
    active_mask = (sparse_tensor > threshold).float()
    return active_mask.sum(dim=-1).mean().item()

def resolve_data_path(data_path, synthetic_data):
    """Require an explicit choice; a missing file must not silently become synthetic data."""
    if synthetic_data:
        return None
    if not data_path or not os.path.isfile(data_path):
        raise FileNotFoundError(f"Training data file not found: {data_path}")
    return data_path


def reserve_output_dir(path):
    """Claim a new or empty directory before loading a model or starting training."""
    os.makedirs(path, exist_ok=True)
    if os.listdir(path):
        raise FileExistsError(f"Training output directory is not empty: {path}")
    with open(os.path.join(path, ".training-run"), "x", encoding="utf-8") as marker:
        marker.write("Reserved for one training run; use a new output directory to rerun.\n")


def train(args):
    data_path = resolve_data_path(args.data_path, args.synthetic_data)
    reserve_output_dir(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using device: {device}")

    # 1. 词表分词器与模型初始化
    print(f"[Init] Loading model and tokenizer from {args.model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = SpladeModel(args.model_name_or_path).to(device)
    
    # 2. 损失函数与优化器
    criterion = SpladeLoss(lambda_flops=args.lambda_flops, margin=args.margin)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # 3. 数据集与加载器
    dataset = TripletDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_len_q=args.max_len_q,
        max_len_d=args.max_len_d
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn
    )

    history = []

    print(f"[Train] Starting fine-tuning for {args.epochs} epochs (Dataset size: {len(dataset)})...")
    model.train()
    step = 0

    for epoch in range(args.epochs):
        epoch_total_loss = 0.0
        for batch in dataloader:
            step += 1
            optimizer.zero_grad()

            q_inputs = {k: v.to(device) for k, v in batch["query"].items()}
            pos_inputs = {k: v.to(device) for k, v in batch["pos"].items()}
            neg_inputs = {k: v.to(device) for k, v in batch["neg"].items()}

            # 前向编码全词表稀疏向量
            sparse_q = model(input_ids=q_inputs["input_ids"], attention_mask=q_inputs["attention_mask"])
            sparse_pos = model(input_ids=pos_inputs["input_ids"], attention_mask=pos_inputs["attention_mask"])
            sparse_neg = model(input_ids=neg_inputs["input_ids"], attention_mask=neg_inputs["attention_mask"])

            # 计算排序内积得分
            score_pos = model.compute_similarity(sparse_q, sparse_pos)
            score_neg = model.compute_similarity(sparse_q, sparse_neg)

            # 联合损失反向传播
            loss, loss_dict = criterion(score_pos, score_neg, sparse_q, sparse_pos, sparse_neg)
            loss.backward()
            optimizer.step()

            epoch_total_loss += loss.item()

            # 计算稀疏度度量指标
            nz_q = count_non_zeros(sparse_q)
            nz_pos = count_non_zeros(sparse_pos)

            log_entry = {
                "step": step,
                "epoch": epoch + 1,
                "total_loss": round(loss_dict["total_loss"], 5),
                "rank_loss": round(loss_dict["rank_loss"], 5),
                "sparse_loss": round(loss_dict["sparse_loss"], 5),
                "active_terms_q": round(nz_q, 1),
                "active_terms_doc": round(nz_pos, 1),
                "score_margin": round((score_pos - score_neg).mean().item(), 4)
            }
            history.append(log_entry)

            if step % args.log_interval == 0 or step == 1:
                print(f"Step {step:03d} | Total Loss: {log_entry['total_loss']:.4f} "
                      f"(Rank: {log_entry['rank_loss']:.4f}, Sparsity: {log_entry['sparse_loss']:.4f}) | "
                      f"Margin: {log_entry['score_margin']:.3f} | "
                      f"Active Terms (Q/D): {log_entry['active_terms_q']}/{log_entry['active_terms_doc']}")

    # 4. 保存微调权重与实验指标
    model_save_path = os.path.join(args.output_dir, "splade_weights.pt")
    torch.save(model.state_dict(), model_save_path)
    print(f"[Save] Model weights saved to {model_save_path}")

    metrics_save_path = os.path.join(args.output_dir, "training_metrics.json")
    with open(metrics_save_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"[Save] Metrics written to {metrics_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SPLADE Fine-Tuning Pipeline")
    parser.add_argument("--model_name_or_path", type=str, default="bert-base-chinese")
    data_source = parser.add_mutually_exclusive_group(required=True)
    data_source.add_argument("--data_path", help="JSONL file containing query, pos and neg")
    data_source.add_argument("--synthetic_data", action="store_true", help="Use four built-in synthetic examples")
    parser.add_argument("--output_dir", type=str, default=".work/artifacts")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lambda_flops", type=float, default=0.0008)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--max_len_q", type=int, default=64)
    parser.add_argument("--max_len_d", type=int, default=256)
    parser.add_argument("--log_interval", type=int, default=2)

    args = parser.parse_args()
    train(args)
