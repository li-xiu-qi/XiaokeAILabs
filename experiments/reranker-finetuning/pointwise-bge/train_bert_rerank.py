import argparse
import os
from pathlib import Path
import json
import random
import math
import numpy as np
import torch
import tqdm
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup
from accelerate import Accelerator
from accelerate.utils import set_seed
import matplotlib.pyplot as plt
from collections import defaultdict

# 为了避免 Tokenizer 的并行处理可能引发的问题，这里设置为 false
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 统一设定随机种子与确定性配置，增强可复现性
def seed_everything(seed: int = 42):
    # 固定 Python 层散列的随机性（影响 dict/集合迭代顺序等）
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 说明：CUBLAS_WORKSPACE_CONFIG 影响 GPU 上某些 GEMM 路径的确定性。
    # 必须在 Python 进程启动前设置（命令行/外部环境），此处仅做提示：
    #   CUBLAS_WORKSPACE_CONFIG=:16:8  或  :4096:2
    # 示例启动命令：
    #   CUBLAS_WORKSPACE_CONFIG=:16:8 PYTHONHASHSEED=42 python train_bert_rerank.py

    # 固定各层随机数发生器
    random.seed(seed)        # Python 随机
    np.random.seed(seed)     # NumPy 随机

    torch.manual_seed(seed)          # CPU 上的 torch 随机
    torch.cuda.manual_seed_all(seed) # 所有可见 GPU 上的 torch 随机

    # 强制使用确定性算法：若遇到不支持确定性的算子，会抛出错误，便于定位并替换
    torch.use_deterministic_algorithms(True)

    # cuDNN 设置：
    # deterministic=True  强制选择确定性实现
    # benchmark=False     关闭基于输入尺寸的自动最优算法搜索，以避免不同运行产生不同选择
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 关闭 TF32：FP32 运算落入 TensorCore 的 TF32 快速路径可能造成数值路径差异
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

# -----------------------------------
# 1. 定义损失函数 (Loss Function)
# -----------------------------------
def pointwise_bce(logits, labels):
    """
    Pointwise Binary Cross-Entropy loss.
    这是一个常用的损失函数，适用于处理二分类或者标签被归一化到 0-1 区间的情况。
    """
    return nn.BCEWithLogitsLoss(reduction="mean")(logits, labels)

# -----------------------------------
# 2. 定义模型 (Model Definition)
# -----------------------------------
class BertRerankerModel(nn.Module):
    """
    BERT Reranker 模型 (Cross-Encoder)。
    """
    def __init__(self, hf_model, tokenizer, query_format="{}", document_format="{}"):
        super().__init__()
        self.model = hf_model
        self.tokenizer = tokenizer
        self.query_format = query_format
        self.document_format = document_format

    def forward(self, batch, labels=None):
        output = self.model(**batch)
        loss = None
        if labels is not None:
            # 将模型的输出 logits 维度从 (batch_size, 1) 压缩到 (batch_size,)
            logits = output.logits.squeeze(-1)
            loss = pointwise_bce(logits, labels)
        
        # 将 loss 也包含在模型输出里，方便训练
        return {"logits": output.logits, "loss": loss}

    def preprocess(self, sentences_pairs, max_len):
        """
        处理输入的 query-document 对，将其拼接为模型需要的格式。
        """
        new_sentences_pairs = []
        for query, document in sentences_pairs:
            new_query = self.query_format.format(query.strip())
            new_document = self.document_format.format(document.strip())
            new_sentences_pairs.append([new_query, new_document])

        tokens = self.tokenizer(
            new_sentences_pairs,
            add_special_tokens=True,
            padding="longest",
            max_length=max_len,
            truncation='only_second', # 当超长时，只截断 document 部分
            return_tensors="pt",
        )
        return tokens

    @classmethod
    def from_pretrained(cls, model_name_or_path, num_labels=1, **kwargs):
        hf_model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=num_labels,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, use_fast=True, trust_remote_code=True
        )
        return cls(hf_model, tokenizer, **kwargs)
    
    def save_pretrained(self, save_dir):
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

# -----------------------------------
# 3. 定义数据集 (Dataset)
# -----------------------------------
class PointwiseRankerDataset(Dataset):
    def __init__(self, data_path, target_model, max_len=512, max_label=1, min_label=0):
        self.model = target_model
        self.max_len = max_len
        self.max_label = max_label
        self.min_label = min_label
        self.data = self._read_data(data_path)

    def _read_data(self, data_path):
        data = []
        label_distribution = defaultdict(int)
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in tqdm.tqdm(f, desc="Loading data"):
                item = json.loads(line.strip())
                query = item["query"].strip()
                text = item["content"].strip()
                # 将离散的 label 映射到 0-1 之间
                label = self._map_label(item.get("label", 0))
                label_distribution[f"{label:.2f}"] += 1
                data.append([query, text, label])
        
        # 打印标签分布，方便检查数据
        print("----- Label Distribution -----")
        total = sum(label_distribution.values())
        for label, count in sorted(label_distribution.items()):
            print(f"Label {label}: {count} ({count/total:.2%})")
            
        return data
    
    def _map_label(self, label):
        # 如果 label 范围不是 0-1，则进行归一化
        if self.max_label == self.min_label:
            return float(label)
        return (label - self.min_label) / (self.max_label - self.min_label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, batch):
        queries, docs, labels = zip(*batch)
        
        # 将 (query, doc) 对转换成模型需要的输入格式
        tokens = self.model.preprocess(list(zip(queries, docs)), self.max_len)
        label_batch = torch.tensor(labels, dtype=torch.float32)

        return tokens, label_batch

# -----------------------------------
# 4. 主训练逻辑 (Main Training Logic)
# -----------------------------------
def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Fine-tune a pointwise cross-encoder reranker")
    parser.add_argument("--model-name-or-path", "--model", default="BAAI/bge-reranker-large")
    parser.add_argument("--train-dataset", "--data", default="default_train_data.jsonl")
    parser.add_argument("--output-dir", "--output", default="./output_bge_reranker_large_finetuned")
    parser.add_argument("--max-len", type=positive_int, default=512)
    parser.add_argument("--epochs", type=positive_int, default=30)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", "--batch", type=positive_int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-proportion", type=float, default=0.1)
    parser.add_argument("--gradient-accumulation-steps", "--grad-accumulation", type=positive_int, default=8)
    parser.add_argument("--mixed-precision", "--precision", choices=("no", "fp16", "bf16"), default="bf16")
    args = parser.parse_args(argv)
    if not 0 <= args.warmup_proportion <= 1:
        parser.error("--warmup-proportion must be between 0 and 1")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("--lr must be a positive finite number")
    return vars(args)


def validate_paths(args):
    dataset = Path(args["train_dataset"])
    output = Path(args["output_dir"])
    if not dataset.is_file():
        raise FileNotFoundError(f"Training data not found: {dataset}")
    if output.exists() and not output.is_dir():
        raise NotADirectoryError(f"Output path is not a directory: {output}")
    for artifact in (output / "final_model", output / "loss_curve.png"):
        if artifact.exists() or artifact.is_symlink():
            raise FileExistsError(f"Output already exists: {artifact}; choose a new output directory")


def accumulated_loss_scale(step, total_samples, batch_size, accumulation_steps, actual_batch_size):
    """Correct Accelerate's fixed accumulation divisor for unequal microbatches."""
    group_start = (step // accumulation_steps) * accumulation_steps * batch_size
    group_samples = min(total_samples - group_start, accumulation_steps * batch_size)
    return accumulation_steps * actual_batch_size / group_samples


def main(argv=None):
    args = parse_args(argv)
    # 避免模型加载和训练发生后才发现输出将覆盖已有文件。
    validate_paths(args)

    # --- 初始化 ---
    seed_everything(args["seed"])  # 保证确定性与随机源一致
    set_seed(args["seed"])         # Accelerate 内部也使用相同种子
    accelerator = Accelerator(
        mixed_precision=args["mixed_precision"],
        gradient_accumulation_steps=args["gradient_accumulation_steps"],
    )
    if accelerator.num_processes != 1:
        raise RuntimeError("This example currently supports one process only.")
    accelerator.print(f"Training device: {accelerator.device}")
    accelerator.print(f"Training Args: {args}")

    # 用于保存每步 loss
    loss_list = []

    # --- 加载模型和 Tokenizer ---
    model = BertRerankerModel.from_pretrained(
        model_name_or_path=args["model_name_or_path"],
    )

    # --- 加载数据 ---
    train_dataset = PointwiseRankerDataset(
        data_path=args["train_dataset"],
        target_model=model,
        max_len=args["max_len"],
    )
    # 为 DataLoader 固定随机源
    g = torch.Generator()
    g.manual_seed(args["seed"])

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args["batch_size"],
        collate_fn=train_dataset.collate_fn,
        shuffle=True,
        num_workers=0,
        generator=g,
    )

    # --- 设置优化器和学习率调度器 ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args["lr"]), weight_decay=0.01)
    # Accelerate 在每轮 DataLoader 结束时同步剩余梯度，故每轮单独向上取整。
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
    updates_per_epoch = math.ceil(len(train_dataloader) / accelerator.gradient_accumulation_steps)
    total_steps = updates_per_epoch * args["epochs"]
    num_warmup_steps = int(args["warmup_proportion"] * total_steps)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps,
    )
    lr_scheduler = accelerator.prepare(lr_scheduler)

    # --- 开始训练 ---
    accelerator.print(f"Start training for {args['epochs']} epochs ...")
    progress_bar = tqdm.tqdm(range(total_steps), disable=not accelerator.is_local_main_process)
    
    for epoch in range(args["epochs"]):
        model.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                batch_inputs, batch_labels = batch
                outputs = model(batch_inputs, batch_labels)
                loss = outputs["loss"]

                # loss 是批内均值；按样本数加权，补偿 Accelerate 的固定累积除数。
                scale = accumulated_loss_scale(
                    step, len(train_dataset), args["batch_size"],
                    accelerator.gradient_accumulation_steps, len(batch_labels),
                )
                accelerator.backward(loss * scale)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # --- 日志记录 ---
            avg_loss = accelerator.gather(loss).mean().item()
            loss_list.append(avg_loss)

            if accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped:
                progress_bar.set_description(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, LR: {lr_scheduler.get_last_lr()[0]:.6f}")
                progress_bar.update(1)
    
    # --- 训练结束，保存模型 ---
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.print("Training finished! Saving model ...")
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(os.path.join(args["output_dir"], "final_model"))
        accelerator.print("Model saved successfully!")

    # 绘制损失曲线并保存图片
    plt.figure()
    plt.plot(loss_list)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.savefig(os.path.join(args['output_dir'], 'loss_curve.png'))
    plt.close()
    accelerator.end_training()


if __name__ == "__main__":
    main()
