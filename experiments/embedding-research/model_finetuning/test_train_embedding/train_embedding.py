# -*- coding: utf-8 -*-
"""
一个用于微调文本嵌入模型的PyTorch训练脚本。

该脚本实现了基于对比学习的文本嵌入模型训练流程，
使用in-batch negatives策略来构建正负样本对。
"""
import os
import json
import argparse
from tqdm.auto import tqdm
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from sentence_transformers import SentenceTransformer

# 设置环境变量，避免多进程分词器可能引发的问题
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- 模型定义 ---
class EmbeddingModel(nn.Module):
    """
    文本嵌入模型，封装了SentenceTransformer以进行微调。
    """
    def __init__(self, model_name_or_path, temperature=0.05):
        super().__init__()
        self.model = SentenceTransformer(model_name_or_path, trust_remote_code=True)
        self.temperature = temperature

    def forward(self, query_input_ids, query_attention_mask, pos_doc_input_ids, pos_doc_attention_mask):
        """模型的前向传播"""
        query_embeddings = self.model({'input_ids': query_input_ids, 'attention_mask': query_attention_mask})['sentence_embedding']
        pos_doc_embeddings = self.model({'input_ids': pos_doc_input_ids, 'attention_mask': pos_doc_attention_mask})['sentence_embedding']

        loss, accuracy = self.calculate_contrastive_loss(query_embeddings, pos_doc_embeddings)
        
        return {'loss': loss, 'accuracy': accuracy}

    def calculate_contrastive_loss(self, query_embeddings, pos_doc_embeddings):
        """
        计算In-batch Negatives的对比学习损失。
        """
        # 步骤1: 归一化嵌入向量
        query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
        pos_doc_embeddings = F.normalize(pos_doc_embeddings, p=2, dim=-1)

        # 步骤2: 计算批内所有查询与文档的相似度矩阵
        sim_matrix = query_embeddings @ pos_doc_embeddings.transpose(-1, -2)
        sim_matrix = sim_matrix / self.temperature
        
        # 步骤3: 创建标签，对角线上的为正样本对
        labels = torch.arange(query_embeddings.size(0), device=query_embeddings.device, dtype=torch.long)
        
        # 步骤4: 计算交叉熵损失
        loss = F.cross_entropy(sim_matrix, labels)
        
        # 步骤5: (可选) 计算准确率
        _, predicted = torch.max(sim_matrix, 1)
        correct = (predicted == labels).sum().item()
        accuracy = correct / labels.size(0)
        
        return loss, accuracy

    def save(self, save_dir):
        """保存模型和分词器"""
        self.model.save(save_dir)


# --- 数据处理 ---
class TextDataset(Dataset):
    """
    PyTorch数据集类，用于加载JSONL格式的文本对数据。
    """
    def __init__(self, data_path):
        super().__init__()
        self.samples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch, tokenizer, query_max_len, passage_max_len):
    """
    数据整理函数，将文本数据批处理为模型输入张量。
    """
    queries = [item["query"] for item in batch]
    pos_docs = [item["pos_doc"] for item in batch]

    query_tokens = tokenizer(queries, padding=True, truncation=True, max_length=query_max_len, return_tensors="pt")
    pos_doc_tokens = tokenizer(pos_docs, padding=True, truncation=True, max_length=passage_max_len, return_tensors="pt")

    return {
        "query_input_ids": query_tokens["input_ids"],
        "query_attention_mask": query_tokens["attention_mask"],
        "pos_doc_input_ids": pos_doc_tokens["input_ids"],
        "pos_doc_attention_mask": pos_doc_tokens["attention_mask"],
    }


# --- 训练器 ---
class Trainer:
    """
    负责执行模型训练和评估的训练器。
    """
    def __init__(self, model, optimizer, lr_scheduler, train_dataloader, epochs, device, output_dir):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.epochs = epochs
        self.device = device
        self.output_dir = output_dir

    def train(self):
        """执行完整的训练流程"""
        for epoch in range(1, self.epochs + 1):
            self.train_epoch(epoch)
            self.save_checkpoint(epoch)

    def train_epoch(self, epoch):
        """执行单个epoch的训练"""
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch}/{self.epochs}", unit="batch")
        
        for step, batch in enumerate(progress_bar):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            outputs = self.model(**batch)
            loss = outputs['loss']
            
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            
            total_loss += loss.item()
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}", 
                'acc': f"{outputs['accuracy']:.2%}"
            })

        avg_loss = total_loss / len(self.train_dataloader)
        print(f"Epoch {epoch} finished. Average Loss: {avg_loss:.4f}")

    def save_checkpoint(self, epoch):
        """保存模型检查点"""
        save_dir = os.path.join(self.output_dir, f"checkpoint-epoch-{epoch}")
        os.makedirs(save_dir, exist_ok=True)
        print(f"Saving model checkpoint to {save_dir}")
        self.model.save(save_dir)


# --- 辅助函数和主逻辑 ---
def create_demo_data(file_path="demo_data.jsonl"):
    """如果数据文件不存在，则创建一个演示用的数据文件。"""
    if os.path.exists(file_path):
        return
    print(f"Demo data file not found. Creating at: {file_path}")
    data = [
        {"query": "如何学习Python？", "pos_doc": "学习Python首先要安装环境，然后从基础语法开始，比如变量、循环和函数。"},
        {"query": "什么是机器学习？", "pos_doc": "机器学习是人工智能的一个分支，它研究计算机如何从数据中学习。"},
        {"query": "北京有哪些旅游景点？", "pos_doc": "北京有很多名胜古迹，比如故宫、长城和颐和园。"},
        {"query": "红烧肉的家常做法", "pos_doc": "红烧肉的关键是炒糖色，然后加入五花肉慢炖入味。"}
    ]
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description="文本嵌入模型微调脚本")
    # 模型与数据路径参数
    parser.add_argument("--model_name_or_path", type=str, default="BAAI/bge-base-zh-v1.5", help="预训练模型名称或路径")
    parser.add_argument("--train_dataset", type=str, default="demo_data.jsonl", help='训练数据集路径')
    parser.add_argument('--output_dir', type=str, default="./models/bge-finetuned", help='模型输出目录')
    # 训练超参数
    parser.add_argument('--epochs', type=int, default=3, help='训练轮次')
    parser.add_argument("--lr", type=float, default=2e-5, help="学习率")
    parser.add_argument('--batch_size', type=int, default=4, help='每台设备的批处理大小')
    parser.add_argument('--query_max_len', type=int, default=64, help="查询的最大长度")
    parser.add_argument('--passage_max_len', type=int, default=256, help="文档段落的最大长度")
    args = parser.parse_args()

    # 1. 环境与数据准备
    create_demo_data(args.train_dataset)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. 初始化模型与分词器
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = EmbeddingModel(args.model_name_or_path, temperature=0.02).to(device)

    # 3. 创建数据集与数据加载器
    train_dataset = TextDataset(args.train_dataset)
    collate_with_tokenizer = partial(collate_fn, tokenizer=tokenizer, query_max_len=args.query_max_len, passage_max_len=args.passage_max_len)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_with_tokenizer, num_workers=4)

    # 4. 初始化优化器与学习率调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_dataloader) * args.epochs
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # 5. 初始化训练器并开始训练
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_dataloader=train_dataloader,
        epochs=args.epochs,
        device=device,
        output_dir=args.output_dir
    )

    print("--- Training Started ---")
    trainer.train()
    print("--- Training Finished ---")
    
    # 6. 保存最终模型
    final_save_dir = os.path.join(args.output_dir, "final_model")
    model.save(final_save_dir)
    tokenizer.save_pretrained(final_save_dir)
    print(f"Final model saved to: {final_save_dir}")


if __name__ == "__main__":
    main()