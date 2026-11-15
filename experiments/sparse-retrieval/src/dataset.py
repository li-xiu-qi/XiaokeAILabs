import json
import torch
from torch.utils.data import Dataset

class TripletDataset(Dataset):
    """
    稀疏检索三元组数据集：(Query, Positive Document, Negative Document)
    支持从 JSONL 读取真实数据或生成合成微调数据用于冒烟验证。
    """
    def __init__(self, data_path: str = None, tokenizer = None, max_len_q: int = 64, max_len_d: int = 256):
        self.tokenizer = tokenizer
        self.max_len_q = max_len_q
        self.max_len_d = max_len_d
        self.samples = []

        if data_path:
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.samples.append(json.loads(line))
        else:
            # 默认构造合成实验样本（覆盖专有名词与词项扩展场景）
            self.samples = [
                {
                    "query": "什么是稀疏向量检索",
                    "pos": "稀疏向量检索利用神经网络如 SPLADE 预测全词表权重，并在未出现的词项上进行语义扩展，结合倒排索引实现高精度检索。",
                    "neg": "向量数据库中的稠密检索使用固定维度的连续稠密向量表征语义，通常在专有名词与精确匹配上容易发生语义漂移。"
                },
                {
                    "query": "SPLADE 词项扩展机制",
                    "pos": "SPLADE 重用预训练语言模型的 MLM 预测头，通过序列最大值池化将整句上下文投影至高维词表空间，自发激活同义词与近义词。",
                    "neg": "BM25 是经典的无监督稀疏检索统计模型，完全基于词频与逆文档频率计算匹配得分，无法扩展未在文档中出现的词项。"
                },
                {
                    "query": "FLOPS 正则化损失的作用",
                    "pos": "FLOPS 正则化针对每个词表维度的平均激活值施加平方和惩罚，严格约束倒排链长度，防止模型退化为伪稠密表示。",
                    "neg": "深度神经网络优化器如 AdamW 能够自适应调整学习率并施加权重衰减，是目前主流大模型预训练的基础优化算法。"
                },
                {
                    "query": "传统 BM25 的局限性",
                    "pos": "传统 BM25 存在词汇不匹配问题，当查询中的关键词未在目标文档中字面出现时，即使用户意图高度相关也无法被召回。",
                    "neg": "流水线并行是一种将模型不同层切分至多个计算节点的分布式训练与推理技术，能够显著减少单机显存压力。"
                }
            ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        return item["query"], item["pos"], item["neg"]

    def collate_fn(self, batch):
        queries, poses, negs = zip(*batch)
        
        q_enc = self.tokenizer(
            list(queries),
            padding=True,
            truncation=True,
            max_length=self.max_len_q,
            return_tensors="pt"
        )
        pos_enc = self.tokenizer(
            list(poses),
            padding=True,
            truncation=True,
            max_length=self.max_len_d,
            return_tensors="pt"
        )
        neg_enc = self.tokenizer(
            list(negs),
            padding=True,
            truncation=True,
            max_length=self.max_len_d,
            return_tensors="pt"
        )
        
        return {
            "query": q_enc,
            "pos": pos_enc,
            "neg": neg_enc
        }
