import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM

class SpladeModel(nn.Module):
    """
    SPLADE: Sparse Lexical and Animate Language Model
    利用预训练语言模型的 MLM 预测头生成全词表加权稀疏向量，
    通过序列最大值池化（Sequence Max-Pooling）实现上下文加权与词项扩展。
    """
    def __init__(self, model_name_or_path: str):
        super().__init__()
        # 加载带 MLM 词表预测头的预训练模型
        self.encoder = AutoModelForMaskedLM.from_pretrained(model_name_or_path)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # 获取全词表未归一化 logits [B, L, |V|]
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # 掩码屏蔽填充位置
        mask_expanded = attention_mask.unsqueeze(-1).expand_as(logits)
        
        # ReLU 激活并应用对数缩放：w_j = log(1 + ReLU(logits))
        relu_log = torch.log(1.0 + F.relu(logits)) * mask_expanded
        
        # 沿序列维度执行最大值池化（Max-Pooling over sequence length）
        sparse_vec, _ = torch.max(relu_log, dim=1)  # [B, |V|]
        return sparse_vec

    @staticmethod
    def compute_similarity(sparse_q: torch.Tensor, sparse_d: torch.Tensor) -> torch.Tensor:
        """
        计算查询向量与文档向量之间的词项内积得分
        Score = sum_{v in V} w_{q, v} * w_{d, v}
        """
        return torch.sum(sparse_q * sparse_d, dim=-1)
