import torch
import torch.nn as nn
import torch.nn.functional as F

class SpladeLoss(nn.Module):
    """
    SPLADE 联合损失函数：
    1. 排序对比损失（Margin Ranking Loss 或 InfoNCE）拉大正负例在词项点积上的差距；
    2. FLOPS 物理正则化损失约束词表平均激活幅度的平方和，防止倒排链膨胀。
    """
    def __init__(self, lambda_flops: float = 0.0008, margin: float = 0.2):
        super().__init__()
        self.lambda_flops = lambda_flops
        self.margin = margin
        self.ranking_loss_fn = nn.MarginRankingLoss(margin=margin)

    @staticmethod
    def compute_flops(sparse_weights: torch.Tensor) -> torch.Tensor:
        """
        计算批次内词项激活度的 FLOPS 正则化惩罚项：
        L_flops = sum_{j=1}^{|V|} ( (1/B) * sum_{k=1}^B |w_{k, j}| )^2
        """
        # 平均激活幅度 [|V|]
        mean_abs = torch.mean(torch.abs(sparse_weights), dim=0)
        # 平方和惩罚
        return torch.sum(mean_abs ** 2)

    def forward(
        self,
        score_pos: torch.Tensor,
        score_neg: torch.Tensor,
        sparse_q: torch.Tensor,
        sparse_pos: torch.Tensor,
        sparse_neg: torch.Tensor
    ):
        # 1. 排序间隔损失：score_pos - score_neg > margin
        target = torch.ones_like(score_pos)
        rank_loss = self.ranking_loss_fn(score_pos, score_neg, target)
        
        # 2. 针对查询与文档分别计算 FLOPS 稀疏惩罚
        flops_q = self.compute_flops(sparse_q)
        flops_d = 0.5 * (self.compute_flops(sparse_pos) + self.compute_flops(sparse_neg))
        sparse_loss = 0.5 * (flops_q + flops_d)
        
        # 3. 联合加权总损失
        total_loss = rank_loss + self.lambda_flops * sparse_loss
        
        return total_loss, {
            "rank_loss": rank_loss.item(),
            "sparse_loss": sparse_loss.item(),
            "total_loss": total_loss.item()
        }
