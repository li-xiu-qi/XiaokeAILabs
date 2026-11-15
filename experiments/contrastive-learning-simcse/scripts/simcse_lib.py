# -*- coding: utf-8 -*-
"""
SimCSE 对比学习实验的公共库。

本文件被同目录下的训练与评测脚本共用，只依赖 PyTorch 与 Transformers，
不依赖 sentence-transformers：句向量由本仓库自己的池化代码产生，
避免引入额外依赖，也让损失与池化逻辑可被独立测试。

约定：
- 句子编码走 CLS + 线性变换的 SimCSE 原始方案；mean 池化仅作对照开关。
- 所有对比损失都在 L2 归一化后的单位超球面上计算，即余弦相似度。
"""

import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# 训练与评测共用的默认超参数。写在这里是为了让三个脚本的口径一致，
# 单次运行仍可用命令行覆盖。
DEFAULT_MAX_LENGTH = 32
DEFAULT_TEMPERATURE = 0.05
DEFAULT_POOLING = "cls"


@dataclass
class TrainConfig:
    """一次训练的完整超参数集合，便于落盘与复现。"""

    model_name_or_path: str
    epochs: int = 1
    batch_size: int = 64
    learning_rate: float = 3e-5
    temperature: float = DEFAULT_TEMPERATURE
    max_length: int = DEFAULT_MAX_LENGTH
    pooling: str = DEFAULT_POOLING
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    seed: int = 42
    device: str = "auto"
    # 监督版专用
    hard_negative_weight: float = 1.0
    # 无监督版专用：SimCSE 原论文的 MLM 辅助损失权重
    mlm_weight: float = 0.0
    mlm_probability: float = 0.15
    extra: Dict[str, str] = field(default_factory=dict)


# --------------------------------------------------------------------------
# 数据读取
# --------------------------------------------------------------------------
def read_jsonl(path: str) -> List[dict]:
    """读取 JSON Lines 文件，跳过空行。"""
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno} 不是合法 JSON：{exc}") from exc
    return rows


def read_sentences(path: str) -> List[str]:
    """
    读取无监督训练用的句子集合。

    每行一个 JSON 对象，取 "sentence" 或 "text" 字段；也接受纯文本行。
    """
    sentences: List[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                row = json.loads(line)
                text = row.get("sentence") or row.get("text")
                if not text:
                    raise ValueError(f"{path}:{lineno} 缺少 sentence/text 字段")
                sentences.append(str(text))
            else:
                sentences.append(line)
    if not sentences:
        raise ValueError(f"{path} 没有读到任何句子")
    return sentences


def read_triplets(path: str) -> List[Tuple[str, str, str]]:
    """
    读取监督训练用的 (anchor, positive, negative) 三元组。

    每行一个 JSON 对象，字段名接受 anchor/sentence1/query 三种别名。
    """
    triplets: List[Tuple[str, str, str]] = []
    for lineno, row in enumerate(read_jsonl(path), start=1):
        anchor = row.get("anchor") or row.get("sentence1") or row.get("query")
        positive = row.get("positive") or row.get("sentence2")
        negative = row.get("negative") or row.get("hard_negative")
        if not (anchor and positive and negative):
            raise ValueError(f"{path}:{lineno} 缺少 anchor/positive/negative 字段")
        triplets.append((str(anchor), str(positive), str(negative)))
    if not triplets:
        raise ValueError(f"{path} 没有读到任何三元组")
    return triplets


def read_sts_pairs(path: str) -> List[Tuple[str, str, float]]:
    """
    读取语义相似度评测对。

    每行一个 JSON 对象，字段为 sentence1 / sentence2 / score。
    """
    pairs: List[Tuple[str, str, float]] = []
    for lineno, row in enumerate(read_jsonl(path), start=1):
        s1 = row.get("sentence1") or row.get("s1")
        s2 = row.get("sentence2") or row.get("s2")
        score = row.get("score")
        if s1 is None or s2 is None or score is None:
            raise ValueError(f"{path}:{lineno} 缺少 sentence1/sentence2/score 字段")
        pairs.append((str(s1), str(s2), float(score)))
    if not pairs:
        raise ValueError(f"{path} 没有读到任何句子对")
    return pairs


def write_jsonl(path: str, rows: Iterable[dict]) -> None:
    """按 JSON Lines 写文件，供冒烟测试生成小样本。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------
# 编码器
# --------------------------------------------------------------------------
class SimCSEEncoder(nn.Module):
    """
    句向量编码器：Transformer 主干 + 池化 + SimCSE 的线性投影头。

    SimCSE 原文在 pooled output 之后接一层 tanh 激活的全连接（pooler）。
    该层随机初始化，与主干一起训练；mean 池化对照模式下不使用它。
    """

    def __init__(
        self,
        model_name_or_path: str,
        pooling: str = DEFAULT_POOLING,
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> None:
        super().__init__()
        if pooling not in ("cls", "mean"):
            raise ValueError(f"不支持的池化方式：{pooling}（可选 cls / mean）")
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.backbone = AutoModel.from_pretrained(model_name_or_path)
        self.pooling = pooling
        self.max_length = max_length
        self.hidden_size = self.config.hidden_size
        self.pooler = nn.Linear(self.hidden_size, self.hidden_size)
        nn.init.normal_(self.pooler.weight, std=0.02)
        nn.init.zeros_(self.pooler.bias)

    def encode_tokens(self, texts: Sequence[str]) -> Dict[str, torch.Tensor]:
        """把文本批量分词，返回可直接喂给主干的张量字典。"""
        batch = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {key: value.to(self.backbone.device) for key, value in batch.items()}

    def forward(self, texts: Sequence[str]) -> torch.Tensor:
        """前向一次，返回未归一化的句向量。"""
        encoded = self.encode_tokens(texts)
        output = self.backbone(**encoded)
        hidden = output.last_hidden_state
        if self.pooling == "cls":
            pooled = hidden[:, 0]
            return torch.tanh(self.pooler(pooled))
        mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    @torch.no_grad()
    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 64,
        normalize: bool = True,
    ) -> torch.Tensor:
        """推理用编码：关梯度、按 batch 切分，返回 (N, hidden) 矩阵。"""
        was_training = self.training
        self.eval()
        vectors: List[torch.Tensor] = []
        for start in range(0, len(texts), batch_size):
            chunk = self.forward(list(texts[start : start + batch_size]))
            if normalize:
                chunk = F.normalize(chunk, p=2, dim=-1)
            vectors.append(chunk.float().cpu())
        if was_training:
            self.train()
        return torch.cat(vectors, dim=0) if vectors else torch.empty(0, self.hidden_size)

    def save_pretrained(self, out_dir: str) -> None:
        """保存主干、分词器与池化配置，供评测脚本原样加载。"""
        os.makedirs(out_dir, exist_ok=True)
        self.backbone.save_pretrained(out_dir)
        self.tokenizer.save_pretrained(out_dir)
        torch.save(self.pooler.state_dict(), os.path.join(out_dir, "pooler.pt"))
        with open(os.path.join(out_dir, "simcse_config.json"), "w", encoding="utf-8") as handle:
            json.dump({"pooling": self.pooling, "max_length": self.max_length}, handle, indent=2)


def load_encoder(model_name_or_path: str, device: str = "auto") -> SimCSEEncoder:
    """
    加载编码器。

    目录里带 simcse_config.json 时按训练时保存的池化与截断长度恢复，
    否则视为普通 Transformers 模型，用默认的 CLS 池化。
    """
    cfg_path = os.path.join(model_name_or_path, "simcse_config.json")
    pooling, max_length = DEFAULT_POOLING, DEFAULT_MAX_LENGTH
    if os.path.isfile(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as handle:
            saved = json.load(handle)
        pooling = saved.get("pooling", pooling)
        max_length = saved.get("max_length", max_length)
    model = SimCSEEncoder(model_name_or_path, pooling=pooling, max_length=max_length)
    pooler_path = os.path.join(model_name_or_path, "pooler.pt")
    if os.path.isfile(pooler_path):
        state = torch.load(pooler_path, map_location="cpu")
        model.pooler.load_state_dict(state)
    return model.to(resolve_device(device))


def resolve_device(name: str = "auto") -> torch.device:
    """解析设备：auto 时按 CUDA、MPS、CPU 的顺序取第一个可用的。"""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """固定 Python、NumPy 与 PyTorch 的随机源。"""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------
# 对比损失
# --------------------------------------------------------------------------
def info_nce(
    z1: torch.Tensor, z2: torch.Tensor, temperature: float = DEFAULT_TEMPERATURE
) -> torch.Tensor:
    """
    SimCSE 的无监督对比损失。

    z1 与 z2 是同一批句子的两次独立前向（靠 dropout 产生差异）。
    批内除对角线外的样本全部充当负例，损失在两个方向上取平均。
    """
    if z1.shape != z2.shape:
        raise ValueError(f"两次前向的形状不一致：{tuple(z1.shape)} vs {tuple(z2.shape)}")
    z1 = F.normalize(z1, p=2, dim=-1)
    z2 = F.normalize(z2, p=2, dim=-1)
    batch_size = z1.size(0)
    if batch_size < 2:
        raise ValueError("info_nce 需要 batch_size >= 2，否则批内没有负例")
    labels = torch.arange(batch_size, device=z1.device)
    logits = z1 @ z2.t() / temperature
    loss = F.cross_entropy(logits, labels)
    loss = loss + F.cross_entropy(logits.t(), labels)
    return loss / 2


def supervised_contrastive(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    temperature: float = DEFAULT_TEMPERATURE,
) -> Tuple[torch.Tensor, float]:
    """
    监督版对比损失（Supervised SimCSE 的 NLI 三元组形式）。

    每个 anchor 的正例是它自己的 positive，负例有两类：
    1. 与它配对的 hard negative；
    2. 批内其他样本的 positive。

    返回 (loss, 批内准确率)。准确率指正例在 softmax 后是否排第一，
    是比 loss 更直观的收敛信号。
    """
    anchor = F.normalize(anchor, p=2, dim=-1)
    positive = F.normalize(positive, p=2, dim=-1)
    negative = F.normalize(negative, p=2, dim=-1)
    batch_size = anchor.size(0)
    labels = torch.arange(batch_size, device=anchor.device)
    sim_a_p = anchor @ positive.t() / temperature
    sim_a_n = anchor @ negative.t() / temperature
    sim_p_a = positive @ anchor.t() / temperature
    sim_p_n = positive @ negative.t() / temperature
    loss_a = F.cross_entropy(torch.cat([sim_a_p, sim_a_n], dim=1), labels)
    loss_p = F.cross_entropy(torch.cat([sim_p_a, sim_p_n], dim=1), labels)
    loss = (loss_a + loss_p) / 2
    with torch.no_grad():
        correct = (sim_a_p.argmax(dim=1) == labels).float().mean().item()
    return loss, correct


def mlm_loss(
    model: SimCSEEncoder,
    texts: Sequence[str],
    probability: float = 0.15,
) -> torch.Tensor:
    """
    SimCSE 原论文的可选辅助损失：在句子上做 masked language modeling。

    它防止主干被对比损失过度破坏。默认关闭，权重由训练脚本传入。
    掩码采用简化版：被选中的位置直接替换为 [MASK]，不做 80/10/10 拆分。
    """
    encoded = model.encode_tokens(texts)
    labels = encoded["input_ids"].clone()
    # 只在非特殊 token 上做掩码，避免模型学会复制 [CLS]/[SEP]
    special = set(model.tokenizer.all_special_ids)
    for row in range(labels.size(0)):
        for col in range(labels.size(1)):
            token_id = int(labels[row, col])
            if token_id in special:
                labels[row, col] = -100
            elif torch.rand(()) < probability:
                labels[row, col] = -100
    corrupted = dict(encoded)
    corrupted["input_ids"] = encoded["input_ids"].clone()
    mask_positions = labels != -100
    if not mask_positions.any():
        return encoded["input_ids"].new_zeros((), dtype=torch.float)
    corrupted["input_ids"][mask_positions] = model.tokenizer.mask_token_id
    logits = model.backbone(**corrupted).logits
    if logits is None:
        raise ValueError("该主干没有 MLM 头，无法使用 mlm_weight")
    return nn.functional.cross_entropy(
        logits[mask_positions].float(), labels[mask_positions], ignore_index=-100
    )


# --------------------------------------------------------------------------
# 评测指标
# --------------------------------------------------------------------------
def _rank(values: torch.Tensor) -> torch.Tensor:
    """平均秩排名， ties 取平均，等价于 scipy.stats.rankdata。"""
    order = torch.argsort(values)
    ranks = torch.empty_like(values, dtype=torch.float)
    sorted_values = values[order]
    index = 0
    while index < sorted_values.numel():
        stop = index
        while stop + 1 < sorted_values.numel() and sorted_values[stop + 1] == sorted_values[index]:
            stop += 1
        average = (index + stop) / 2 + 1
        ranks[order[index : stop + 1]] = average
        index = stop + 1
    return ranks


def spearman(predictions: Sequence[float], references: Sequence[float]) -> float:
    """Spearman 秩相关，纯 PyTorch 实现，避免额外依赖 scipy。"""
    pred = torch.tensor(list(predictions), dtype=torch.float)
    ref = torch.tensor(list(references), dtype=torch.float)
    if pred.numel() < 2:
        raise ValueError("Spearman 至少需要两个样本")
    pred_rank = _rank(pred)
    ref_rank = _rank(ref)
    pred_rank = pred_rank - pred_rank.mean()
    ref_rank = ref_rank - ref_rank.mean()
    denominator = pred_rank.norm() * ref_rank.norm()
    if denominator == 0:
        return 0.0
    return float((pred_rank @ ref_rank / denominator).item())


def cosine_similarities(
    model: SimCSEEncoder, pairs: Sequence[Tuple[str, str]], batch_size: int = 64
) -> List[float]:
    """批量计算句子对的余弦相似度。"""
    left = model.encode([pair[0] for pair in pairs], batch_size=batch_size)
    right = model.encode([pair[1] for pair in pairs], batch_size=batch_size)
    return [float(v) for v in (left * right).sum(dim=1)]


def recall_at_k(
    model: SimCSEEncoder,
    queries: Sequence[str],
    candidates: Sequence[str],
    gold: Sequence[Sequence[int]],
    batch_size: int = 64,
    k: int = 10,
) -> float:
    """
    检索 Recall@k。

    gold[i] 是第 i 个查询在 candidates 中的正确下标集合。
    """
    query_vectors = model.encode(queries, batch_size=batch_size)
    cand_vectors = model.encode(candidates, batch_size=batch_size)
    scores = query_vectors @ cand_vectors.t()
    topk = scores.topk(min(k, scores.size(1)), dim=1).indices
    hits = 0
    for row, gold_rows in enumerate(gold):
        if any(int(idx) in set(gold_rows) for idx in topk[row]):
            hits += 1
    return hits / max(len(queries), 1)


def pearson(predictions: Sequence[float], references: Sequence[float]) -> float:
    """Pearson 相关，作为 Spearman 的补充指标输出。"""
    pred = torch.tensor(list(predictions), dtype=torch.float)
    ref = torch.tensor(list(references), dtype=torch.float)
    pred = pred - pred.mean()
    ref = ref - ref.mean()
    denominator = pred.norm() * ref.norm()
    if denominator == 0:
        return 0.0
    return float((pred @ ref / denominator).item())


def format_metrics(metrics: Dict[str, float]) -> str:
    """把指标字典格式化成对齐的文本，供终端输出与日志复用。"""
    width = max(len(key) for key in metrics) if metrics else 0
    return "\n".join(f"{key.ljust(width)}  {value:.4f}" for key, value in metrics.items())


def steps_per_epoch(num_samples: int, batch_size: int) -> int:
    """向上取整的每 epoch 步数。"""
    return max(1, math.ceil(num_samples / batch_size))
