# -*- coding: utf-8 -*-
"""
分割质量评估模块

评测轴（五个，互不替代）：

1. **边界 F1** —— 算法给出的断点与 Wikipedia 天然段落边界的重合度。
   容忍窗 ±w token：预测断点落在真值断点 ±w 内算命中。w 取 block 长度的函数，
   太严会把「语义等价但位置偏一两句」的正确切分全判错。

2. **Pk / WindowDiff** —— 主题分割领域标准指标，衡量「相邻两块是否被错分到同一主题」。
   它对绝对位置不敏感，只看相对聚类对不对，正好补 F1 的盲区。

3. **块尺寸分布** —— 均值、标准差、最大值、超长块占比。
   分割算法的工程价值一半在「块别太大」（塞不进上下文/编码截断），
   光看召回会奖励切成单句。

4. **检索可用性** —— 把切出的块编码成向量，用段落标题或首句做查询，
   看答案所在块能不能被检索到（Recall@k / MRR）。
   这是分割的唯一目的：块切得好不好，最终要看检索认不认。

5. **成本** —— 每篇耗时、每篇 LLM token 消耗（仅 LLM 算法）。
   语义算法贵，要在质量-成本曲线上给出位置。

所有断点统一在 token 空间比较（见 splitter_base）。
"""
import json
import time
from dataclasses import dataclass, field, asdict

import numpy as np


# ── 1. 边界 F1 ────────────────────────────────────────────────────────────

def boundary_prf(pred, gold, tol):
    """带容忍窗的边界精确率/召回率/F1。

    Args:
        pred: 预测断点位置列表
        gold: 真值断点位置列表
        tol: 容忍窗（token 数）

    Returns:
        (precision, recall, f1)
    """
    pred, gold = sorted(pred), sorted(gold)
    if not pred and not gold:
        return 1.0, 1.0, 1.0
    if not pred:
        return 0.0, 0.0, 0.0
    if not gold:
        return 0.0, 1.0, 0.0

    hit_p = set()
    for p in pred:
        if any(abs(p - g) <= tol for g in gold):
            hit_p.add(p)
    hit_g = set()
    for g in gold:
        if any(abs(p - g) <= tol for p in pred):
            hit_g.add(g)

    prec = len(hit_p) / len(pred)
    rec = len(hit_g) / len(gold)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


# ── 2. Pk 与 WindowDiff ──────────────────────────────────────────────────

def _reference_segments(n_tokens, gold, k):
    """真值分割在长度为 k 的滑动窗下的「是否跨界」序列。"""
    ref = [0] * (n_tokens - k + 1) if n_tokens >= k else []
    gs = set(gold)
    for i in range(len(ref)):
        # 窗 [i, i+k) 内若含真值断点则 ref[i]=1
        ref[i] = 1 if any(i < g < i + k for g in gs) else 0
    return ref


def pk_score(pred, gold, n_tokens, k):
    """Pk：预测分割与真值分割在滑动窗上的不一致概率，越小越好。"""
    ps = [0] * (n_tokens - k + 1) if n_tokens >= k else []
    pset = set(pred)
    for i in range(len(ps)):
        ps[i] = 1 if any(i < p < i + k for p in pset) else 0
    ref = _reference_segments(n_tokens, gold, k)
    if not ps:
        return 0.0
    return float(np.mean([a != b for a, b in zip(ps, ref)]))


def windowdiff(pred, gold, n_tokens, k):
    """WindowDiff：比较窗内断点数是否一致，越小越好。"""
    ps = [0] * (n_tokens - k + 1) if n_tokens >= k else []
    pset = set(pred)
    for i in range(len(ps)):
        ps[i] = sum(1 for p in pset if i < p < i + k)
    gset = set(gold)
    ref = [sum(1 for g in gset if i < g < i + k)
           for i in range(len(ps))]
    if not ps:
        return 0.0
    return float(np.mean([a != b for a, b in zip(ps, ref)]))


# ── 3. 块尺寸分布 ─────────────────────────────────────────────────────────

@dataclass
class SizeStats:
    n_chunks: int
    mean: float
    std: float
    p50: float
    p95: float
    max: int
    over_ratio: float        # 超过 limit 的块占比


def size_stats(spans, n_tokens, limit=512):
    sizes = [e - s for s, e in spans]
    if not sizes:
        return SizeStats(0, 0, 0, 0, 0, 0, 0.0)
    a = np.array(sizes)
    return SizeStats(
        n_chunks=len(sizes),
        mean=float(a.mean()),
        std=float(a.std()),
        p50=float(np.percentile(a, 50)),
        p95=float(np.percentile(a, 95)),
        max=int(a.max()),
        over_ratio=float((a > limit).mean()),
    )


# ── 5. 成本 ───────────────────────────────────────────────────────────────

def timed(fn, *args, **kwargs):
    """返回 (结果, 秒数)。"""
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0


# ── 汇总 ─────────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    algo: str
    kind: str
    n_docs: int
    f1: float
    precision: float
    recall: float
    pk: float
    windowdiff: float
    size: SizeStats
    avg_seconds: float
    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    mrr: float = 0.0
    total_tokens: int = 0
    notes: str = ""

    def to_dict(self):
        d = asdict(self)
        d["size"] = asdict(d["size"])
        return d
