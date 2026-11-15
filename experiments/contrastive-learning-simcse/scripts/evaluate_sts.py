# -*- coding: utf-8 -*-
"""
句向量评测脚本：语义相似度（Spearman / Pearson）与检索 Recall@k。

用法示例：
    python evaluate_sts.py --model runs/unsup --sts-files sts_dev.jsonl sts_test.jsonl
    python evaluate_sts.py --model runs/sup --sts-files sts_test.jsonl --retrieval retrieval.jsonl
"""

import argparse
import json
import os
from typing import Dict, List, Sequence, Tuple

import torch

from simcse_lib import (
    cosine_similarities,
    format_metrics,
    load_encoder,
    pearson,
    read_jsonl,
    read_sts_pairs,
    recall_at_k,
    resolve_device,
    spearman,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SimCSE 句向量评测")
    parser.add_argument("--model", required=True, help="待评测模型目录或 Transformers 模型名")
    parser.add_argument(
        "--sts-files",
        nargs="+",
        default=[],
        help="相似度评测文件，JSON Lines，每行含 sentence1/sentence2/score",
    )
    parser.add_argument(
        "--retrieval-file",
        default="",
        help="可选检索评测文件，JSON Lines，每行含 query 与 candidates/gold",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--recall-k", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--json-out", default="", help="把指标写成 JSON 文件，便于留档")
    return parser.parse_args()


def read_retrieval(path: str) -> Tuple[List[str], List[str], List[List[int]]]:
    """
    读取检索评测集。

    每行一个 JSON 对象：
      {"query": "...", "candidates": ["...", "..."], "gold": [0, 3]}
    gold 是该查询在 candidates 中的正确下标。所有行共用一个候选池，
    即第一个非空 candidates 决定候选集。
    """
    queries: List[str] = []
    gold: List[List[int]] = []
    candidates: List[str] = []
    for lineno, row in enumerate(read_jsonl(path), start=1):
        query = row.get("query") or row.get("sentence")
        rows_gold = row.get("gold") or row.get("positives")
        if query is None or rows_gold is None:
            raise ValueError(f"{path}:{lineno} 缺少 query/gold 字段")
        if not candidates:
            candidates = [str(item) for item in row.get("candidates", [])]
            if not candidates:
                raise ValueError(f"{path}:{lineno} 缺少 candidates 字段")
        queries.append(str(query))
        if isinstance(rows_gold[0], str):
            # gold 写成候选文本时，转成候选池中的下标
            index_of = {text: idx for idx, text in enumerate(candidates)}
            gold.append([index_of[str(item)] for item in rows_gold])
        else:
            gold.append([int(item) for item in rows_gold])
    if not queries:
        raise ValueError(f"{path} 没有读到任何检索样本")
    return queries, candidates, gold


def evaluate(model, args: argparse.Namespace) -> Dict[str, float]:
    """依次跑相似度与检索评测，返回指标字典。"""
    metrics: Dict[str, float] = {}
    for path in args.sts_files:
        pairs = read_sts_pairs(path)
        scores = cosine_similarities(model, pairs, batch_size=args.batch_size)
        name = os.path.splitext(os.path.basename(path))[0]
        metrics[f"{name}/spearman"] = spearman(scores, [score for _, _, score in pairs])
        metrics[f"{name}/pearson"] = pearson(scores, [score for _, _, score in pairs])
        metrics[f"{name}/n"] = float(len(pairs))
    if args.sts_files:
        keys = [key for key in metrics if key.endswith("/spearman")]
        metrics["spearman_mean"] = sum(metrics[key] for key in keys) / len(keys)
    if args.retrieval_file:
        queries, candidates, gold = read_retrieval(args.retrieval_file)
        metrics["retrieval/recall@k"] = recall_at_k(
            model, queries, candidates, gold, batch_size=args.batch_size, k=args.recall_k
        )
        metrics["retrieval/n_queries"] = float(len(queries))
    return metrics


def main() -> None:
    args = parse_args()
    torch.set_grad_enabled(False)
    device = resolve_device(args.device)
    model = load_encoder(args.model, device=args.device)
    metrics = evaluate(model, args)
    if not metrics:
        raise SystemExit("没有指定任何评测输入：至少给一个 --sts-files 或 --retrieval-file")
    print(f"设备 {device} · 模型 {args.model}")
    print(format_metrics(metrics))
    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, ensure_ascii=False, indent=2)
        print(f"指标已写入 {args.json_out}")


if __name__ == "__main__":
    main()
