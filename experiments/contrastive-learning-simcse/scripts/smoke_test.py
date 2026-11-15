# -*- coding: utf-8 -*-
"""
冒烟测试：用小样本合成数据把训练与评测脚本跑通一遍。

目的是验证脚本本身可运行、指标可计算，不是验证 SimCSE 的效果。
合成数据是模板拼的随机句子，与真实语料无关；效果结论必须用真实数据重跑。

用法示例：
    python smoke_test.py --model hf-internal-testing/tiny-random-bert
    python smoke_test.py --model /path/to/local/small-model
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))

TOPICS = [
    "向量检索",
    "注意力机制",
    "梯度下降",
    "梯度裁剪",
    "余弦相似度",
    "语义空间",
    "负例挖掘",
    "批内负例",
]
VERBS = ["优化", "约束", "度量", "逼近", "对齐", "均匀化"]
OBJECTS = ["表征几何", "损失曲面", "超球面", "余弦分布", "排序目标", "表示流形"]
CONNECTIVES = ["因为", "所以", "同时", "然而", "此外"]


def make_sentence(rng: random.Random) -> str:
    """拼一句语法通顺但语义随机的中文短句。"""
    pattern = rng.choice(
        [
            "{topic}通过{verb}来{verb2}{obj}",
            "{connective}{topic}的{obj}需要{verb}",
            "如果{obj}发生{verb}，那么{topic}就会{verb2}{obj2}",
        ]
    )
    return pattern.format(
        topic=rng.choice(TOPICS),
        verb=rng.choice(VERBS),
        verb2=rng.choice(VERBS),
        obj=rng.choice(OBJECTS),
        obj2=rng.choice(OBJECTS),
        connective=rng.choice(CONNECTIVES),
    )


def build_corpus(rng: random.Random, count: int) -> tuple:
    """生成句子、三元组、相似度对与检索样本。"""
    sentences = [make_sentence(rng) for _ in range(count)]
    triplets = []
    for _ in range(count):
        anchor = make_sentence(rng)
        positive = anchor.replace(rng.choice(TOPICS), rng.choice(TOPICS))
        negative = make_sentence(rng)
        triplets.append({"anchor": anchor, "positive": positive, "negative": negative})
    pairs = []
    for _ in range(count):
        left = make_sentence(rng)
        right = left if rng.random() < 0.5 else make_sentence(rng)
        pairs.append(
            {
                "sentence1": left,
                "sentence2": right,
                "score": 5.0 if left == right else rng.uniform(0, 2),
            }
        )
    candidates = sentences[: max(8, count // 2)]
    retrieval = []
    for index, candidate in enumerate(candidates[:5]):
        retrieval.append({"query": candidate, "candidates": candidates, "gold": [index]})
    return sentences, triplets, pairs, retrieval


def write_jsonl(path: str, rows: list) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run(command: list) -> None:
    """运行子进程，失败时把输出打出来并中止测试。"""
    print("+", " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=HERE, capture_output=True, text=True)
    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr, file=sys.stderr)
        raise SystemExit(f"命令失败：{' '.join(command)}")


def read_metrics(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="SimCSE 套件冒烟测试")
    parser.add_argument("--model", default="hf-internal-testing/tiny-random-bert")
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--keep", action="store_true", help="保留临时目录便于排查")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    sentences, triplets, pairs, retrieval = build_corpus(rng, args.samples)
    workdir = tempfile.mkdtemp(prefix="simcse-smoke-")
    print(f"临时目录 {workdir}")
    try:
        write_jsonl(os.path.join(workdir, "sentences.jsonl"), [{"sentence": s} for s in sentences])
        write_jsonl(os.path.join(workdir, "triplets.jsonl"), triplets)
        write_jsonl(os.path.join(workdir, "sts.jsonl"), pairs)
        write_jsonl(os.path.join(workdir, "retrieval.jsonl"), retrieval)

        common = ["--model", args.model, "--device", "cpu", "--epochs", "1", "--log-every", "1"]
        run(
            [
                sys.executable,
                os.path.join(HERE, "train_unsup_simcse.py"),
                "--data",
                os.path.join(workdir, "sentences.jsonl"),
                "--out",
                os.path.join(workdir, "unsup"),
                "--batch-size",
                "4",
                "--limit",
                "12",
            ]
            + common
        )
        run(
            [
                sys.executable,
                os.path.join(HERE, "train_sup_simcse.py"),
                "--data",
                os.path.join(workdir, "triplets.jsonl"),
                "--out",
                os.path.join(workdir, "sup"),
                "--batch-size",
                "4",
                "--limit",
                "12",
            ]
            + common
        )
        for tag in ("unsup", "sup"):
            run(
                [
                    sys.executable,
                    os.path.join(HERE, "evaluate_sts.py"),
                    "--model",
                    os.path.join(workdir, tag),
                    "--sts-files",
                    os.path.join(workdir, "sts.jsonl"),
                    "--retrieval-file",
                    os.path.join(workdir, "retrieval.jsonl"),
                    "--json-out",
                    os.path.join(workdir, f"{tag}-eval.json"),
                    "--device",
                    "cpu",
                ]
            )

        for tag in ("unsup", "sup"):
            metrics = read_metrics(os.path.join(workdir, f"{tag}-eval.json"))
            assert "spearman_mean" in metrics, f"{tag} 缺少 spearman_mean"
            assert "retrieval/recall@k" in metrics, f"{tag} 缺少检索指标"
        print("冒烟测试通过：训练、保存、加载、评测链路可用")
        print("注意：指标由随机模板数据算出，不代表 SimCSE 效果")
    finally:
        if args.keep:
            print(f"临时目录已保留：{workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
