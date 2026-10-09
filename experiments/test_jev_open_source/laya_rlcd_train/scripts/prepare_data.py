"""下载 LocalLLaMA/typed-decisions 并转成 laya 训练用的 JSONL。

每行一个案例：{"state": ..., "questions": ..., "gold": ...}，与 notebook
cell 6（laya_finetune_typed_decisions_2xT4_kaggle.ipynb）的转换一致。

用法（DGX Spark 上，laya-rlcd-env）：
    HF_ENDPOINT=https://hf-mirror.com python prepare_data.py --out-dir data/
"""
import argparse
import json
import os
from datasets import load_dataset

DATASET_ID = "LocalLLaMA/typed-decisions"


def dump_split(split: str, out_dir: str) -> str:
    ds = load_dataset(DATASET_ID, "all", split=split)
    out_path = os.path.join(out_dir, f"typed_decisions_{split}.jsonl")
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for row in ds:
            rec = {
                "state": json.loads(row["state"]),
                "questions": json.loads(row["questions"]),
                "gold": json.loads(row["gold"]),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"{split}: {n} cases -> {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    train_path = dump_split("train", args.out_dir)
    test_path = dump_split("test", args.out_dir)
    print("done:", train_path, test_path)


if __name__ == "__main__":
    main()
