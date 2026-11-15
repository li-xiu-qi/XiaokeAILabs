"""下载 LocalLLaMA/typed-decisions 并转成 laya 训练用的 JSONL。

每行一个案例：{"state": ..., "questions": ..., "gold": ...}，与 notebook
cell 6（laya_finetune_typed_decisions_2xT4_kaggle.ipynb）的转换一致。

用法（DGX Spark 上，laya-rlcd-env）：
    HF_ENDPOINT=https://hf-mirror.com python prepare_data.py --out-dir data/
"""
import argparse
from contextlib import ExitStack
import json
import os
from datasets import load_dataset

DATASET_ID = "LocalLLaMA/typed-decisions"


def dump_split(split: str, out_path: str, f, revision: str | None) -> str:
    ds = load_dataset(DATASET_ID, "all", split=split, revision=revision)
    n = 0
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
    ap.add_argument("--revision", default=None,
                    help="Hugging Face dataset commit SHA; omit only for the current revision")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    paths = {split: os.path.join(args.out_dir, f"typed_decisions_{split}.jsonl")
             for split in ("train", "test")}
    existing = [path for path in paths.values() if os.path.lexists(path)]
    if existing:
        raise FileExistsError(f"目标已存在，请使用新目录: {', '.join(existing)}")
    # 两个目标先排他创建，均成功后才开始获取数据。
    with ExitStack() as stack:
        files = {split: stack.enter_context(open(path, "x", encoding="utf-8"))
                 for split, path in paths.items()}
        train_path = dump_split("train", paths["train"], files["train"], args.revision)
        test_path = dump_split("test", paths["test"], files["test"], args.revision)
    print("done:", train_path, test_path)


if __name__ == "__main__":
    main()
