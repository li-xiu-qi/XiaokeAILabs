"""把已有的 20news_emb.npz 转成索引实验所需的 float32 二进制向量。

输入应包含 train_emb、test_emb、train_labels、test_labels。脚本只做校验、
L2 归一化和可选复制，不重新下载语料或编码文本。真实 embedding 的几何结构
用于召回验证；复制只用于延迟标定，不应被解释为新的独立语料。

用法：
    python scripts/prepare_corpus_from_20news.py --src /path/to/20news_emb.npz
    python scripts/prepare_corpus_from_20news.py --src /path/to/20news_emb.npz --replicate 2
"""
import argparse
import json
import os
import time

import numpy as np

OUT_DIR = "corpus"


def normalized_rows(values, name):
    values = np.ascontiguousarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty 2D matrix, got {values.shape}")
    norms = np.linalg.norm(values, axis=1)
    if not np.isfinite(norms).all() or np.any(norms == 0):
        raise ValueError(f"{name} contains non-finite or zero vector norms")
    print(
        f"{name} norm before: min={norms.min():.3f} max={norms.max():.3f} "
        f"mean={norms.mean():.3f}",
        flush=True,
    )
    values /= norms[:, None]
    after = np.linalg.norm(values, axis=1)
    if not np.allclose(after, 1.0, atol=1e-5):
        raise ValueError(f"{name} normalization failed")
    return values


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="Input 20news_emb.npz")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--corpus", default="user-provided", help="Source corpus identifier, supplied by the user")
    ap.add_argument("--model", default="unknown", help="Embedding model and revision, supplied by the user")
    ap.add_argument(
        "--replicate",
        type=int,
        default=1,
        help="底库复制倍数，用于延迟标定。1 = 不复制（原始 11293 条）",
    )
    args = ap.parse_args()
    if args.replicate < 1:
        raise ValueError("replicate must be >= 1")

    print(f"loading {args.src} ...", flush=True)
    started = time.time()
    with np.load(args.src, allow_pickle=True) as data:
        required = {"train_emb", "test_emb", "train_labels", "test_labels"}
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"Missing arrays in {args.src}: {sorted(missing)}")
        train = normalized_rows(data["train_emb"], "train_emb")
        test = normalized_rows(data["test_emb"], "test_emb")
        train_labels = np.asarray(data["train_labels"])
        test_labels = np.asarray(data["test_labels"])
    print(f"loaded in {time.time() - started:.1f}s", flush=True)

    if train.shape[1] != test.shape[1]:
        raise ValueError(f"Dimension mismatch: train={train.shape}, test={test.shape}")
    if len(train_labels) != train.shape[0] or len(test_labels) != test.shape[0]:
        raise ValueError("Embedding and label counts do not match")
    print(f"train {train.shape}  test {test.shape}", flush=True)

    if args.replicate > 1:
        train = np.ascontiguousarray(np.tile(train, (args.replicate, 1)), dtype=np.float32)
        train_labels = np.tile(train_labels, args.replicate)
        print(f"replicated x{args.replicate} -> {train.shape}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    base_path = os.path.join(args.out_dir, "base_vectors.bin")
    query_path = os.path.join(args.out_dir, "query_vectors.bin")
    with open(base_path, "wb") as handle:
        handle.write(train.tobytes())
    with open(query_path, "wb") as handle:
        handle.write(test.tobytes())

    dim = int(train.shape[1])
    meta = {
        "source": os.path.basename(args.src),
        "corpus": args.corpus,
        "model": args.model,
        "dim": dim,
        "n_base": int(train.shape[0]),
        "n_query": int(test.shape[0]),
        "replicate": args.replicate,
        "normalized": True,
        "dtype": "float32",
        "labels": sorted(set(map(str, train_labels)).union(map(str, test_labels))),
        "label_counts_base": {
            str(key): int(value)
            for key, value in zip(*np.unique(train_labels, return_counts=True))
        },
    }
    meta_path = os.path.join(args.out_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)

    print("=" * 55, flush=True)
    print(f"base  : {base_path}  {os.path.getsize(base_path) / 1e6:.1f} MB")
    print(f"        {train.shape[0]} x {dim} raw float32")
    print(f"query : {query_path}  {os.path.getsize(query_path) / 1e6:.1f} MB")
    print(f"        {test.shape[0]} x {test.shape[1]}")
    print(f"meta  : {meta_path}")
    print(f"labels: {len(meta['labels'])} 类", flush=True)


if __name__ == "__main__":
    main()
