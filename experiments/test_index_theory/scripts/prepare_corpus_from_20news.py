"""
从已有的 20news_emb.npz 准备索引实验用的向量。

为什么不重新编码语料：test_vector_db_bench 里已经有一份
all-MiniLM-L6-v2 编码的 20 Newsgroups（11293 train + 600 test，384 维），
五库对比用的就是它。召回验证要的是真实 embedding 的几何结构，与语料规模
和语言无关，这份数据已经具备。重新下载 2GB 语料再 GPU 编码一小时是重复劳动。

需要做的处理只有一件：L2 归一化。原始向量的模长从 0.92 到 6.36 分布很散，
不归一化时内积不等于余弦相似度，检索结果会被模长主导而不是方向。

用法：
    python prepare_corpus_from_20news.py
"""
import argparse
import json
import os
import time

import numpy as np

SRC_NPZ = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "test_vector_db_bench", "data", "20news_emb.npz",
)
OUT_DIR = "corpus"
DIM = 384


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC_NPZ)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument(
        "--replicate",
        type=int,
        default=1,
        help="底库复制倍数，用于延迟标定。1 = 不复制（11293 条）",
    )
    args = ap.parse_args()

    print(f"loading {args.src} ...", flush=True)
    t0 = time.time()
    data = np.load(args.src, allow_pickle=True)
    train = np.ascontiguousarray(data["train_emb"], dtype=np.float32)
    test = np.ascontiguousarray(data["test_emb"], dtype=np.float32)
    train_labels = data["train_labels"]
    test_labels = data["test_labels"]
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    print(f"train {train.shape}  test {test.shape}", flush=True)

    # 原始模长分布：不归一化的话内积会被模长主导
    n_train = np.linalg.norm(train, axis=1)
    print(
        f"norm before: min={n_train.min():.3f} max={n_train.max():.3f} "
        f"mean={n_train.mean():.3f}",
        flush=True,
    )

    train /= n_train[:, None]
    test /= np.linalg.norm(test, axis=1)[:, None]
    print(
        f"norm after : {np.linalg.norm(train, axis=1).min():.6f} "
        f"(应全为 1.0)",
        flush=True,
    )

    if args.replicate > 1:
        # 复制不改变几何结构：每个向量的近邻集合不变，只是 n 变大。
        # 延迟标定只关心 n 与延迟的关系，这个操作是有效的。
        train = np.tile(train, (args.replicate, 1))
        train = np.ascontiguousarray(train, dtype=np.float32)
        print(f"replicated x{args.replicate} -> {train.shape}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    base_path = os.path.join(args.out_dir, "base_vectors.bin")
    query_path = os.path.join(args.out_dir, "query_vectors.bin")

    with open(base_path, "wb") as f:
        f.write(train.tobytes())
    with open(query_path, "wb") as f:
        f.write(test.tobytes())

    # 元数据只存标签，正文太长不入库
    meta = {
        "source": os.path.basename(args.src),
        "model": "all-MiniLM-L6-v2",
        "dim": DIM,
        "n_base": int(train.shape[0]),
        "n_query": int(test.shape[0]),
        "replicate": args.replicate,
        "normalized": True,
        "labels": sorted(set(map(str, train_labels)).union(map(str, test_labels))),
        "label_counts_base": {
            str(k): int(v) for k, v in zip(*np.unique(train_labels, return_counts=True))
        },
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("=" * 55, flush=True)
    print(f"base  : {base_path}  {os.path.getsize(base_path)/1e6:.1f} MB")
    print(f"        {train.shape[0]} x {train.shape[1]} raw float32")
    print(f"query : {query_path}  {os.path.getsize(query_path)/1e6:.1f} MB")
    print(f"        {test.shape[0]} x {test.shape[1]}")
    print(f"labels: {len(meta['labels'])} 类", flush=True)


if __name__ == "__main__":
    main()
