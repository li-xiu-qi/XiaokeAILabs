# -*- coding: utf-8 -*-
"""
共享语料加载模块

从 wikipedia 20231101.zh 的 parquet 分片读取文章，构造带 ground truth 段落边界的
评测样本。Wikipedia 文章的 `text` 字段用 `\n\n` 分段，天然段落边界即 ground truth。

输出结构：
    docs: list[Doc]，每个 Doc 有 text / gold_boundaries（段落起始句索引）/ n_sentences

采样策略：文章长度在 [min_chars, max_chars] 之间，保证样本可比。
"""
import os
import glob
import json
from dataclasses import dataclass, field, asdict

import numpy as np


@dataclass
class Doc:
    doc_id: int
    title: str
    text: str
    gold_boundaries: list = field(default_factory=list)   # 段落起始句索引（不含 0）
    n_sentences: int = 0
    n_paragraphs: int = 0


def load_wiki_parquet(parquet_paths, min_chars=200, max_chars=4000, max_docs=None,
                      seed=42):
    """从 wikipedia parquet 读取并过滤文章。

    Args:
        parquet_paths: parquet 文件路径列表
        min_chars / max_chars: 文章长度过滤区间（字符数）
        max_docs: 最多取多少篇，None 表示全取
        seed: 采样随机种子

    Returns:
        list[Doc]，按 doc_id 排序
    """
    import pyarrow.parquet as pq

    rng = np.random.default_rng(seed)
    docs = []
    for path in sorted(parquet_paths):
        if max_docs and len(docs) >= max_docs:
            break
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=500,
                                     columns=["id", "title", "text"]):
            rows = batch.to_pylist()
            for r in rows:
                text = (r.get("text") or "").strip()
                if not (min_chars <= len(text) <= max_chars):
                    continue
                # 至少 3 段，否则没有可评测的边界
                if text.count("\n\n") < 2:
                    continue
                docs.append((r["id"], r.get("title") or "", text))
            if max_docs and len(docs) >= max_docs * 2:
                break

    if max_docs and len(docs) > max_docs:
        idx = rng.choice(len(docs), size=max_docs, replace=False)
        docs = [docs[i] for i in sorted(idx)]

    out = []
    for i, (wid, title, text) in enumerate(docs):
        paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        out.append(Doc(doc_id=i, title=title, text="\n\n".join(paras),
                       n_paragraphs=len(paras)))
    return out


def load_corpus(data_dir, n_docs=300, seed=42, cache_path=None):
    """加载评测语料，优先用缓存。

    缓存文件格式：jsonl，每行一个 Doc 的 dict。
    """
    if cache_path and os.path.exists(cache_path):
        docs = []
        with open(cache_path, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                docs.append(Doc(**d))
        if n_docs and len(docs) > n_docs:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(docs), size=n_docs, replace=False)
            docs = [docs[i] for i in sorted(idx)]
        return docs

    paths = glob.glob(os.path.join(data_dir, "20231101.zh", "*.parquet"))
    if not paths:
        raise FileNotFoundError(f"未找到 wikipedia parquet: {data_dir}")
    docs = load_wiki_parquet(paths, max_docs=n_docs, seed=seed)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            for d in docs:
                f.write(json.dumps(asdict(d), ensure_ascii=False) + "\n")
    return docs


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/wiki_zh"
    cache = sys.argv[2] if len(sys.argv) > 2 else "corpus/wiki_eval_300.jsonl"
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    ds = load_corpus(data_dir, n_docs=n, cache_path=cache)
    chars = [len(d.text) for d in ds]
    paras = [d.n_paragraphs for d in ds]
    print(f"docs={len(ds)} chars mean={np.mean(chars):.0f} "
          f"median={np.median(chars):.0f} paragraphs mean={np.mean(paras):.1f}")
    print(f"cache -> {cache}")
