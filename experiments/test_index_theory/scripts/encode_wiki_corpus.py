"""用 all-MiniLM-L6-v2 编码中文维基百科，产出百万级向量语料。

为什么用维基而不是短文本分类数据集：要凑到百万条真实文本，中文维基是唯一
能干净拿到百万级条目的公开来源，且条目为连续书面语，与 20news 的短帖不同
构但都是真实自然语言的几何结构。AG News、DBpedia、Yelp 在 HF 上的标准划分
每个只有 7.7 万条，拼不到百万（已核实）。

预算控制：只取前 100 万条，原始 parquet 1.39 GB，编码后 float32 向量
100 万 × 384 × 4 = 1.53 GB。原始 parquet 在编码完成后删除。

用法：
    .venv-embed/Scripts/python.exe scripts/encode_wiki_corpus.py --limit 2000   # 试跑
    .venv-embed/Scripts/python.exe scripts/encode_wiki_corpus.py                # 全量
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
WIKI_DIR = ROOT / "data" / "wiki_zh" / "20231101.zh"
MODEL_DIR = ROOT / "models" / "all-MiniLM-L6-v2"
OUT_DIR = ROOT / "corpus"
DIM = 384

# 维基条目长短差异极大（中位约 3700 字符，均值约 6700），统一截断到前 600 字。
# 理由：MiniLM 的 max_seq_length 是 256 token，约 400 到 800 个汉字，超出部分
# 本就被截断；而条目尾部大量是参考文献、外部链接、导航模板，噪声高于信息。
MAX_CHARS = 600
MIN_CHARS = 50


def load_texts(limit):
    """顺序读取 parquet，返回 (texts, titles)。跳过过短条目。"""
    import pyarrow.parquet as pq

    files = sorted(WIKI_DIR.glob("train-*.parquet"))
    texts, titles = [], []
    for f in files:
        if len(texts) >= limit:
            break
        pf = pq.ParquetFile(f)
        for rg in range(pf.num_row_groups):
            if len(texts) >= limit:
                break
            t = pf.read_row_group(rg, columns=["title", "text"])
            for title, text in zip(t.column("title").to_pylist(),
                                   t.column("text").to_pylist()):
                if len(texts) >= limit:
                    break
                body = " ".join(str(text).split())
                if len(body) < MIN_CHARS:
                    continue
                texts.append(body[:MAX_CHARS])
                titles.append(str(title))
    return texts, titles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=1_000_000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--out", default=str(OUT_DIR / "wiki_zh_1m.npz"))
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    t0 = time.perf_counter()
    texts, titles = load_texts(args.limit)
    t_load = time.perf_counter() - t0
    print(f"载入 {len(texts):,} 条，耗时 {t_load:.1f}s")

    model = SentenceTransformer(str(MODEL_DIR))
    model.max_seq_length = 256
    if device == "cuda":
        model = model.half()

    t0 = time.perf_counter()
    emb = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    t_enc = time.perf_counter() - t0
    print(f"编码 {len(emb):,} 条，耗时 {t_enc:.1f}s，{len(emb)/t_enc:.0f} 条/秒")
    print(f"维度 {emb.shape[1]}，dtype {emb.dtype}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        emb=emb.astype("float32"),
        titles=np.array(titles, dtype=object),
    )
    size_gb = out.stat().st_size / 2 ** 30
    print(f"已写入 {out}（{size_gb:.2f} GB）")

    meta = {
        "source": "wikimedia/wikipedia 20231101.zh",
        "model": "all-MiniLM-L6-v2",
        "dim": int(emb.shape[1]),
        "n": int(emb.shape[0]),
        "max_chars": MAX_CHARS,
        "min_chars": MIN_CHARS,
        "max_seq_length": 256,
        "normalized": True,
        "dtype": "float32",
        "device": device,
        "batch_size": args.batch_size,
        "load_s": round(t_load, 1),
        "encode_s": round(t_enc, 1),
        "docs_per_s": round(len(emb) / t_enc, 1),
        "output_gb": round(size_gb, 3),
    }
    meta_path = out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"元数据 {meta_path}")


if __name__ == "__main__":
    main()
