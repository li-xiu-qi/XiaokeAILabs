#!/usr/bin/env python
"""音频侧跨模态基线：音频对文本描述，对标图文 A 组口径。

素材 FSD50K 一个 parquet 分片。查询侧是音频，候选侧是 description 文本，
与 A 组「图检索文本」同构（query=媒体，document=文本）。

只报跨模态 margin 与检索指标，不报折扣比例。原因：FSD50K 每条只有一条
description，没有「同音频多条不同表述」，构造不出与 Flickr30k 同源的同模态
基线。折扣比例留空并在报告里写明，不拿不同构的数据硬凑。

用法：
  ~/xinfer-env/bin/python audio_baseline.py [--limit N] [--batch 8] [--out DIR]
"""
import argparse, io, json, os, sys, time, warnings
import io
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

warnings.filterwarnings("ignore")
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer

MODEL = "jinaai/jina-embeddings-v5-omni-small"
PARQUET = ("/home/ke/crossmodal_lab/datasets/fsd50k/parquet/"
           "train-00000-of-00031-31accf869b307ce3.parquet")


def dec16k(b):
    """WAV bytes -> 16k mono float32。44.1k 降 16k 用抗混叠重采样。"""
    x, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != 16000:
        x = resample_poly(x, 16000, sr).astype(np.float32)
    return np.ascontiguousarray(x, dtype=np.float32)


def load_rows(limit):
    t = pq.read_table(PARQUET)
    desc = t.column("description").to_pylist()
    aud = t.column("audio").to_pylist()
    title = t.column("title").to_pylist()
    rows = []
    for i in range(min(limit, len(desc))):
        a = aud[i]
        b = a["bytes"] if isinstance(a, dict) else None
        if not b or b[:4] != b"RIFF":
            continue
        d = (desc[i] or "").strip()
        if not d:
            continue
        rows.append({"idx": i, "desc": d, "audio": dec16k(b), "title": title[i]})
    return rows


def batched(it, n):
    for i in range(0, len(it), n):
        yield it[i:i + n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=1322)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default="/home/ke/crossmodal_lab/results/jina-embeddings-v5-omni-small")
    ap.add_argument("--pool", type=int, default=0, help="切片候选池大小，0=全量；用于与图文 500 池对齐")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    rows = load_rows(args.limit)
    print(f"可用样本 {len(rows)} 条（已剔除非 RIFF 与空描述）", flush=True)

    m = SentenceTransformer(MODEL, trust_remote_code=True,
                            model_kwargs={"default_task": "retrieval"})
    print(f"模型已加载 device={m.device}", flush=True)

    # 音频作 query，文本作 document，对齐 A 组「图检索文本」
    aud_vecs, txt_vecs = [], []
    for bi, chunk in enumerate(batched(rows, args.batch)):
        av = m.encode_query([r["audio"] for r in chunk])
        tv = m.encode_document([r["desc"] for r in chunk])
        aud_vecs.append(np.asarray(av, dtype=np.float32))
        txt_vecs.append(np.asarray(tv, dtype=np.float32))
        if (bi + 1) % 5 == 0:
            el = time.time() - t0
            print(f"  {bi+1} 批 / {len(chunk)*(bi+1)} 条, 已用 {el:.0f}s", flush=True)
    A = np.vstack(aud_vecs)
    T = np.vstack(txt_vecs)
    print(f"音频向量 {A.shape}, 文本向量 {T.shape}", flush=True)

    # 全量相似度矩阵：行=音频 query，列=文本 document
    S = A @ T.T
    rel = np.diag(S)
    # 无关组：每个音频对其余所有文本，取该行中位数
    mask = ~np.eye(len(rows), dtype=bool)
    unrel = np.median(S[mask].reshape(len(rows), -1), axis=1)

    # 检索：每个音频在自己以外的 1321 条文本里找对应描述
    n = len(rows)
    np.save(os.path.join(args.out, "e_audio_simmat.npy"), S)
    # 切片：只用前 pool 条构造候选池，R@1 才与图文的 500 池可比
    if args.pool and args.pool < n:
        Sp = S[:args.pool, :args.pool]
        order = np.argsort(-Sp, axis=1)
        ranks = np.array([int(np.where(order[i] == i)[0][0]) for i in range(args.pool)])
        n_pool = args.pool
    else:
        order = np.argsort(-S, axis=1)
        n_pool = n
    r1 = float((ranks == 0).mean())
    r5 = float((ranks < 5).mean())
    r10 = float((ranks < 10).mean())

    # 按描述长度分层：短（<=4 词，多为标签式）与长
    wl = np.array([len(r["desc"].split()) for r in rows])
    short, long = wl <= 4, wl > 4

    def sub(m_):
        idx = np.where(m_ & (np.arange(len(rows)) < n_pool))[0]
        if len(idx) < 20:
            return None
        r_ = ranks[idx]
        return {
            "n": int(len(idx)),
            "rel_median": round(float(np.median(rel[idx])), 4),
            "unrel_median": round(float(np.median(unrel[idx])), 4),
            "margin": round(float(np.median(rel[idx]) - np.median(unrel[idx])), 4),
            "r@1": round(float((r_ == 0).mean()), 4),
            "r@5": round(float((r_ < 5).mean()), 4),
            "r@10": round(float((r_ < 10).mean()), 4),
            "median_rank": int(np.median(r_)),
        }

    summary = {
        "model": MODEL,
        "task": "retrieval",
        "dataset": "FSD50K train-00000",
        "source": PARQUET,
        "n": int(n),
        "random_baseline_r1": round(1.0 / n, 6),
        "overall": {
            "rel_median": round(float(np.median(rel)), 4),
            "unrel_median": round(float(np.median(unrel)), 4),
            "margin": round(float(np.median(rel) - np.median(unrel)), 4),
            "r@1": round(r1, 4), "r@5": round(r5, 4), "r@10": round(r10, 4),
            "median_rank": int(np.median(ranks)),
        },
        "by_desc_len": {"short_le4words": sub(short), "long_gt4words": sub(long)},
        "notes": [
            "query=音频, document=文本描述，对齐图文 A 组的图检索文本",
            "无关组为每个音频对其余全部文本的中位数，样本量对等",
            "不报折扣比例：FSD50K 每条仅一条描述，无同源同模态基线",
            "description 有详略（标签式与句子式），故按词数分层",
        ],
        "elapsed_sec": round(time.time() - t0, 1),
    }

    with open(os.path.join(args.out, "e_audio_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 逐样本 csv
    import csv
    with open(os.path.join(args.out, "e_audio_retrieval.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "title", "desc", "desc_words", "rank", "sim_rel", "sim_unrel"])
        for i, r in enumerate(rows[:n_pool]):
            w.writerow([r["idx"], r["title"], r["desc"], int(wl[i]), int(ranks[i]),
                        round(float(rel[i]), 4), round(float(unrel[i]), 4)])

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
