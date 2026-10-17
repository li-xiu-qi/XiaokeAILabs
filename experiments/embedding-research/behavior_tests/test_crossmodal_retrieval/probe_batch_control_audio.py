# -*- coding: utf-8 -*-
"""E 组音频侧受控复核：margin 是否随 batch 配置摆动。

E 组（v5-omni，FSD50K）原读数 margin 0.084、rel_median 0.2439、unrel_median 0.16。
图文侧的受控实验已确认：jina-v4 的 shuffle gap 跨五档 batch 摆动 0.00269，
BGE-VL 摆动 0.00012，batch 不是主要混淆源。但那是图文双塔模型，音频侧是单塔
统一架构（文本、图像、音频共用一个向量空间），编码路径不同，batch 行为可能
也不同，必须单独测。

v5-omni 的编码入口是 encode_query 与 encode_document 两个方法，E 组按 batch
分组循环调用。本脚本固定样本集，只改 batch 大小扫多档，同时做三种编码方式对照：

  sep   现在 E 组的做法：每批内先编全部音频，再编全部文本（两次独立调用）
  mix   音频与文本交错进同一次调用，两侧批量组成完全相同
  pair  [audio_i, desc_i] 作为两条一批，每对处在完全相同的批次上下文里

判据与图文侧一致：margin 跨 batch 档位的摆动幅度若接近或大于 margin 本身
（0.084），则该读数不可归因；若摆动幅度远小于 margin，则读数可用。

用法：
  ~/xinfer-env/bin/python scripts/probe_batch_control_audio.py --limit 600
"""
import argparse, io, json, os, time

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL = "jinaai/jina-embeddings-v5-omni-small"
PARQUET = ("/home/ke/crossmodal_lab/datasets/fsd50k/parquet/"
           "train-00000-of-00031-31accf869b307ce3.parquet")
OUT = "/home/ke/crossmodal_lab/results/batch_control"
BATCHES = [1, 2, 3, 4, 8, 16, 32, 64]


def batched(it, n):
    for i in range(0, len(it), n):
        yield it[i:i + n]


def load_rows(limit):
    import pyarrow.parquet as pq
    import soundfile as sf
    from scipy.signal import resample_poly

    def dec16k(b):
        x, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=False)
        if x.ndim > 1:
            x = x.mean(axis=1)
        if sr != 16000:
            x = resample_poly(x, 16000, sr).astype(np.float32)
        return np.ascontiguousarray(x, dtype=np.float32)

    t = pq.read_table(PARQUET)
    desc = t.column("description").to_pylist()
    aud = t.column("audio").to_pylist()
    rows = []
    for i in range(min(limit + 200, len(desc))):
        a = aud[i]
        b = a["bytes"] if isinstance(a, dict) else None
        if not b or b[:4] != b"RIFF":
            continue
        d = (desc[i] or "").strip()
        if not d:
            continue
        rows.append({"idx": i, "desc": d, "audio": dec16k(b)})
        if len(rows) >= limit:
            break
    return rows


def stats(A, T):
    """A 音频向量，T 文本向量，按 E 组口径算 margin 与检索读数。"""
    S = A @ T.T
    rel = np.diag(S)
    mask = ~np.eye(len(rel), dtype=bool)
    unrel = np.median(S[mask].reshape(len(rel), -1), axis=1)
    margin = float(np.median(rel) - np.median(unrel))
    # R@1：每个音频在其余文本里找对应描述
    ranks = []
    for i in range(len(rel)):
        s = S[i].copy()
        s[i] = -np.inf
        ranks.append(int(np.where(np.argsort(-s) == i)[0][0]) + 1)
    r1 = float(np.mean(np.array(ranks) == 1))
    return {"rel_median": round(float(np.median(rel)), 4),
            "unrel_median": round(float(np.median(unrel)), 4),
            "margin": round(margin, 5), "r@1": round(r1, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=600)
    args = ap.parse_args()

    rows = load_rows(args.limit)
    print(f"样本 {len(rows)} 条（已剔除非 RIFF 与空描述）", flush=True)

    m = SentenceTransformer(MODEL, trust_remote_code=True,
                            model_kwargs={"default_task": "retrieval"})
    print(f"模型已加载 device={m.device}", flush=True)

    res = {"model": MODEL, "n": len(rows), "batches": {}, "mix": {}, "pair": {}}
    audios = [r["audio"] for r in rows]
    descs = [r["desc"] for r in rows]

    # --- sep：每批内先编音频再编文本，扫 batch ---
    for bs in BATCHES:
        t0 = time.time()
        av, tv = [], []
        for ch in batched(list(range(len(rows))), bs):
            a = m.encode_query([audios[i] for i in ch])
            t = m.encode_document([descs[i] for i in ch])
            av.append(np.asarray(a, dtype=np.float32))
            tv.append(np.asarray(t, dtype=np.float32))
        A, T = np.vstack(av), np.vstack(tv)
        st = stats(A, T)
        res["batches"][bs] = st
        print(f"sep  batch {bs:3d}  margin {st['margin']:.5f}  "
              f"rel {st['rel_median']:.4f}  unrel {st['unrel_median']:.4f}  "
              f"R@1 {st['r@1']:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    # --- mix：音频与文本交错进同一次调用 ---
    # 注意：encode_query 与 encode_document 只是给 encode 设不同的 prompt_name
    # （query 对 document）。交错时必须交替调用这两个方法，不能都走 query，
    # 否则两侧 prompt 相同，读数与 E 组的 query/document 口径不可比。第一版就是
    # 这么错的，mix 与 pair 的 rel_median 掉到 0.11 而 sep 是 0.26，是 prompt 用错
    # 不是 batch 效应。
    inter = [x for pair in zip(audios, descs) for x in pair]
    for bs in (8, 32):
        out = []
        for ch in batched(inter, bs):
            # 交错序列的偶数位是音频、奇数位是文本
            e = [m.encode_query([ch[k]])[0] if k % 2 == 0
                 else m.encode_document([ch[k]])[0]
                 for k in range(len(ch))]
            out.append(np.asarray(e, dtype=np.float32))
        E = np.vstack(out)
        A, T = E[0::2], E[1::2]
        st = stats(A, T)
        res["mix"][f"mix_{bs}"] = st
        print(f"mix  batch {bs:3d}  margin {st['margin']:.5f}  "
              f"rel {st['rel_median']:.4f}  unrel {st['unrel_median']:.4f}",
              flush=True)

    # --- 小批次专项：batch 1/2/3，对应逐条编码的真实部署场景 ---
    # 现象：pair 那档 batch=1 的 margin 掉到 0.00314，而 batch>=4 稳定在 0.0838。
    # batch=1 与 batch=4 差 27 倍，需确认这是真实效应还是 pair 实现的问题。
    for bs in (1, 2, 3):
        t0 = time.time()
        av, tv = [], []
        for ch in batched(list(range(len(rows))), bs):
            a = m.encode_query([audios[i] for i in ch])
            t = m.encode_document([descs[i] for i in ch])
            av.append(np.asarray(a, dtype=np.float32))
            tv.append(np.asarray(t, dtype=np.float32))
        A, T = np.vstack(av), np.vstack(tv)
        st = stats(A, T)
        res.setdefault("small_batch", {})[bs] = st
        print(f"小批 batch {bs:3d}  margin {st['margin']:.5f}  "
              f"rel {st['rel_median']:.4f}  unrel {st['unrel_median']:.4f}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    # --- pair：每对两条一批，同样交替 prompt ---
    m2 = min(300, len(rows))
    out = []
    for i in range(m2):
        e = [m.encode_query([audios[i]])[0],
             m.encode_document([descs[i]])[0]]
        out.append(np.asarray(e, dtype=np.float32))
    E = np.vstack(out)
    A, T = E[0::2], E[1::2]
    st = stats(A[:m2], T[:m2])
    res["pair"]["pair_one_call"] = dict(st, n=m2)
    print(f"pair 成对 n={m2}  margin {st['margin']:.5f}  "
          f"rel {st['rel_median']:.4f}  unrel {st['unrel_median']:.4f}", flush=True)

    ms = [v["margin"] for v in res["batches"].values()]
    res["sep_swing"] = round(max(ms) - min(ms), 5)
    print(f"\nsep 跨 batch 摆动幅度 {res['sep_swing']:.5f}", flush=True)

    os.makedirs(OUT, exist_ok=True)
    outp = os.path.join(OUT, "batch_control_audio_v5omni.json")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"落盘 -> {outp}", flush=True)


if __name__ == "__main__":
    main()
