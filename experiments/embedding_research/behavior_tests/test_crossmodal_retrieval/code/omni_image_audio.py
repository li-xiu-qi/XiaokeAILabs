#!/usr/bin/env python
"""F 组：用 jina-embeddings-v5-omni-small 同模型内测图文与音频，闭合折扣比例。

动因：E 组音频侧换了模型，缺同源同模态基线，折扣比例的分母缺失，无法回答
「音频的跨模态损失比图文大还是小」。v5-omni 是单塔统一架构，图文音共用一个
向量空间，可以在同一模型内把两侧都测，分母与分子同源，折扣比例才可比。

三个读数：
  I2T  图对文本（跨模态），与 A 组同构，query=图、document=caption
  T2T  文本对文本（同模态），同图两条不同 caption 对跨图 caption
  A2T  音频对文本（跨模态），与 E 组同构，query=音频、document=描述
折扣比例 = 跨模态 margin / 同模态 margin，I2T 与 A2T 共用 T2T 作分母。

用法：
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ~/xinfer-env/bin/python omni_image_audio.py \
      [--nimg 500] [--naud 500] [--batch 8] [--out DIR]
"""
import argparse, csv, io, json, os, time, warnings
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from scipy.signal import resample_poly

warnings.filterwarnings("ignore")
from sentence_transformers import SentenceTransformer

MODEL = "jinaai/jina-embeddings-v5-omni-small"
FLICKR_CSV = "/home/ke/crossmodal_lab/datasets/flickr30k/flickr_annotations_30k.csv"
FLICKR_IMG = "/home/ke/crossmodal_lab/datasets/flickr30k/images/flickr30k-images"
FSD_PARQUET = ("/home/ke/crossmodal_lab/datasets/fsd50k/parquet/"
               "train-00000-of-00031-31accf869b307ce3.parquet")


def log(m):
    print(m, flush=True)


def dec16k(b):
    """WAV bytes -> 16k mono float32。44.1k 降 16k 用抗混叠重采样。"""
    x, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != 16000:
        x = resample_poly(x, 16000, sr).astype(np.float32)
    return np.ascontiguousarray(x, dtype=np.float32)


def load_flickr(nimg):
    """[(filename, [captions])]，按文件名排序取前 nimg 张，保证与 A 组同批。"""
    rows = []
    with open(FLICKR_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                caps = json.loads(row["raw"])
            except Exception:
                continue
            caps = [c.strip() for c in caps if c and c.strip()]
            if caps:
                rows.append((row["filename"], caps))
    rows.sort()
    return rows[:nimg]


def load_audio(naud):
    t = pq.read_table(FSD_PARQUET)
    desc = t.column("description").to_pylist()
    aud = t.column("audio").to_pylist()
    out = []
    for i in range(min(naud, len(desc))):
        a = aud[i]
        b = a["bytes"] if isinstance(a, dict) else None
        if not b or b[:4] != b"RIFF":
            continue
        d = (desc[i] or "").strip()
        if d:
            out.append({"idx": i, "desc": d, "wav": dec16k(b)})
    return out


def batched(it, n):
    for i in range(0, len(it), n):
        yield it[i:i + n]


def cosmat(A, B):
    """行归一化后点积。模型输出已 L2 归一化，这里再归一一次防数值漂移。"""
    A = A / np.clip(np.linalg.norm(A, axis=1, keepdims=True), 1e-12, None)
    B = B / np.clip(np.linalg.norm(B, axis=1, keepdims=True), 1e-12, None)
    return (A @ B.T).astype(np.float32)


def ranks_of(S):
    """每行对角线元素在该行的排名（1 起）。"""
    n = S.shape[0]
    out = np.empty(n, dtype=np.int64)
    for i in range(n):
        order = np.argsort(-S[i])
        out[i] = int(np.where(order == i)[0][0]) + 1
    return out


def retriev(S, label):
    rk = ranks_of(S)
    n = S.shape[0]
    diag = np.diag(S)
    mask = ~np.eye(n, dtype=bool)
    unrel = np.median(S[mask].reshape(n, -1), axis=1)
    return {
        "label": label, "n": int(n),
        "rel_median": round(float(np.median(diag)), 4),
        "unrel_median": round(float(np.median(unrel)), 4),
        "margin": round(float(np.median(diag) - np.median(unrel)), 4),
        "r@1": round(float((rk == 1).mean()), 4),
        "r@5": round(float((rk <= 5).mean()), 4),
        "r@10": round(float((rk <= 10).mean()), 4),
        "median_rank": float(np.median(rk)),
        "random_r1": round(1.0 / n, 6),
    }


def t2t(m, caps_per_img, batch):
    """同模态：每图取两条 caption 作相关，跨图作无关。query/document 同侧。"""
    q_idx, d_idx = [], []
    for gi, caps in enumerate(caps_per_img):
        if len(caps) >= 2:
            q_idx.append((gi, 0))
            d_idx.append((gi, 1))
    if not q_idx:
        return None
    log(f"T2T 样本 {len(q_idx)} 对")
    qs = [caps_per_img[g][i] for g, i in q_idx]
    ds = [caps_per_img[g][i] for g, i in d_idx]
    Q, D = [], []
    for c in batched(qs, batch):
        Q.append(np.asarray(m.encode_query(c), dtype=np.float32))
    for c in batched(ds, batch):
        D.append(np.asarray(m.encode_document(c), dtype=np.float32))
    Q, D = np.vstack(Q), np.vstack(D)
    S = cosmat(Q, D)
    base = retriev(S, "T2T")
    base["pairs"] = "同图 caption0 对 caption1，query/document 严格分侧"
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nimg", type=int, default=500)
    ap.add_argument("--naud", type=int, default=500)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default="/home/ke/crossmodal_lab/results/jina-embeddings-v5-omni-small")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    m = SentenceTransformer(MODEL, trust_remote_code=True,
                            model_kwargs={"default_task": "retrieval"})
    log(f"模型已加载 device={m.device}")

    out = {"model": MODEL, "task": "retrieval", "elapsed_sec": None}

    # ---------- I2T：图对文本 ----------
    rows = load_flickr(args.nimg)
    log(f"Flickr30k {len(rows)} 张图")
    paths = [os.path.join(FLICKR_IMG, fn) for fn, _ in rows]
    first_caps = [caps[0] for _, caps in rows]
    all_caps = [caps for _, caps in rows]

    IE, TE = [], []
    for bi, ch in enumerate(batched(paths, args.batch)):
        IE.append(np.asarray(m.encode_query(ch), dtype=np.float32))
        if (bi + 1) % 10 == 0:
            log(f"  图像 {args.batch*(bi+1)}/{len(paths)}")
    for ch in batched(first_caps, 32):
        TE.append(np.asarray(m.encode_document(ch), dtype=np.float32))
    IE, TE = np.vstack(IE), np.vstack(TE)
    Si = cosmat(IE, TE)
    out["I2T"] = retriev(Si, "I2T")
    log("I2T " + json.dumps(out["I2T"], ensure_ascii=False))
    np.save(os.path.join(args.out, "f_i2t_simmat.npy"), Si)

    # ---------- T2T：文本对文本，作折扣比例的分母 ----------
    out["T2T"] = t2t(m, all_caps, args.batch)
    log("T2T " + json.dumps(out["T2T"], ensure_ascii=False))

    # ---------- A2T：音频对文本 ----------
    auds = load_audio(args.naud)
    log(f"FSD50K {len(auds)} 条音频")
    AE, AD = [], []
    for ch in batched(auds, args.batch):
        AE.append(np.asarray(m.encode_query([r["wav"] for r in ch]), dtype=np.float32))
    for ch in batched(auds, 32):
        AD.append(np.asarray(m.encode_document([r["desc"] for r in ch]), dtype=np.float32))
    AE, AD = np.vstack(AE), np.vstack(AD)
    Sa = cosmat(AE, AD)
    out["A2T"] = retriev(Sa, "A2T")
    log("A2T " + json.dumps(out["A2T"], ensure_ascii=False))
    np.save(os.path.join(args.out, "f_a2t_simmat.npy"), Sa)

    # ---------- 折扣比例，分母共用 T2T ----------
    den = out["T2T"]["margin"]
    out["discount"] = {
        "denominator": "T2T margin",
        "T2T_margin": den,
        "I2T_over_T2T": round(out["I2T"]["margin"] / den, 4) if den else None,
        "A2T_over_T2T": round(out["A2T"]["margin"] / den, 4) if den else None,
        "note": "分子分母同模型同侧编码，是本次能报折扣比例的前提；仍不可与 A 组两模型的折扣比例互比，因为那两侧分母是各自模型的 T2T",
    }

    out["elapsed_sec"] = round(time.time() - t0, 1)
    with open(os.path.join(args.out, "f_omni_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log(json.dumps(out, ensure_ascii=False, indent=2))
    log("DONE")


if __name__ == "__main__":
    main()
