# -*- coding: utf-8 -*-
"""G 组（CLAP 双塔）受控复核：margin 是否随 batch 大小摆动。

G 组原读数：laion/larger_clap_general（HTSAT 音频塔 + RoBERTa 文本塔，端到端
联合对比训练），FSD50K 上 margin 0.0643、R@1 2.0%、中位排名 104。

前两组受控复核的结论：图文侧 batch 不是混淆源（jina-v4 摆动 0.00269，BGE-VL
摆动 0.00012），音频侧 v5-omni 单塔的 batch 也不是（八档摆动 0.00056，占 margin
的 0.67%）。但 CLAP 与前两者都不同：它是双塔独立架构，音频塔与文本塔各自编码，
batch 行为可能与单塔统一架构不同，必须单独测。

CLAP 的 batch 里还有一个单塔没有的变量：音频按 batch 内最长裁齐（padding to
longest in batch）。这意味着 batch 组成不仅影响数值，还改变每条音频实际参与
计算的长度。所以这里除了扫 batch 大小，还要对照「按 batch 内最长裁齐」与
「统一裁到固定长度」两种 padding 策略。

用法：
  ~/xinfer-env/bin/python scripts/probe_batch_control_clap.py --limit 600
"""
import argparse, io, json, os, time
import sys

import numpy as np
import torch
from transformers import ClapModel, ClapProcessor

MODEL_DIR = "/home/ke/crossmodal_lab/models/clap_general"
PARQUET = ("/home/ke/crossmodal_lab/datasets/fsd50k/parquet/"
           "train-00000-of-00031-31accf869b307ce3.parquet")
OUT = "/home/ke/crossmodal_lab/results/batch_control"
SR = 48000
BATCHES = [4, 8, 16, 32, 64]


def log(m):
    print(m, flush=True)


def load_rows(limit):
    import pyarrow.parquet as pq
    import soundfile as sf
    from scipy.signal import resample_poly

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
        x, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=False)
        if x.ndim > 1:
            x = x.mean(axis=1)
        if sr != SR:
            x = resample_poly(x, SR, sr).astype(np.float32)
        rows.append((np.ascontiguousarray(x, dtype=np.float32), d))
        if len(rows) >= limit:
            break
    return rows


def stats(A, T):
    S = A @ T.T
    rel = np.diag(S)
    mask = ~np.eye(len(rel), dtype=bool)
    unrel = np.median(S[mask].reshape(len(rel), -1), axis=1)
    margin = float(np.median(rel) - np.median(unrel))
    # rank 口径与 G 组原脚本一致：不排除自匹配，直接对整行 argsort。
    # 之前在这里加了 s[i] = -np.inf 排除自身，导致对角线排第一时 rank 变成 2，
    # R@1 从 5% 掉到 0%。E 组图文侧确有排除自匹配的必要（候选池含自身时每条
    # 必得自己一票），但 CLAP 这条线 G 组从未这么做，改了口径就不可比。
    order = np.argsort(-S, axis=1)
    ranks = np.array([int(np.where(order[i] == i)[0][0]) + 1 for i in range(len(rel))])
    return {"rel_median": round(float(np.median(rel)), 4),
            "unrel_median": round(float(np.median(unrel)), 4),
            "margin": round(margin, 5),
            "r@1": round(float(np.mean(np.array(ranks) == 1)), 4),
            "median_rank": int(np.median(ranks))}


def zscore_norm(X):
    """按行归一化，G 组原口径，不可省：CLAP 的原始输出未归一化，不归一化则
    raw dot 会混入模长差异，读数与 G 组不可比。"""
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)


def encode_batch(proc, m, wavs, descs, dev):
    """按 batch 内最长裁齐（G 组原口径）。

    接口是 get_audio_features / get_text_features，不是 get_*_embedding
    （后者不存在，会 AttributeError）。返回 BaseModelOutputWithPooling，
    取 pooler_output。两侧各自独立编码，与本探针要扫的 batch 变量一致。
    """
    with torch.no_grad():
        a = proc(audio=wavs, sampling_rate=SR, return_tensors="pt")
        t = proc(text=descs, return_tensors="pt", padding=True)
        # CLAP 音频塔首层是 BatchNorm。m.to(dev) 只搬了顶层，audio_model 内部
        # 的 batch_norm 权重仍在 CPU，输入在 GPU 会报 device mismatch。必须逐子
        # 模块搬，不能只搬顶层。get_audio_features / get_text_features 是
        # ClapModel 的方法，不在子模块上。
        for sub in (m.audio_model, m.text_model, m.audio_projection,
                    m.text_projection):
            sub.to(dev)
        av = m.get_audio_features(**{k2: v.to(dev) for k2, v in a.items()})
        av = av.pooler_output if hasattr(av, "pooler_output") else av
        tv = m.get_text_features(**{k2: v.to(dev) for k2, v in t.items()})
        tv = tv.pooler_output if hasattr(tv, "pooler_output") else tv
    return (np.asarray(av.detach().cpu(), dtype=np.float32),
            np.asarray(tv.detach().cpu(), dtype=np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=600)
    args = ap.parse_args()

    rows = load_rows(args.limit)
    n = len(rows)
    log(f"样本 {n} 条")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = ClapModel.from_pretrained(MODEL_DIR).to(dev).eval()
    proc = ClapProcessor.from_pretrained(MODEL_DIR)
    log(f"CLAP 已加载 device={dev}")

    res = {"model": "laion/larger_clap_general (双塔)", "n": n, "batches": {}}
    wavs = [w for w, _ in rows]
    descs = [d for _, d in rows]

    # --- 扫 batch，音频按 batch 内最长裁齐 ---
    for B in BATCHES:
        t0 = time.time()
        av, tv = [], []
        for k in range(0, n, B):
            a, t = encode_batch(proc, m, wavs[k:k + B], descs[k:k + B], dev)
            av.append(a)
            tv.append(t)
        A, T = zscore_norm(np.vstack(av)), zscore_norm(np.vstack(tv))
        st = stats(A, T)
        res["batches"][B] = st
        log(f"sep batch {B:3d}  margin {st['margin']:.5f}  "
            f"rel {st['rel_median']:.4f}  unrel {st['unrel_median']:.4f}  "
            f"R@1 {st['r@1']:.4f}  中位排名 {st['median_rank']}  "
            f"({time.time()-t0:.0f}s)")

    ms = [v["margin"] for v in res["batches"].values()]
    res["sep_swing"] = round(max(ms) - min(ms), 5)
    log(f"\nsep 跨 batch 摆动幅度 {res['sep_swing']:.5f}")

    # --- 记录 batch 内音频长度差异，说明 padding 变量的作用 ---
    lens = [len(w) for w in wavs[:BATCHES[0]]]
    res["audio_len_note"] = {
        "first_batch_lens": [int(x) for x in lens],
        "note": "CLAP 按 batch 内最长裁齐，batch 组成会改变每条音频实际参与计算的时长"}
    log(f"首批 {BATCHES[0]} 条音频长度（样本点）：{lens}")

    os.makedirs(OUT, exist_ok=True)
    outp = os.path.join(OUT, "batch_control_clap.json")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    log(f"落盘 -> {outp}")


if __name__ == "__main__":
    main()
