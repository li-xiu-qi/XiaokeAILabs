#!/usr/bin/env python
"""G 组：用 CLAP 测同一套音频对文本，判定「模态难」还是「v5-omni 音频塔弱」。

动因：F 组得到 v5-omni 内音频折扣 1.7%、图文 66.5%，差 40 倍。但只有一个音频模型，
无法区分「音频模态本身难对齐」与「这个模型的音频塔弱」。CLAP 是双塔独立架构
（HTSAT 音频编码器 + RoBERTa 文本编码器，直接对比学习），与 v5-omni 的单塔统一
（Whisper 编码器投影进语言模型）是两条路线，测同一套数据正好分离两个解释。

走 transformers 的 ClapModel，不用 laion_clap 包（它的 load_ckpt 只认自家 .pt 格式）。

用法：
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ~/xinfer-env/bin/python clap_audio.py \
      [--limit 500] [--out DIR]
"""
import argparse, io, json, os, time, warnings
import numpy as np

warnings.filterwarnings("ignore")
import pyarrow.parquet as pq
import soundfile as sf

MODEL_DIR = "/home/ke/crossmodal_lab/models/clap_general"
FSD = ("/home/ke/crossmodal_lab/datasets/fsd50k/parquet/"
       "train-00000-of-00031-31accf869b307ce3.parquet")
SR = 48000  # CLAP 官方 48k 梅尔


def load_rows(limit):
    t = pq.read_table(FSD)
    desc = t.column("description").to_pylist()
    aud = t.column("audio").to_pylist()
    out = []
    for i in range(min(limit, len(desc))):
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
        out.append((np.ascontiguousarray(x, dtype=np.float32), sr, d))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--out", default="/home/ke/crossmodal_lab/results/clap_general")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    log = lambda m: print(m, flush=True)

    import torch
    from transformers import ClapModel, ClapProcessor
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = ClapModel.from_pretrained(MODEL_DIR).to(dev).eval()
    proc = ClapProcessor.from_pretrained(MODEL_DIR)
    log(f"CLAP 已加载 device={dev}")

    rows = load_rows(args.limit)
    n = len(rows)
    log(f"样本 {n} 条")

    ae, te = [], []
    B = 16
    with torch.no_grad():
        for k in range(0, n, B):
            ch = rows[k:k + B]
            # 统一重采样到 48k，按 batch 内最长裁齐
            wavs = []
            for w, sr, _ in ch:
                if sr != SR:
                    from scipy.signal import resample_poly
                    w = resample_poly(w, SR, sr).astype(np.float32)
                wavs.append(w)
            a = proc(audio=wavs, sampling_rate=SR, return_tensors="pt")
            av = m.get_audio_features(**{k2: v.to(dev) for k2, v in a.items()})
            av = av.pooler_output if hasattr(av, "pooler_output") else av
            ae.append(np.asarray(av.detach().cpu(), dtype=np.float32))
            if (k // B + 1) % 5 == 0:
                log(f"  音频 {k+len(ch)}/{n} 已用 {time.time()-t0:.0f}s")
        for k in range(0, n, 32):
            caps = [r[2] for r in rows[k:k + 32]]
            t = proc(text=caps, return_tensors="pt", padding=True)
            tv = m.get_text_features(**{k2: v.to(dev) for k2, v in t.items()})
            tv = tv.pooler_output if hasattr(tv, "pooler_output") else tv
            te.append(np.asarray(tv.detach().cpu(), dtype=np.float32))

    AE, TE = np.vstack(ae), np.vstack(te)
    log(f"音频向量 {AE.shape} 文本向量 {TE.shape}")
    AE = AE / np.clip(np.linalg.norm(AE, axis=1, keepdims=True), 1e-12, None)
    TE = TE / np.clip(np.linalg.norm(TE, axis=1, keepdims=True), 1e-12, None)
    S = (AE @ TE.T).astype(np.float32)
    diag = np.diag(S)
    mask = ~np.eye(n, dtype=bool)
    unrel = np.median(S[mask].reshape(n, -1), axis=1)
    order = np.argsort(-S, axis=1)
    ranks = np.array([int(np.where(order[i] == i)[0][0]) + 1 for i in range(n)])

    out = {
        "model": "laion/larger_clap_general (HTSAT + RoBERTa, 双塔)",
        "group": "G",
        "dataset": "FSD50K train-00000",
        "n": int(n),
        "rel_median": round(float(np.median(diag)), 4),
        "unrel_median": round(float(np.median(unrel)), 4),
        "margin": round(float(np.median(diag) - np.median(unrel)), 4),
        "r@1": round(float((ranks == 1).mean()), 4),
        "r@5": round(float((ranks <= 5).mean()), 4),
        "r@10": round(float((ranks <= 10).mean()), 4),
        "median_rank": float(np.median(ranks)),
        "random_r1": round(1.0 / n, 6),
        "note": "与 E/F 组同数据同口径，换模型以分离「模态难」与「音频塔弱」",
        "elapsed_sec": round(time.time() - t0, 1),
    }
    np.save(os.path.join(args.out, "g_clap_simmat.npy"), S)
    with open(os.path.join(args.out, "g_clap_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log(json.dumps(out, ensure_ascii=False, indent=2))
    log("DONE")


if __name__ == "__main__":
    main()
