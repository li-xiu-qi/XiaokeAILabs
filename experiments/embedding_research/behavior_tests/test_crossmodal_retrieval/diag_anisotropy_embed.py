# -*- coding: utf-8 -*-
"""各向异性：在 embedding 矩阵上重算，用于核对报告「结果九」的有效秩读数。

背景：diag_geometry_simmat.py 已把 hubness 与 RRF 的读数全部复现，但有效秩
对不上（余弦矩阵上复算最高 21.5，报告写 85.1）。所以报告那个数不是在余弦矩阵
上算的。本脚本回到 embedding 矩阵上算，同时把公式钉死，让读数可复现。

要算的矩阵（都取自 F 组与 G 组的同一批 500 样本）：
  omni img  v5-omni，500 张 Flickr30k 图
  omni aud  v5-omni，500 段 FSD50K 音频
  clap aud  CLAP，500 段 FSD50K 音频
报告的「图对文 / 音频对文」行，各向异性取查询侧（图或音频）的 embedding 矩阵。

embedding 落盘缓存到 results/diag_cache/，重跑不重新编码。

用法：~/xinfer-env/bin/python diag_anisotropy_embed.py [--n 500] [--check]
"""
import argparse, io, json, os, warnings
import numpy as np

warnings.filterwarnings("ignore")

FLICKR_CSV = "/home/ke/crossmodal_lab/datasets/flickr30k/flickr_annotations_30k.csv"
FLICKR_IMG = "/home/ke/crossmodal_lab/datasets/flickr30k/images/flickr30k-images"
FSD = ("/home/ke/crossmodal_lab/datasets/fsd50k/parquet/"
       "train-00000-of-00031-31accf869b307ce3.parquet")
CACHE = "/home/ke/crossmodal_lab/results/diag_cache"
OUT = "/home/ke/crossmodal_lab/results/geometry_diag"
SR_CLAP = 48000
SR_OMNI = 16000

REPORT = {
    "omni_i2t": {"eff_rank": 85.1, "rank_ratio": 0.170, "top_sv": 0.157},
    "omni_a2t": {"eff_rank": 7.3, "rank_ratio": 0.015, "top_sv": 0.655},
    "clap_a2t": {"eff_rank": 7.1, "rank_ratio": 0.014, "top_sv": 0.605},
}


def log(m):
    print(m, flush=True)


def load_flickr(n):
    import csv
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
    return rows[:n]


def load_fsd(n):
    import pyarrow.parquet as pq
    import soundfile as sf
    t = pq.read_table(FSD)
    desc = t.column("description").to_pylist()
    aud = t.column("audio").to_pylist()
    out = []
    for i in range(min(n, len(desc))):
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
    return out[:n]


def cached(tag, what, fn):
    p = os.path.join(CACHE, f"{tag}_{what}.npy")
    if os.path.exists(p):
        return np.load(p)
    arr = fn()
    os.makedirs(CACHE, exist_ok=True)
    np.save(p, arr)
    return arr


def enc_omni(tag, n):
    """v5-omni 单塔，图与音频各一个 embedding 矩阵。"""
    def _img():
        import sys
        sys.path.insert(0, "/home/ke/crossmodal_lab/scripts")
        from mm_lab import CrossModalLab
        lab = CrossModalLab("jinaai/jina-embeddings-v5-omni-small")
        rows = load_flickr(n)
        E = lab.encode_image([os.path.join(FLICKR_IMG, fn) for fn, _ in rows], batch_size=8)
        ok = ~np.isnan(E).any(axis=1)
        lab.close()
        return E[ok]

    def _aud():
        import sys
        import torch
        from scipy.signal import resample_poly
        sys.path.insert(0, "/home/ke/crossmodal_lab/scripts")
        from mm_lab import CrossModalLab
        lab = CrossModalLab("jinaai/jina-embeddings-v5-omni-small")
        waves = []
        for x, sr, _ in load_fsd(n):
            if sr != SR_OMNI:
                x = resample_poly(x, SR_OMNI, sr).astype(np.float32)
            waves.append(np.ascontiguousarray(x, dtype=np.float32))
        m = lab.model
        with torch.no_grad():
            out = []
            for i in range(0, len(waves), 8):
                out.append(np.asarray(m.encode(waves[i:i + 8], task=lab.task,
                                               normalize_embeddings=False), dtype=np.float64))
        lab.close()
        return np.concatenate(out)

    return {w: cached(tag, w, f) for w, f in [("img", _img), ("aud", _aud)]}


def enc_clap(tag, n):
    def _mk(side):
        import torch
        from scipy.signal import resample_poly
        from transformers import AutoProcessor, ClapModel
        mdir = "/home/ke/crossmodal_lab/models/clap_general"
        dev = "cuda"
        model = ClapModel.from_pretrained(mdir).to(dev).eval()
        proc = AutoProcessor.from_pretrained(mdir)
        arows = load_fsd(n)
        if side == "audio":
            items = []
            for x, sr, _ in arows:
                if sr != SR_CLAP:
                    x = resample_poly(x, SR_CLAP, sr).astype(np.float32)
                items.append(np.ascontiguousarray(x, dtype=np.float32))

            def batch(i):
                # transformers 5.x 起 audios 已废弃，必须是 audio。踩过一次。
                inp = proc(audio=items[i:i + 32], sampling_rate=SR_CLAP, return_tensors="pt")
                o = model.get_audio_features(**{k: v.to(dev) for k, v in inp.items()})
                return o.pooler_output if hasattr(o, "pooler_output") else o
        else:
            items = [d for _, _, d in arows]

            def batch(i):
                inp = proc(text=items[i:i + 32], return_tensors="pt", padding=True)
                o = model.get_text_features(**{k: v.to(dev) for k, v in inp.items()})
                return o.pooler_output if hasattr(o, "pooler_output") else o

        with torch.no_grad():
            out = [batch(i).cpu().numpy() for i in range(0, len(items), 32)]
        return np.concatenate(out).astype(np.float64)

    return {w: cached(tag, w, lambda w=w: _mk(w)) for w in ("audio", "text")}


def stats(X, center):
    """有效秩与第一奇异值占比。

    center=True 先减列均值，去掉所有向量共享的公共分量。共享分量是常量偏置，
    不是维度参与，各向异性度量应当排除它。
    分母 min(n, d)，与报告「满秩 500」口径一致。
    """
    if center:
        X = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(X, compute_uv=False)
    lam = s ** 2
    lam = lam[lam > 1e-12]
    p = lam / lam.sum()
    er = float(np.exp(-(p * np.log(p)).sum()))
    return er, er / min(X.shape), float(lam[0] / lam.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    log("编码（有缓存则跳过）")
    om = enc_omni("omni", args.n)
    cl = enc_clap("clap", args.n)
    log(f"  omni img {om['img'].shape}  aud {om['aud'].shape}")
    log(f"  clap aud {cl['audio'].shape}  txt {cl['text'].shape}")

    targets = {"omni_i2t": om["img"], "omni_a2t": om["aud"], "clap_a2t": cl["audio"]}
    res = {}
    print(f"\n{'矩阵':10s} {'n':>4s} {'d':>5s} | {'er原始':>7s} {'占比':>6s} {'首奇异':>7s} | {'er中心化':>8s} {'占比':>6s} {'首奇异':>7s}")
    for tag, X in targets.items():
        e0, r0, t0 = stats(X, center=False)
        e1, r1, t1 = stats(X, center=True)
        res[tag] = {"shape": list(X.shape),
                    "raw": {"eff_rank": round(e0, 1), "rank_ratio": round(r0, 3), "top_sv": round(t0, 3)},
                    "centered": {"eff_rank": round(e1, 1), "rank_ratio": round(r1, 3), "top_sv": round(t1, 3)}}
        print(f"{tag:10s} {X.shape[0]:4d} {X.shape[1]:5d} | {e0:7.1f} {r0:6.1%} {t0:7.1%} | {e1:8.1f} {r1:6.1%} {t1:7.1%}")

    if args.check:
        print("\n== 与报告读数比对（中心化口径）==")
        for tag, want in REPORT.items():
            got = res[tag]["centered"]
            for k, v in want.items():
                g = got.get(k)
                tol = 0.02 if v >= 1 else max(0.002, v * 0.2)
                print(f"  {'OK  ' if abs(g - v) <= tol else '偏差'} {tag}.{k}: 复算 {g}  报告 {v}")

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "anisotropy_embed.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    log(f"\n落盘 -> {OUT}/anisotropy_embed.json")


if __name__ == "__main__":
    main()
