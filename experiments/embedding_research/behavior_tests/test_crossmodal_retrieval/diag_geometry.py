# -*- coding: utf-8 -*-
"""三项几何诊断：各向异性、hubness、RRF 融合。

动因：G 组把「音频折扣低」拆成模态与模型两个变量后，剩下三个问题。它们的
共同点是都能从相似度结构直接回答，但这些读数此前只存在于终端，DGX 上无
脚本无日志，不能落盘。本脚本把它们改成可复现的一次性重算。

三项分别回答什么：
  anisotropy  各向异性，用一个数描述「向量是否都挤在一个锥里」。预测跨模态
               cos 失效程度：锥越窄，无关对之间相似度越高，margin 越薄。
  hubness     近邻集中度。某些候选会被不成比例地多路查询反复选中，这种集中度
              用 in-degree（被多少查询选进 top-k）的基尼系数衡量。
  rrf         两路 simmat 的 RRF 等权融合。检验「多模型融合是折中」是否成立。

口径：余弦一律用 raw embedding 算（不预归一化），与 mm_lab 一致。embedding
落盘缓存到 results/diag_cache/，重跑不重新编码。
"""
import argparse, io, json, os, time, warnings
import numpy as np

warnings.filterwarnings("ignore")

FLICKR_CSV = "/home/ke/crossmodal_lab/datasets/flickr30k/flickr_annotations_30k.csv"
FLICKR_IMG = "/home/ke/crossmodal_lab/datasets/flickr30k/images/flickr30k-images"
FSD = ("/home/ke/crossmodal_lab/datasets/fsd50k/parquet/"
       "train-00000-of-00031-31accf869b307ce3.parquet")
CACHE = "/home/ke/crossmodal_lab/results/diag_cache"
OUT = "/home/ke/crossmodal_lab/results/geometry_diag"
SR_CLAP = 48000


def log(m):
    print(m, flush=True)


# ---------- 数据 ----------

def load_flickr(nimg):
    """[(filename, [captions])]，按文件名排序取前 nimg，与 A/F 组同批。"""
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
    return rows[:nimg]


def load_fsd(limit):
    """FSD50K 前 limit 条 (audio, sr, desc)，与 E/G 组同批同口径。"""
    import pyarrow.parquet as pq
    import soundfile as sf
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
    return out[:limit]


# ---------- 编码 ----------

def enc_image_text(model_id, tag, nimg):
    """图文两侧 embedding。cache 命中则跳过编码。返回 (E_img, E_txt)。"""
    os.makedirs(CACHE, exist_ok=True)
    pi, pt = f"{CACHE}/{tag}_img.npy", f"{CACHE}/{tag}_txt.npy"
    if os.path.exists(pi) and os.path.exists(pt):
        return np.load(pi), np.load(pt)
    import sys
    sys.path.insert(0, "/home/ke/crossmodal_lab/scripts")
    from mm_lab import CrossModalLab
    rows = load_flickr(nimg)
    lab = CrossModalLab(model_id)
    paths = [os.path.join(FLICKR_IMG, fn) for fn, _ in rows]
    t0 = time.time()
    E_img = lab.encode_image(paths, batch_size=8)
    E_txt = lab.encode_text([c for _, caps in rows for c in caps], batch_size=32)
    ok = ~np.isnan(E_img).any(axis=1)
    np.save(pi, E_img[ok])
    np.save(pt, E_txt)
    log(f"  [{tag}] 编码 {int(ok.sum())} 图 + {E_txt.shape[0]} 文本 用 {time.time()-t0:.1f}s")
    lab.close()
    return E_img[ok], E_txt


def enc_clap(tag, naud):
    """CLAP 双塔：HTSAT 音频 + RoBERTa 文本。"""
    os.makedirs(CACHE, exist_ok=True)
    pa, pt = f"{CACHE}/{tag}_aud.npy", f"{CACHE}/{tag}_txt.npy"
    if os.path.exists(pa) and os.path.exists(pt):
        return np.load(pa), np.load(pt)
    import torch
    from scipy.signal import resample_poly
    from transformers import AutoProcessor, ClapModel
    mdir = "/home/ke/crossmodal_lab/models/clap_general"
    dev = "cuda"
    model = ClapModel.from_pretrained(mdir).to(dev).eval()
    proc = AutoProcessor.from_pretrained(mdir)
    rows = load_fsd(naud)
    t0 = time.time()
    A, T = [], []
    for x, sr, d in rows:
        if sr != SR_CLAP:
            x = resample_poly(x, SR_CLAP, sr).astype(np.float32)
        A.append(np.ascontiguousarray(x, dtype=np.float32))
        T.append(d)
    with torch.no_grad():
        av = []
        for i in range(0, len(A), 32):
            inp = proc(audios=A[i:i + 32], sampling_rate=SR_CLAP, return_tensors="pt")
            o = model.get_audio_features(**{k: v.to(dev) for k, v in inp.items()})
            av.append((o.pooler_output if hasattr(o, "pooler_output") else o).cpu().numpy())
        tv = []
        for i in range(0, len(T), 32):
            inp = proc(text=T[i:i + 32], return_tensors="pt", padding=True)
            o = model.get_text_features(**{k: v.to(dev) for k, v in inp.items()})
            tv.append((o.pooler_output if hasattr(o, "pooler_output") else o).cpu().numpy())
    E_a = np.concatenate(av).astype(np.float64)
    E_t = np.concatenate(tv).astype(np.float64)
    np.save(pa, E_a)
    np.save(pt, E_t)
    log(f"  [{tag}] 编码 {len(A)} 音频 + {len(T)} 文本 用 {time.time()-t0:.1f}s")
    return E_a, E_t


def enc_omni(tag, nimg, naud):
    """v5-omni 单塔，图文音同一空间。"""
    os.makedirs(CACHE, exist_ok=True)
    p_i, p_a, p_t = f"{CACHE}/{tag}_img.npy", f"{CACHE}/{tag}_aud.npy", f"{CACHE}/{tag}_txt.npy"
    if os.path.exists(p_i) and os.path.exists(p_a) and os.path.exists(p_t):
        return np.load(p_i), np.load(p_a), np.load(p_t)
    import sys
    sys.path.insert(0, "/home/ke/crossmodal_lab/scripts")
    from mm_lab import CrossModalLab
    from scipy.signal import resample_poly
    frows = load_flickr(nimg)
    arows = load_fsd(naud)
    lab = CrossModalLab("jinaai/jina-embeddings-v5-omni-small")
    t0 = time.time()
    E_img = lab.encode_image([os.path.join(FLICKR_IMG, fn) for fn, _ in frows], batch_size=8)
    caps = [c for _, cs in frows for c in cs]
    auds = []
    for x, sr, _ in arows:
        if sr != 16000:
            x = resample_poly(x, 16000, sr).astype(np.float32)
        auds.append(np.ascontiguousarray(x, dtype=np.float32))
    E_txt_c = lab.encode_text(caps, batch_size=32)
    m = lab.model
    E_aud = []
    with m.no_grad() if hasattr(m, "no_grad") else _null():
        pass
    import torch
    with torch.no_grad():
        for i in range(0, len(auds), 8):
            o = m.encode(auds[i:i + 8], task=lab.task, normalize_embeddings=False)
            E_aud.append(np.asarray(o, dtype=np.float64))
    E_aud = np.concatenate(E_aud)
    adesc = [d for _, _, d in arows]
    E_txt_a = lab.encode_text(adesc, batch_size=32)
    np.save(p_i, E_img)
    np.save(p_t, E_txt_a)
    np.save(p_a, E_aud)
    log(f"  [{tag}] 编码 {E_img.shape[0]} 图 + {len(auds)} 音频 用 {time.time()-t0:.1f}s")
    lab.close()
    return E_img, E_aud, E_txt_a


class _null:
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False


# ---------- 指标 ----------

def cosmat(A, B):
    """行归一化后点积 = 余弦。"""
    An = A / np.linalg.norm(A, axis=1, keepdims=True)
    Bn = B / np.linalg.norm(B, axis=1, keepdims=True)
    return An @ Bn.T


def eff_rank_ratio(X):
    """有效秩 / min(n, d)。参与维度占比，1.0 表示各向同性铺满，趋 0 表示挤在一维。"""
    Xc = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    ev = s ** 2
    ev = ev[ev > 1e-12]
    er = (ev.sum() ** 2) / (ev ** 2).sum()
    return float(er / min(X.shape)), float(er)


def top_sv_frac(X):
    """第一奇异值占总方差比例。各向异性高时少数方向吃掉大部分方差。"""
    Xc = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    ev = s ** 2
    return float(ev[0] / ev.sum())


def selfsim(X):
    """随机对余弦，给无关相似度水位与负值占比。"""
    n = X.shape[0]
    if n > 400:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, 400, replace=False)
        X = X[idx]
    S = cosmat(X, X)
    iu = np.triu_indices(S.shape[0], 1)
    v = S[iu]
    return {"median": float(np.median(v)), "neg_ratio": float((v < 0).mean())}


def anisotropy(X):
    return {"eff_rank_ratio": eff_rank_ratio(X)[0],
            "eff_rank": eff_rank_ratio(X)[1],
            "top_sv_frac": top_sv_frac(X),
            **{f"selfsim_{k}": v for k, v in selfsim(X).items()}}


def hubness(S, k=10):
    """候选侧 in-degree：每列被多少 query 选进 top-k。基尼系数衡量集中度。"""
    n = S.shape[0]
    idx = np.argsort(-S, axis=1)[:, :k]
    deg = np.bincount(idx.ravel(), minlength=n).astype(float)
    deg_sorted = np.sort(deg)
    m = len(deg_sorted)
    cum = np.cumsum(deg_sorted)
    gini = float((2 * np.arange(1, m + 1) - m - 1).dot(deg_sorted) / (m * deg_sorted.sum()))
    top = np.argsort(-deg)[:5]
    return {"k": k, "gini": gini, "deg_mean": float(deg.mean()),
            "deg_max": float(deg.max()),
            "top5": [{"cand": int(i), "deg": int(deg[i])} for i in top]}


def rrf(ranks_list, weights=None):
    """RRF 融合多路 rank。ranks_list: [ (n_query, n_cand) 的 0-based rank ] 。"""
    nq, nc = ranks_list[0].shape
    if weights is None:
        weights = [1.0] * len(ranks_list)
    score = np.zeros((nq, nc))
    for R, w in zip(ranks_list, weights):
        score += w / (60.0 + R.astype(np.float64) + 1.0)
    return np.argsort(-score, axis=1)


def r1(S):
    """对角线是否排第一（查询 i 的正确候选项是 i）。"""
    return float((np.argmax(S, axis=1) == np.arange(S.shape[0])).mean())


def rank_of_correct(S):
    return np.argsort(-S, axis=1).argsort(axis=1).diagonal() + 1


def pkg_retrieval(S):
    r = rank_of_correct(S)
    return {"r@1": float((r <= 1).mean()), "r@5": float((r <= 5).mean()),
            "r@10": float((r <= 10).mean()), "median_rank": float(np.median(r)),
            "margin": float(np.median(S.diagonal()) - np.median(S + 1e9 * np.eye(S.shape[0])[0:1][0] if False else S + np.where(np.eye(S.shape[0], dtype=bool), 1e9, 0))),
            "rel_median": float(np.median(S.diagonal())),
            "unrel_median": float(np.median(S + np.where(np.eye(S.shape[0], dtype=bool), 1e9, 0)))}


def single(S, correct_fn):
    """S.shape = (n_query, n_cand)，correct_fn 给出正确候选下标。"""
    correct = np.asarray(correct_fn(S))
    r = np.argsort(-S, axis=1).argsort(axis=1)[np.arange(S.shape[0]), correct]
    r = r + 1
    off = S.copy()
    off[np.arange(S.shape[0]), correct] = -np.inf
    return {"r@1": float((r <= 1).mean()), "r@5": float((r <= 5).mean()),
            "r@10": float((r <= 10).mean()), "median_rank": float(np.median(r)),
            "rel_median": float(np.median(S[np.arange(S.shape[0]), correct])),
            "unrel_median": float(np.median(off))}


# ---------- 主流程 ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nimg", type=int, default=500)
    ap.add_argument("--naud", type=int, default=500)
    ap.add_argument("--skip-rrf", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    res = {"config": {"nimg": args.nimg, "naud": args.naud,
                      "cos": "raw embedding，不预归一化"},
           "anisotropy": {}, "hubness": {}, "rrf": {}}

    # --- 1. 各向异性：图文用 caption 文本矩阵，音频用描述文本矩阵 + 音频矩阵 ---
    log("[1/3] 各向异性")
    v4_img, v4_txt = enc_image_text("jinaai/jina-embeddings-v4", "v4", args.nimg)
    bvl_img, bvl_txt = enc_image_text("BAAI/BGE-VL-base", "bvl", args.nimg)
    cl_a, cl_t = enc_clap("clap", args.naud)
    om_img, om_a, om_t = enc_omni("omni", args.nimg, args.naud)

    res["anisotropy"]["v4_text"] = anisotropy(v4_txt)
    res["anisotropy"]["v4_image"] = anisotropy(v4_img)
    res["anisotropy"]["bvl_text"] = anisotropy(bvl_txt)
    res["anisotropy"]["bvl_image"] = anisotropy(bvl_img)
    res["anisotropy"]["clap_audio"] = anisotropy(cl_a)
    res["anisotropy"]["clap_text"] = anisotropy(cl_t)
    res["anisotropy"]["omni_image"] = anisotropy(om_img)
    res["anisotropy"]["omni_audio"] = anisotropy(om_a)
    res["anisotropy"]["omni_text"] = anisotropy(om_t)

    # --- 2. hubness：查询选候选的集中度 ---
    log("[2/3] hubness")
    res["hubness"]["v4_i2t"] = hubness(cosmat(v4_img, v4_txt[:len(v4_img)]))
    res["hubness"]["bvl_i2t"] = hubness(cosmat(bvl_img, bvl_txt[:len(bvl_img)]))
    res["hubness"]["clap_a2t"] = hubness(cosmat(cl_a, cl_t))
    res["hubness"]["omni_i2t"] = hubness(cosmat(om_img, om_t))
    res["hubness"]["omni_a2t"] = hubness(cosmat(om_a, om_t))

    # --- 3. RRF：jina-v4 与 BGE-VL 同批图文，等权融合 ---
    log("[3/3] RRF")
    if not args.skip_rrf:
        n = min(len(v4_img), len(bvl_img))
        S4 = cosmat(v4_img[:n], v4_txt[:n])
        SB = cosmat(bvl_img[:n], bvl_txt[:n])
        res["rrf"]["v4_alone"] = single(S4, lambda s: np.arange(n))
        res["rrf"]["bvl_alone"] = single(SB, lambda s: np.arange(n))
        R4 = np.argsort(-S4, axis=1).argsort(axis=1)
        RB = np.argsort(-SB, axis=1).argsort(axis=1)
        Srrf = rrf([R4, RB])
        # 融合后 correct=diagonal
        fused_rank = Srrf.argsort(axis=1).argsort(axis=1)[np.arange(n), np.arange(n)] + 1
        res["rrf"]["fused"] = {"r@1": float((fused_rank <= 1).mean()),
                               "r@5": float((fused_rank <= 5).mean()),
                               "r@10": float((fused_rank <= 10).mean()),
                               "median_rank": float(np.median(fused_rank))}
        res["rrf"]["quality_gap"] = float(res["rrf"]["v4_alone"]["r@1"] /
                                          max(res["rrf"]["bvl_alone"]["r@1"], 1e-9))

    with open(f"{OUT}/geometry_diag.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    log(f"完成 -> {OUT}/geometry_diag.json")


if __name__ == "__main__":
    main()
