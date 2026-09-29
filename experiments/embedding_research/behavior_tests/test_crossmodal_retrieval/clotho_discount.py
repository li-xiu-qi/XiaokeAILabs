# -*- coding: utf-8 -*-
"""H 组：Clotho 一音五标，构造音频自己的同模态分母，闭合计折扣比例。

动机与缺口
  F 组算出 v5-omni 内视听频折扣 1.7%、图文 66.5%，差 40 倍。但 F 组的分母是
  Flickr30k caption 的 T2T，不是音频自己的同模态。音频与图文共用一个分母，这个
  折扣比例的可比性一直存疑。
  Clotho v2.1 development 是 3839 段音频，每段恰好 5 条人工 captions，一音五标
  结构与 Flickr30k 同构。于是可以在同一个模型内同时算出：
    A2T  音频对自己某条 caption 对 对其他 clip 的 caption（跨模态）
    T2T  同一 clip 的两条不同 caption 之间（音频自己的同模态）
  折扣比例 = A2T margin / T2T margin，分子分母同模型同数据集同侧，这是此前
  一直缺的东西。

为什么要两个模型
  只有 v5-omni 一个点时，「音频折扣 40 倍」只是单个模型的观测。G 组已给出
  CLAP 在同批 FSD50K 上的读数但分母仍是借的。用 Clotho 把两个模型都跑一遍，
  才能判断这个折扣是模态属性还是模型属性。

口径与 F/G 组保持一致
  余弦用 raw embedding 不预归一化；rel/unrel 中位数都先按查查询取中位数再取
  中位数（median of per-query medians）；RRF 之类不涉及。

用法（DGX）：
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ~/xinfer-env/bin/python clotho_discount.py \
      --model v5-omni --limit 500
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ~/xinfer-env/bin/python clotho_discount.py \
      --model clap --limit 500
"""
import argparse, csv, io, json, os, time, warnings
import numpy as np

warnings.filterwarnings("ignore")

CLOTHO_DIR = "/home/ke/crossmodal_lab/datasets/clotho"
CLOTHO_CSV = os.path.join(CLOTHO_DIR, "clotho_captions_development.csv")
AUDIO_ROOT_CANDIDATES = [
    os.path.join(CLOTHO_DIR, "clotho_audio_development"),
    os.path.join(CLOTHO_DIR, "development"),
    CLOTHO_DIR,
]
OUT_ROOT = "/home/ke/crossmodal_lab/results/clotho"
SR_OMNI = 16000
SR_CLAP = 48000


def log(m):
    print(m, flush=True)


def find_audio_root():
    """7z 解压后的 wav 落在哪，找一个含 .wav 的。"""
    for c in AUDIO_ROOT_CANDIDATES:
        if os.path.isdir(c):
            wavs = [f for f in os.listdir(c) if f.lower().endswith(".wav")]
            if wavs:
                return c
    for root, dirs, files in os.walk(CLOTHO_DIR):
        if any(f.lower().endswith(".wav") for f in files):
            return root
    raise SystemExit(f"找不到 wav，先解压 clotho_audio_development.7z 到 {CLOTHO_DIR}")


def audio_lookup(root):
    """strip 后的文件名 -> 磁盘真实文件名。

    解压出来的文件名带前后空格（如 ' Ambience Birds.wav'），而标注 CSV 里是干净
    的。直接拿 CSV 原名拼路径，3839 段里只能命中 3837，且缺失不报错。所以建查找
    表，两边都 strip 后匹配，实测可拿满 3839。
    """
    table = {}
    for f in os.listdir(root):
        if not f.lower().endswith(".wav"):
            continue
        table.setdefault(f.strip().lower(), f)
    return table


def load_clotho(limit):
    """[(wav_path, [5 captions])]，按文件名排序取前 limit，与 FSD50K 组同口径。"""
    if not os.path.exists(CLOTHO_CSV):
        raise SystemExit(f"缺标注文件 {CLOTHO_CSV}")
    root = find_audio_root()
    table = audio_lookup(root)
    rows = []
    with open(CLOTHO_CSV, encoding="utf-8") as f:
        rd = csv.DictReader(f)
        cols = rd.fieldnames
        cap_cols = [c for c in cols if c and c.lower().startswith("caption")]
        for row in rd:
            fn = (row.get("file_name") or "").strip()
            caps = [(row.get(c) or "").strip() for c in cap_cols]
            caps = [c for c in caps if c]
            if not fn or not caps:
                continue
            real = table.get(fn.lower())
            if real is None:
                continue
            rows.append((os.path.join(root, real), caps))
    rows.sort(key=lambda r: os.path.basename(r[0]).strip().lower())
    log(f"Clotho 音频根 {root}")
    log(f"标注列 {cap_cols}")
    log(f"命中 {len(rows)} 段（CSV 侧已 strip 匹配磁盘真实文件名）")
    if rows:
        log(f"每段 caption 条数示例 {len(rows[0][1])}")
    return rows[:limit], cap_cols


# ---------- 编码 ----------

def read_mono(path, sr_target):
    """读 wav 并重采样到 sr_target，返回 1-D float32。"""
    import soundfile as sf
    from scipy.signal import resample_poly
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != sr_target:
        x = resample_poly(x, sr_target, sr).astype(np.float32)
    return np.ascontiguousarray(x, dtype=np.float32)


def enc_omni(rows, limit):
    import sys
    sys.path.insert(0, "/home/ke/crossmodal_lab/scripts")
    import torch
    from mm_lab import CrossModalLab
    lab = CrossModalLab("jinaai/jina-embeddings-v5-omni-small")
    waves = [read_mono(p, SR_OMNI) for p, _ in rows]
    caps = [c for _, cs in rows for c in cs]
    E_a = []
    with torch.no_grad():
        for i in range(0, len(waves), 8):
            o = lab.model.encode(waves[i:i + 8], task=lab.task, normalize_embeddings=False)
            E_a.append(np.asarray(o, dtype=np.float64))
    E_t = lab.encode_text(caps, batch_size=32)
    lab.close()
    return np.concatenate(E_a), E_t, caps


def enc_clap(rows, limit):
    import torch
    from transformers import AutoProcessor, ClapModel
    mdir = "/home/ke/crossmodal_lab/models/clap_general"
    dev = "cuda"
    model = ClapModel.from_pretrained(mdir).to(dev).eval()
    proc = AutoProcessor.from_pretrained(mdir)
    waves = [read_mono(p, SR_CLAP) for p, _ in rows]
    caps = [c for _, cs in rows for c in cs]
    E_a, E_t = [], []
    with torch.no_grad():
        for i in range(0, len(waves), 32):
            # transformers 5.x：audios 已改名 audio
            inp = proc(audio=waves[i:i + 32], sampling_rate=SR_CLAP, return_tensors="pt")
            o = model.get_audio_features(**{k: v.to(dev) for k, v in inp.items()})
            E_a.append((o.pooler_output if hasattr(o, "pooler_output") else o).cpu().numpy())
        for i in range(0, len(caps), 32):
            inp = proc(text=caps[i:i + 32], return_tensors="pt", padding=True)
            o = model.get_text_features(**{k: v.to(dev) for k, v in inp.items()})
            E_t.append((o.pooler_output if hasattr(o, "pooler_output") else o).cpu().numpy())
    return np.concatenate(E_a).astype(np.float64), np.concatenate(E_t).astype(np.float64), caps


# ---------- 指标 ----------

def cosmat(A, B):
    An = A / np.linalg.norm(A, axis=1, keepdims=True)
    Bn = B / np.linalg.norm(B, axis=1, keepdims=True)
    return An @ Bn.T


def per_query_median(off_rows):
    """每查询的中位数再取中位数，与 FSD50K 组口径一致。"""
    return float(np.median([np.median(r) for r in off_rows]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["v5-omni", "clap"], required=True)
    ap.add_argument("--limit", type=int, default=500)
    args = ap.parse_args()

    rows, cap_cols = load_clotho(args.limit)
    n_clip = len(rows)
    caps_per = [len(cs) for _, cs in rows]
    n_cap = int(sum(caps_per))
    log(f"样本 {n_clip} 段音频 / {n_cap} 条 caption")

    t0 = time.time()
    if args.model == "v5-omni":
        E_a, E_t, caps = enc_omni(rows, args.limit)
    else:
        E_a, E_t, caps = enc_clap(rows, args.limit)
    log(f"编码完成 {E_a.shape} / {E_t.shape} 用 {time.time()-t0:.1f}s")

    # 文本 embedding 的段偏移
    starts = np.cumsum([0] + caps_per)

    # --- A2T：音频对文本 ---
    S = cosmat(E_a, E_t)                      # (n_clip, n_cap)
    rel = []
    for i in range(n_clip):
        s, e = starts[i], starts[i + 1]
        rel.extend(S[i, s:e].tolist())        # 自己那几条 caption
    rel = np.array(rel)
    off_rows = []
    for i in range(n_clip):
        s, e = starts[i], starts[i + 1]
        row = np.concatenate([S[i, :s], S[i, e:]])   # 其他 clip 的 caption
        off_rows.append(row)
    a2t = {"rel_median": float(np.median(rel)),
           "unrel_median": per_query_median(off_rows),
           "n_rel": int(rel.size)}

    # --- T2T：同一 clip 的两条 caption 之间（音频自己的同模态） ---
    same, diff = [], []
    T = cosmat(E_t, E_t)
    n = E_t.shape[0]
    for i in range(n_clip):
        s, e = starts[i], starts[i + 1]
        for a in range(s, e):
            for b in range(a + 1, e):
                same.append(float(T[a, b]))
    rng = np.random.default_rng(0)
    n_diff = len(same)
    tries = 0
    while len(diff) < n_diff and tries < n_diff * 50:
        tries += 1
        a, b = rng.integers(0, n, 2)
        if a == b:
            continue
        ci = int(np.searchsorted(starts, a, "right") - 1)
        cj = int(np.searchsorted(starts, b, "right") - 1)
        if ci != cj:
            diff.append(float(T[a, b]))
    t2t = {"rel_median": float(np.median(same)), "unrel_median": float(np.median(diff)),
           "n_pairs": len(same)}

    for tag, d in (("A2T", a2t), ("T2T", t2t)):
        d["margin"] = d["rel_median"] - d["unrel_median"]

    ratio = a2t["margin"] / t2t["margin"] if t2t["margin"] else float("nan")
    res = {
        "model": args.model,
        "dataset": f"Clotho v2.1 development 前 {n_clip} 段",
        "n_clip": n_clip, "n_caption": n_cap, "caps_per_clip": caps_per[:5],
        "elapsed_sec": round(time.time() - t0, 1),
        "A2T": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in a2t.items()},
        "T2T": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in t2t.items()},
        "discount_A2T_over_T2T": round(float(ratio), 4),
        "note": "分子分母同模型同数据集同侧编码，是本次能报折扣比例的前提",
    }

    os.makedirs(os.path.join(OUT_ROOT, args.model), exist_ok=True)
    outp = os.path.join(OUT_ROOT, args.model, f"h_clotho_{args.model}_summary.json")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    log(f"落盘 -> {outp}")


if __name__ == "__main__":
    main()
