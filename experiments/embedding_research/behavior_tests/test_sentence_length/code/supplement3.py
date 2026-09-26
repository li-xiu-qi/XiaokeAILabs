# -*- coding: utf-8 -*-
"""补做四（2026-09-26）：语义粒度分层探索。

动机：主实验与三次补做全部停在「句对」这一层，输入单位固定为句子。但用户
使用相似度时的实际单位跨三层：词与词、句子与句子、段落与段落。三层之间
分数是否一致、哪一层最稳、哪一层最抖，此前没有数据。本补做把三层一次测全，
并配语义空间投影图。

四个阶段，每个模型只加载一次，全部文本一次 encode：

  A 词级     同义 / 反义 / 上下位 / 共现 / 无关 / 跨语言同义 六类词对的相似度。
             检验两个经典假设：反义词因语境共现，相似度高于无关词；
             跨语言同义对只有语言覆盖够的模型才给高分。
  B 表达谱   同一命题的四种说法（口语、书面、简化、详细）对原句的 Δ，
             加变体之间的两两相似度。测「换个说法」的代价。
  C 段落粒度 同一主题拼 1 到 4 段，测跨长度配对的相似度曲线。
             回答长度效应在段落级是否饱和，与主实验句级幅度对比。
  D 图       词空间 t-SNE 投影、表达谱分布、段落饱和曲线。

局限：词表与变体为人工构造，非词频抽样；syn 组内词是语义近似而非严格同义；
t-SNE 只作可视化，距离不可当度量读。

用法（DGX Spark）：
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ~/xinfer-env/bin/python supplement3.py
  ~/xinfer-env/bin/python supplement3.py analyze
"""

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
import lab  # noqa: E402
from semantics_data import (  # noqa: E402
    WORD_GROUPS_ZH, WORD_GROUPS_EN, COOC_PAIRS_ZH, COOC_PAIRS_EN,
    UNRELATED_PAIRS_ZH, UNRELATED_PAIRS_EN, CROSS_PAIRS, ANT_GROUP_PAIRS,
    PARA_VARIANTS, PARA_EXTRA,
)
from themes import THEMES  # noqa: E402

# DGX 上脚本与 results 同级（config 的 RESULTS_DIR 按仓内 code/ 布局解析，此处覆盖）
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

VARIANTS = ["oral", "formal", "brief", "verbose"]
SEGMENTS = [1, 2, 3, 4]


# ------------------------------------------------------------------ 配对构建

def _is_zh(w: str) -> bool:
    """按字符判定中英文。词表是中英混排的，需要分开编码与配对。"""
    return any("\u4e00" <= ch <= "\u9fff" for ch in w)


def build_word_pairs():
    """词级配对表。返回 (pairs, zh_words, en_words)。"""
    pairs = []

    def add(lang, rel, w1, w2, note=""):
        pairs.append({"lang": lang, "rel": rel, "w1": w1, "w2": w2, "note": note})

    for groups, lang in ((WORD_GROUPS_ZH, "zh"), (WORD_GROUPS_EN, "en")):
        by_gid = {g["gid"]: g for g in groups}
        # 组内同义
        for g in groups:
            ws = g["words"]
            for i in range(len(ws)):
                for j in range(i + 1, len(ws)):
                    add(lang, "syn", ws[i], ws[j], g["gid"])
        # 反义组间
        for ga, gb in ANT_GROUP_PAIRS:
            if ga not in by_gid or gb not in by_gid:
                continue
            for w1 in by_gid[ga]["words"]:
                for w2 in by_gid[gb]["words"]:
                    add(lang, "ant", w1, w2, f"{ga}-{gb}")
        # 上下位：具体组 × 上位组
        if "dog" in by_gid and "animal" in by_gid:
            for w1 in by_gid["dog"]["words"]:
                for w2 in by_gid["animal"]["words"]:
                    add(lang, "hyper", w1, w2, "dog-animal")
        # 共现对
        for w1, w2 in (COOC_PAIRS_ZH if lang == "zh" else COOC_PAIRS_EN):
            add(lang, "cooc", w1, w2)
        # 无关对（显式构造）
        for w1, w2 in (UNRELATED_PAIRS_ZH if lang == "zh" else UNRELATED_PAIRS_EN):
            add(lang, "unrelated", w1, w2)
        # 跨组全对基线：不同组之间的所有配对，作为 unrelated 的分布基线
        gids = [g["gid"] for g in groups]
        for i in range(len(gids)):
            for j in range(i + 1, len(gids)):
                ga, gb = gids[i], gids[j]
                if (ga, gb) in ANT_GROUP_PAIRS or (gb, ga) in ANT_GROUP_PAIRS:
                    continue
                if {ga, gb} == {"dog", "animal"}:
                    continue
                for w1 in by_gid[ga]["words"]:
                    for w2 in by_gid[gb]["words"]:
                        add(lang, "crossgroup", w1, w2, f"{ga}-{gb}")

    for w1, w2 in CROSS_PAIRS:
        add("cross", "cross_lang", w1, w2)

    zh_words = sorted({w for p in pairs if _is_zh(p["w1"]) or _is_zh(p["w2"]) for w in (p["w1"], p["w2"]) if _is_zh(w)})
    en_words = sorted({w for p in pairs for w in (p["w1"], p["w2"]) if not _is_zh(w)})
    return pairs, zh_words, en_words


def build_paraphrase_rows():
    """表达谱文本对：orig vs 四个变体，加变体间两两。"""
    theme_by_id = {t["id"]: t for t in THEMES}
    rows = []
    for pv in PARA_VARIANTS:
        t = theme_by_id[pv["id"]]
        for lang, orig, pfx in (("zh", t["a"], ""), ("en", t["en_a"], "en_")):
            versions = {"orig": orig}
            for v in VARIANTS:
                versions[v] = pv[pfx + v]
            keys = list(versions)
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    rows.append({
                        "theme_id": pv["id"],
                        "lang": lang,
                        "pair": f"{keys[i]}-{keys[j]}",
                        "text_x": versions[keys[i]],
                        "text_y": versions[keys[j]],
                    })
    return rows


def build_paragraph_rows():
    """段落粒度文本对：同主题拼 1 到 4 段，跨长度配对，加跨主题负对照。

    负对照（X 前缀 = 对手主题，取列表中下一个主题，轮转）：
      X1-Lk  1 段 vs k 段，跨主题。与 L1-Lk 相减得到「同为跨长度时，
             主题相关性贡献多少」，把长度稀释与主题漂移分开。
      Xk-Lk  k 段 vs k 段，跨主题，同粒度。与 Lk-Lk 相减得到长度不变时
             的纯主题 gap，可与主实验结果四的句级 gap 对比。
    """
    theme_by_id = {t["id"]: t for t in THEMES}
    extra_by_id = {e["id"]: e for e in PARA_EXTRA}
    tids = [pv["id"] for pv in PARA_VARIANTS]
    rows = []
    for pos, tid in enumerate(tids):
        t, e = theme_by_id[tid], extra_by_id[tid]
        rival = tids[(pos + 1) % len(tids)]
        tr, er = theme_by_id[rival], extra_by_id[rival]
        for lang, k, pfx in (("zh", "", ""), ("en", "en_", "en_")):
            def seg_texts(base_t, base_e):
                sents = [base_t[k + "a"], base_e[pfx + "c"], base_e[pfx + "d"], base_t[k + "a_rel"]]
                cleaned = [x.rstrip("。") for x in sents]
                return {n: ("。" if lang == "zh" else " ").join(cleaned[:n]) for n in SEGMENTS}

            own, opp = seg_texts(t, e), seg_texts(tr, er)
            for na, nb in [(1, 1), (1, 2), (1, 3), (1, 4), (2, 2), (2, 3), (2, 4),
                           (3, 3), (3, 4), (4, 4)]:
                rows.append({"theme_id": tid, "lang": lang, "pair": f"L{na}-L{nb}",
                             "seg_a": na, "seg_b": nb,
                             "text_x": own[na], "text_y": own[nb]})
            for nb in SEGMENTS:
                rows.append({"theme_id": tid, "lang": lang, "pair": f"X1-L{nb}",
                             "seg_a": 1, "seg_b": nb,
                             "text_x": own[1], "text_y": opp[nb]})
                rows.append({"theme_id": tid, "lang": lang, "pair": f"X{nb}-L{nb}",
                             "seg_a": nb, "seg_b": nb,
                             "text_x": opp[nb], "text_y": own[nb]})
    return rows


# ------------------------------------------------------------------ 计算

def stage_words(model_name, model):
    pairs, zh_words, en_words = build_word_pairs()
    vocab = zh_words + en_words
    emb_all = lab.encode(model, vocab)
    idx = {w: i for i, w in enumerate(vocab)}

    out = []
    for p in pairs:
        if p["lang"] == "cross":
            va, vb = emb_all[idx[p["w1"]]], emb_all[idx[p["w2"]]]
        else:
            va, vb = emb_all[idx[p["w1"]]], emb_all[idx[p["w2"]]]
        denom = np.clip(np.linalg.norm(va) * np.linalg.norm(vb), 1e-12, None)
        out.append({
            "model": model_name, "lang": p["lang"], "rel": p["rel"],
            "w1": p["w1"], "w2": p["w2"], "note": p["note"],
            "sim": float(np.dot(va, vb) / denom),
        })
    df = pd.DataFrame(out)
    # 词向量留档，供 t-SNE 复现（analyze 阶段重画图不再加载模型）
    np.savez_compressed(
        os.path.join(RESULTS_DIR, f"word_emb_{model_name}.npz"),
        vocab=np.array(vocab, dtype=object), emb=emb_all,
    )
    return df


def stage_pairs(model_name, model, rows, key_fields):
    """通用：按行算余弦，附加 key_fields。"""
    if not rows:
        return pd.DataFrame()
    emb = lab.encode(model, [r["text_x"] for r in rows] + [r["text_y"] for r in rows])
    n = len(rows)
    va, vb = emb[:n], emb[n:]
    denom = np.clip(np.linalg.norm(va, axis=1) * np.linalg.norm(vb, axis=1), 1e-12, None)
    sims = np.sum(va * vb, axis=1) / denom
    out = []
    for r, s in zip(rows, sims):
        rec = {"model": model_name, "sim": float(s)}
        rec.update({k: r[k] for k in key_fields})
        out.append(rec)
    return pd.DataFrame(out)


# ------------------------------------------------------------------ 主流程

def run_all():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    para_rows = build_paraphrase_rows()
    para_seg_rows = build_paragraph_rows()
    frames = {"word": [], "para": [], "seg": []}

    for m in config.MODELS:
        name = m["name"]
        t0 = time.time()
        print(f"[{name}] 加载模型 ...", flush=True)
        model = lab.load_model(m["hf_id"])
        try:
            print(f"[{name}] 阶段 A 词级 ...", flush=True)
            frames["word"].append(stage_words(name, model))
            print(f"[{name}] 阶段 B 表达谱（{len(para_rows)} 对）...", flush=True)
            frames["para"].append(stage_pairs(name, model, para_rows, ["theme_id", "lang", "pair"]))
            print(f"[{name}] 阶段 C 段落粒度（{len(para_seg_rows)} 对）...", flush=True)
            frames["seg"].append(stage_pairs(name, model, para_seg_rows,
                                             ["theme_id", "lang", "pair", "seg_a", "seg_b"]))
            print(f"[{name}] 完成，用时 {time.time() - t0:.0f}s", flush=True)
        finally:
            del model

    for k, fname in (("word", "raw_word_pairs.csv"),
                     ("para", "raw_paraphrase.csv"),
                     ("seg", "raw_paragraph.csv")):
        df = pd.concat(frames[k], ignore_index=True)
        df.to_csv(os.path.join(RESULTS_DIR, fname), index=False, encoding="utf-8-sig")
        print(f"写出 {fname}：{len(df)} 行", flush=True)


# ------------------------------------------------------------------ 分析

def _med(series):
    return float(np.median(series.values)) if len(series) else float("nan")


def analyze():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "AR PL UMing CN", "SimHei", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False

    wdf = pd.read_csv(os.path.join(RESULTS_DIR, "raw_word_pairs.csv"))
    pdf = pd.read_csv(os.path.join(RESULTS_DIR, "raw_paraphrase.csv"))
    sdf = pd.read_csv(os.path.join(RESULTS_DIR, "raw_paragraph.csv"))

    # ---- 词级汇总
    rels = ["syn", "ant", "hyper", "cooc", "crossgroup", "unrelated"]
    rows = []
    for (model, lang), g in wdf.groupby(["model", "lang"]):
        for rel in rels:
            sub = g[g["rel"] == rel]
            if len(sub):
                rows.append({"model": model, "lang": lang, "rel": rel,
                             "n": len(sub), "median": _med(sub["sim"])})
    # 跨语言同义对单独一列（lang 记为 cross）
    for model, g in wdf[wdf["lang"] == "cross"].groupby("model"):
        rows.append({"model": model, "lang": "cross", "rel": "cross_lang",
                     "n": len(g), "median": _med(g["sim"])})
    wsum = pd.DataFrame(rows)
    wsum.to_csv(os.path.join(RESULTS_DIR, "word_summary.csv"), index=False, encoding="utf-8-sig")

    print("=== 词级：各关系类型的中位相似度 ===")
    for lang in ["zh", "en", "cross"]:
        sub = wsum[wsum["lang"] == lang]
        if sub.empty:
            continue
        pivot = sub.pivot_table(index="model", columns="rel", values="median")
        cols = [c for c in ["syn", "ant", "hyper", "cooc", "crossgroup", "unrelated", "cross_lang"] if c in pivot.columns]
        print(f"\n[{lang}]")
        print(pivot[cols].round(3).to_string())

    # 关键检验：反义词 vs 无关词
    print("\n=== 词级坍缩自检：精确等于 1.0 的词对占比（>10% 判为分词坍缩，词级读数作废）===")
    for (model, lang), g in wdf.groupby(["model", "lang"]):
        if len(g) == 0:
            continue
        frac = float((g["sim"] >= 0.999999).mean())
        flag = "  <-- 分词坍缩，该格词级读数不可用" if frac > 0.10 else ""
        print(f"{model:14s} {lang:6s} {frac * 100:5.1f}%{flag}")

    print("\n=== 反义词与无关词的差（ant 减 unrelated，正值=反义词更像）===")
    for lang in ["zh", "en"]:
        sub = wsum[wsum["lang"] == lang]
        if sub.empty:
            continue
        p = sub.pivot_table(index="model", columns="rel", values="median")
        if "ant" in p.columns and "unrelated" in p.columns:
            d = (p["ant"] - p["unrelated"]).round(3)
            print(f"[{lang}] " + "  ".join(f"{m}:{v:+.3f}" for m, v in d.items()))

    # 跨语言同义必须减无关基线：绝对 sim 会被各向异性 hub 抬高，
    # large-zh 的 cross_lang(0.662) 高于它自己的英文同义(0.451) 就是伪影
    print("\n=== 跨语言同义的 discrimination gap（cross_lang 减 unrelated）===")
    for model, g in wdf[wdf["lang"] == "cross"].groupby("model"):
        cross = _med(g["sim"])
        zh_un = _med(wdf[(wdf["model"] == model) & (wdf["lang"] == "zh") & (wdf["rel"] == "unrelated")]["sim"])
        print(f"{model:14s} cross_lang={cross:.3f}  zh_unrelated={zh_un:.3f}  gap={cross - zh_un:+.3f}")

    # ---- 表达谱汇总
    # orig 与自身的相似度恒为 1（同一文本自比），以此为基准算各变体的 Δ
    print("\n=== 表达谱：orig 对各变体的 Δ（中位，基准=同文本自比 1.0）===")
    delta_rows = []
    for _, r in pdf[pdf["pair"].str.startswith("orig-")].iterrows():
        delta_rows.append({"model": r["model"], "lang": r["lang"],
                           "variant": r["pair"].split("-")[1], "delta": r["sim"] - 1.0})
    pdelta = pd.DataFrame(delta_rows)
    if not pdelta.empty:
        ptab = pdelta.pivot_table(index=["model", "variant"], columns="lang", values="delta", aggfunc=_med)
        print(ptab.round(3).to_string())
        pdelta.to_csv(os.path.join(RESULTS_DIR, "paraphrase_delta.csv"),
                      index=False, encoding="utf-8-sig")

    print("\n=== 表达谱：变体两两相似度（中位，orig 除外）===")
    pv = pdf[~pdf["pair"].str.contains("orig")].pivot_table(
        index="pair", columns=["model", "lang"], values="sim", aggfunc=_med)
    print(pv.round(3).to_string())

    # ---- 段落粒度
    print("\n=== 段落粒度：L1 对 Lk 的相似度中位 ===")
    l1 = sdf[sdf["pair"] == "L1-L1"].set_index(["model", "lang", "theme_id"])["sim"]
    curve_rows = []
    for k in SEGMENTS:
        sub = sdf[sdf["pair"] == f"L1-L{k}"]
        for (model, lang), g in sub.groupby(["model", "lang"]):
            ref = l1.get((model, lang), np.nan)
            curve_rows.append({"model": model, "lang": lang, "seg": k,
                               "median": _med(g["sim"]), "L1_base": ref})
    cdf = pd.DataFrame(curve_rows)
    cdf["drop"] = cdf["median"] - cdf["L1_base"]
    print(cdf.pivot_table(index="model", columns=["lang", "seg"], values="median").round(3).to_string())

    # 段落级 discrimination gap。注意 Lk-Lk 是同一文本自比，恒等于 1.0，
    # 不能当同粒度基准用；真正的基准是跨主题的 Xk-Lk。
    #   gap_len(k) = L1-Lk 减 X1-Lk ：同为「1 段 vs k 段」，主题相关与无关的差
    #   gap_same(k) = Lk-Lk 减 Xk-Lk ：同粒度下主题相关与无关的差（纯主题 gap）
    gap_rows = []
    for k in SEGMENTS:
        for (model, lang), g in sdf[sdf["pair"] == f"L1-L{k}"].groupby(["model", "lang"]):
            opp = sdf[(sdf["model"] == model) & (sdf["lang"] == lang) & (sdf["pair"] == f"X1-L{k}")]
            if len(opp):
                gap_rows.append({"model": model, "lang": lang, "seg": k, "gap": "gap_len",
                                 "value": _med(g["sim"]) - _med(opp["sim"])})
    for k in SEGMENTS:
        for (model, lang), g in sdf[sdf["pair"] == f"L{k}-L{k}"].groupby(["model", "lang"]):
            opp = sdf[(sdf["model"] == model) & (sdf["lang"] == lang) & (sdf["pair"] == f"X{k}-L{k}")]
            if len(opp):
                gap_rows.append({"model": model, "lang": lang, "seg": k, "gap": "gap_same",
                                 "value": _med(g["sim"]) - _med(opp["sim"])})
    gdf = pd.DataFrame(gap_rows)
    for kind, title in (("gap_len", "长度不等时的主题 gap（L1-Lk 减 X1-Lk）"),
                        ("gap_same", "同粒度纯主题 gap（Lk-Lk 减 Xk-Lk）")):
        print(f"\n=== {title} ===")
        print(gdf[gdf["gap"] == kind].pivot_table(index="model", columns=["lang", "seg"],
                                                  values="value").round(3).to_string())
    gdf.to_csv(os.path.join(RESULTS_DIR, "paragraph_gap.csv"), index=False, encoding="utf-8-sig")
    cdf.to_csv(os.path.join(RESULTS_DIR, "paragraph_curve.csv"),
               index=False, encoding="utf-8-sig")

    # ---- 图 1：词级关系统计
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    models = config.MODELS
    palette = {"syn": "#2b6cb0", "ant": "#c53030", "hyper": "#2f855a",
               "cooc": "#b77921", "crossgroup": "#718096",
               "unrelated": "#4a5568", "cross_lang": "#805ad5"}
    for ax, m in zip(axes.flat, models):
        name = m["name"]
        for lang, ls in (("zh", "-"), ("en", "--")):
            sub = wsum[(wsum["model"] == name) & (wsum["lang"] == lang)]
            for rel in rels:
                v = sub[sub["rel"] == rel]["median"]
                if len(v):
                    ax.scatter([rel], [v.values[0]], c=palette[rel], marker="s" if lang == "zh" else "o",
                               s=90, edgecolors="white", linewidths=1.2, zorder=3)
        sub = wsum[(wsum["model"] == name) & (wsum["lang"] == "cross")]
        if len(sub):
            ax.scatter(["cross_lang"], [sub["median"].values[0]], c=palette["cross_lang"],
                       marker="D", s=90, edgecolors="white", linewidths=1.2, zorder=3)
        ax.set_title(name, fontsize=11)
        ax.set_ylim(-0.1, 1.02)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", labelrotation=35, labelsize=8)
    handles = [plt.Line2D([], [], marker="s", ls="", c=palette[r], label=r) for r in rels]
    handles.append(plt.Line2D([], [], marker="D", ls="", c=palette["cross_lang"], label="cross_lang"))
    handles.append(plt.Line2D([], [], marker="s", ls="", c="gray", label="中文"))
    handles.append(plt.Line2D([], [], marker="o", ls="", c="gray", label="英文"))
    fig.legend(handles=handles, loc="lower center", ncol=8, fontsize=9, frameon=False)
    fig.suptitle("词级：六类词对的中位相似度（方块=中文，圆点=英文，菱形=跨语言同义）", fontsize=13)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    fig.savefig(os.path.join(FIGURES_DIR, "fig4_word_relations.png"), dpi=150)
    plt.close(fig)

    # ---- 图 2-6：每模型词空间 t-SNE
    from sklearn.manifold import TSNE
    for m in models:
        name = m["name"]
        npz_path = os.path.join(RESULTS_DIR, f"word_emb_{name}.npz")
        if not os.path.exists(npz_path):
            continue
        data = np.load(npz_path, allow_pickle=True)
        vocab = list(data["vocab"])
        emb = data["emb"]
        proj = TSNE(n_components=2, perplexity=5, random_state=42,
                    init="pca", learning_rate="auto").fit_transform(emb)
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        w2g = {}
        for g in WORD_GROUPS_ZH:
            for w in g["words"]:
                w2g[w] = g["gid"]
        for g in WORD_GROUPS_EN:
            for w in g["words"]:
                w2g[w] = g["gid"]
        gcolor = {"joy": "#c53030", "sorrow": "#dd6b20", "big": "#2b6cb0",
                  "small": "#63b3ed", "buy": "#2f855a", "sell": "#38a169",
                  "fast": "#805ad5", "slow": "#b794f4", "dog": "#b77921",
                  "animal": "#d69e2e"}
        # 中英文词必须分开画：词表是中英混排的，混画会把英文词标进中文子图
        for ax, want_zh in ((axes[0], True), (axes[1], False)):
            words = [w for w in vocab if w in w2g and _is_zh(w) == want_zh]
            ax.set_title(f"{name} · {'中文词空间' if want_zh else '英文词空间'}", fontsize=11)
            for w in words:
                i = vocab.index(w)
                gid = w2g[w]
                ax.scatter(proj[i, 0], proj[i, 1], c=gcolor.get(gid, "#718096"),
                           s=110, edgecolors="white", linewidths=1.2, zorder=3)
                ax.annotate(w, (proj[i, 0], proj[i, 1]), fontsize=7.5,
                            xytext=(4, 3), textcoords="offset points")
            ax.set_xticks([])
            ax.set_yticks([])
            if not want_zh:
                handles = [plt.Line2D([], [], marker="o", ls="", c=gcolor[g], label=g)
                           for g in gcolor if g in {w2g[w] for w in words}]
                ax.legend(handles=handles, fontsize=8, loc="best", frameon=False)
        fig.suptitle(f"词空间 t-SNE 投影（perplexity=5，仅可视化，距离不可当度量）", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(os.path.join(FIGURES_DIR, f"fig5_tsne_{name}.png"), dpi=150)
        plt.close(fig)

    # ---- 图 7-11：表达谱 + 段落曲线（每模型一张双子图）
    for m in models:
        name = m["name"]
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        ax = axes[0]
        width = 0.35
        vlist = VARIANTS
        for off, (lang, mk) in enumerate((("zh", 0), ("en", 1))):
            vals = []
            for v in vlist:
                sub = pdelta[(pdelta["model"] == name) & (pdelta["lang"] == lang) &
                             (pdelta["variant"] == v)]["delta"]
                vals.append(_med(sub) if len(sub) else np.nan)
            xpos = np.arange(len(vlist)) + (off - 0.5) * width
            ax.bar(xpos, vals, width=width, label=lang, alpha=0.9,
                   color="#2b6cb0" if lang == "zh" else "#dd6b20")
        ax.axhline(0, color="#4a5568", linewidth=0.8)
        ax.set_xticks(np.arange(len(vlist)))
        ax.set_xticklabels(vlist, fontsize=9)
        ax.set_ylabel("相对原句的相似度变化")
        ax.set_title(f"{name} · 同一命题换个说法掉多少分", fontsize=11)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False, fontsize=9)

        ax = axes[1]
        for lang, mk in (("zh", "s"), ("en", "o")):
            sub = cdf[(cdf["model"] == name) & (cdf["lang"] == lang)].sort_values("seg")
            if len(sub):
                ax.plot(sub["seg"], sub["median"], marker=mk, label=f"{lang} 同主题", linewidth=1.6)
            opp = sdf[(sdf["model"] == name) & (sdf["lang"] == lang) & (sdf["pair"] == "X1-L1") |
                      (sdf["model"] == name) & (sdf["lang"] == lang) & (sdf["pair"].isin(["X1-L2", "X1-L3", "X1-L4"]))]
            if len(opp):
                curve = opp.groupby("seg_b")["sim"].apply(_med)
                ax.plot(curve.index, curve.values, marker=mk, ls="--", alpha=0.75,
                        label=f"{lang} 跨主题", linewidth=1.4)
        ax.set_xlabel("段落数（同主题拼接）")
        ax.set_xticks(SEGMENTS)
        ax.set_ylabel("相似度中位数")
        ax.set_title(f"{name} · 段落粒度的长度曲线（实线同主题，虚线跨主题）", fontsize=11)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, f"fig6_paraphrase_paragraph_{name}.png"), dpi=150)
        plt.close(fig)

    # ---- 汇总图：表达谱 Δ（五模型一图）
    fig, ax = plt.subplots(figsize=(11, 5.5))
    width = 0.16
    xs = np.arange(len(VARIANTS))
    for mi, m in enumerate(models):
        name = m["name"]
        for li, lang in enumerate(["zh", "en"]):
            vals = []
            for v in VARIANTS:
                sub = pdelta[(pdelta["model"] == name) & (pdelta["lang"] == lang) &
                             (pdelta["variant"] == v)]["delta"]
                vals.append(_med(sub) if len(sub) else np.nan)
            ax.bar(xs + (mi * 2 + li - 4.5) * width, vals, width=width,
                   color=plt.cm.tab20(mi * 2 + li), label=f"{name} {lang}")
    ax.axhline(0, color="#4a5568", linewidth=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(VARIANTS, fontsize=10)
    ax.set_ylabel("相对原句的相似度变化（中位）")
    ax.set_title("同一个命题换个说法掉多少分：五模型中英对照", fontsize=12)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=7.5, ncol=5, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig7_paraphrase_delta_all.png"), dpi=150)
    plt.close(fig)

    # ---- 汇总图：段落级长度曲线（五模型一图，实线同主题虚线跨主题）
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, lang in zip(axes, ["zh", "en"]):
        for mi, m in enumerate(models):
            name = m["name"]
            c = plt.cm.tab10(mi)
            sub = cdf[(cdf["model"] == name) & (cdf["lang"] == lang)].sort_values("seg")
            if len(sub):
                ax.plot(sub["seg"], sub["median"], marker="o", color=c, label=name, linewidth=1.5)
            opp = sdf[(sdf["model"] == name) & (sdf["lang"] == lang) &
                      (sdf["pair"].isin(["X1-L1", "X1-L2", "X1-L3", "X1-L4"]))]
            if len(opp):
                curve = opp.groupby("seg_b")["sim"].apply(_med)
                ax.plot(curve.index, curve.values, ls="--", color=c, alpha=0.6, linewidth=1.1)
        ax.set_xticks(SEGMENTS)
        ax.set_xlabel("段落数（同主题拼接）")
        ax.set_ylabel("相似度中位数")
        ax.set_title(f"{'中文' if lang == 'zh' else '英文'}：实线同主题，虚线跨主题", fontsize=11)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("段落级稀释曲线：实线与虚线的间距就是长度差吃掉的主题 gap", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(FIGURES_DIR, "fig8_paragraph_all.png"), dpi=150)
    plt.close(fig)

    print("\n图已写出到 results/figures/", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "analyze":
        analyze()
    else:
        run_all()
