# -*- coding: utf-8 -*-
"""补做实验（2026-09-24）：负对照 gap + 退火机制探针。

动机：主实验缺负对照，无法把 base sim 整体下移区分为「语义理解退化」与
「相似度校准偏移」；退火的机制（表面共性 vs 语义不可扰动）停留在推断。

补做一 N：跨主题无关句对的 similarity，与主实验同主题 base 配对算
discrimination gap = sim(相关) - sim(无关)。
  - gap 在某种语言场景塌陷 → 模型对该语言无法分辨相关与无关
  - 中文专用模型在英文场景的 gap 是否塌陷 = 训练数据是否含英文的行为探针

补做二 M：irr_both 的噪声句换成「词序/字序打乱的乱文本」（保留字符与词
分布，破坏句法与语义）。比较 Δ(乱文本) 与 Δ(原噪声句)：
  - 乱文本效应显著更负 → 原噪声句的句法表层共性贡献了正向成分
  - 两者接近 → 退火与噪声句性质无关，指向语义不可扰动

用法（DGX Spark）：
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ~/xinfer-env/bin/python supplement.py
  ~/xinfer-env/bin/python supplement.py analyze
"""

import os
import random
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
import lab  # noqa: E402


def build_negative_pairs() -> list:
    """跨主题无关句对：全部 C(20,2) = 190 组合，取每主题 a 侧文本。"""
    from themes import THEMES

    rows = []
    for i, ti in enumerate(THEMES):
        for j, tj in enumerate(THEMES):
            if i >= j:
                continue
            rows.append({
                "theme_a": ti["id"], "theme_b": tj["id"],
                "zh_zh": (ti["a"], tj["a"]),
                "en_en": (ti["en_a"], tj["en_a"]),
                "zh_en": (ti["a"], tj["en_a"]),
            })
    return rows


def shuffle_text(s: str, lang: str, rng: random.Random) -> str:
    """保留字符/词分布，破坏顺序与结构。中文按字打乱，英文按词打乱。"""
    body = s.strip().rstrip("。.")
    if lang == "zh":
        chars = [c for c in body if not c.isspace()]
        rng.shuffle(chars)
        return "".join(chars) + "。"
    words = body.split()
    rng.shuffle(words)
    return " ".join(words) + "."


def build_shuffled_pairs(seed: int = 20260924) -> list:
    """irr_both 变体：两侧噪声句换成乱文本。"""
    from themes import THEMES, noise_a_zh, noise_b_zh, noise_a_en, noise_b_en

    rng = random.Random(seed)
    rows = []
    for idx, t in enumerate(THEMES):
        nz_a, nz_b = noise_a_zh(idx), noise_b_zh(idx)
        ne_a, ne_b = noise_a_en(idx), noise_b_en(idx)
        zh_first = {
            "base": t["a"], "irr_both": t["a"] + shuffle_text(nz_a, "zh", rng),
        }
        zh_second = {
            "base": t["b"], "irr_both": t["b"] + shuffle_text(nz_b, "zh", rng),
        }
        en_first = {
            "base": t["en_a"], "irr_both": t["en_a"] + " " + shuffle_text(ne_a, "en", rng),
        }
        en_second = {
            "base": t["en_b"], "irr_both": t["en_b"] + " " + shuffle_text(ne_b, "en", rng),
        }
        for lp, (first, second) in {
            "zh-zh": (zh_first, zh_second),
            "en-en": (en_first, en_second),
            "zh-en": (zh_first, en_second),
        }.items():
            rows.append({
                "theme_id": t["id"], "lang_pair": lp,
                "base_a": first["base"], "base_b": second["base"],
                "shuf_a": first["irr_both"], "shuf_b": second["irr_both"],
            })
    return rows


def run_inference() -> None:
    """跑两个补做，落盘 raw_negative.csv 与 raw_shuffled.csv。"""
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    neg = build_negative_pairs()
    print(f"negative pairs: {len(neg)}", flush=True)

    rng_shuf = build_shuffled_pairs()
    print(f"shuffled rows: {len(rng_shuf)}", flush=True)

    # 预构造两批编码文本列表（所有模型共用同一批）
    neg_texts_a, neg_texts_b, neg_lang = [], [], []
    for r in neg:
        for lp in config.LANG_PAIRS:
            a, b = r[lp.replace("-", "_")]
            neg_texts_a.append(a)
            neg_texts_b.append(b)
            neg_lang.append(lp)

    shuf_a, shuf_b, shuf_lang, shuf_theme = [], [], [], []
    base_a, base_b = [], []
    for r in rng_shuf:
        shuf_a.append(r["shuf_a"])
        shuf_b.append(r["shuf_b"])
        base_a.append(r["base_a"])
        base_b.append(r["base_b"])
        shuf_lang.append(r["lang_pair"])
        shuf_theme.append(r["theme_id"])

    neg_rows, shuf_rows = [], []
    for spec in config.MODELS:
        t0 = time.time()
        try:
            model = lab.load_model(spec["hf_id"])

            e_na = lab.encode(model, neg_texts_a)
            e_nb = lab.encode(model, neg_texts_b)
            neg_sims = np.sum(e_na * e_nb, axis=1) / np.clip(
                np.linalg.norm(e_na, axis=1) * np.linalg.norm(e_nb, axis=1), 1e-12, None
            )
            for lang_val, a, b, s in zip(neg_lang, neg_texts_a, neg_texts_b, neg_sims):
                neg_rows.append({
                    "model": spec["name"], "lang": spec["lang"], "lang_pair": lang_val,
                    "text_a": a, "text_b": b, "sim": float(s),
                })

            e_sa = lab.encode(model, shuf_a)
            e_sb = lab.encode(model, shuf_b)
            e_ba = lab.encode(model, base_a)
            e_bb = lab.encode(model, base_b)

            def cos(x, y):
                return np.sum(x * y, axis=1) / np.clip(
                    np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1), 1e-12, None
                )

            sim_shuf = cos(e_sa, e_sb)
            sim_base = cos(e_ba, e_bb)
            for lang_val, theme, sb, bs in zip(shuf_lang, shuf_theme, sim_shuf, sim_base):
                shuf_rows.append({
                    "model": spec["name"], "lang": spec["lang"], "lang_pair": lang_val,
                    "theme_id": theme, "sim_shuffled": float(sb), "sim_base": float(bs),
                    "delta_shuffled": float(sb - bs),
                })
            print(f"{spec['name']:14s} done in {time.time() - t0:.1f}s", flush=True)
            del model, e_na, e_nb, e_sa, e_sb, e_ba, e_bb
        except Exception as exc:
            print(f"{spec['name']:14s} FAILED: {exc}", flush=True)

    pd.DataFrame(neg_rows).to_csv(
        os.path.join(config.RESULTS_DIR, "raw_negative.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(shuf_rows).to_csv(
        os.path.join(config.RESULTS_DIR, "raw_shuffled.csv"), index=False, encoding="utf-8-sig")
    print("saved raw_negative.csv / raw_shuffled.csv")


def analyze() -> None:
    from scipy import stats

    neg = pd.read_csv(os.path.join(config.RESULTS_DIR, "raw_negative.csv"))
    shuf = pd.read_csv(os.path.join(config.RESULTS_DIR, "raw_shuffled.csv"))
    main_base = pd.read_csv(os.path.join(config.RESULTS_DIR, "raw_sims.csv"))
    main_base = main_base[main_base.condition == "base"]

    print("=== 补做一：discrimination gap = sim(同主题相关) - sim(跨主题无关) ===")
    rows = []
    for (model, lp), g in neg.groupby(["model", "lang_pair"]):
        neg_sim = g["sim"].values
        pos_sim = main_base[(main_base.model == model) & (main_base.lang_pair == lp)]["sim"].values
        gap = float(np.median(pos_sim) - np.median(neg_sim))
        # 正负分布分离度（Mann-Whitney AUC）
        auc = stats.mannwhitneyu(pos_sim, neg_sim, alternative="greater").statistic / (
            len(pos_sim) * len(neg_sim)
        )
        rows.append({
            "model": model, "lang_pair": lp,
            "pos_median": float(np.median(pos_sim)),
            "neg_median": float(np.median(neg_sim)),
            "gap": gap, "auc": float(auc),
        })
    gap_df = pd.DataFrame(rows)
    print(gap_df.pivot_table(index="model", columns="lang_pair", values="gap").round(4).to_string())
    print()
    print("分离度 AUC（0.5=完全分不出，1=完全分开）:")
    print(gap_df.pivot_table(index="model", columns="lang_pair", values="auc").round(3).to_string())
    gap_df.to_csv(os.path.join(config.RESULTS_DIR, "gap_summary.csv"), index=False, encoding="utf-8-sig")

    print()
    print("=== 补做二：乱文本 vs 原噪声句 的 irr_both Δ ===")
    summ = pd.read_csv(os.path.join(config.RESULTS_DIR, "summary_delta.csv"))
    rows = []
    for (model, lp), g in shuf.groupby(["model", "lang_pair"]):
        d_shuf = g["delta_shuffled"].values
        orig = summ[(summ.model == model) & (summ.lang_pair == lp) & (summ.condition == "irr_both")]
        d_orig_median = float(orig["median_delta"].iloc[0]) if len(orig) else np.nan
        rows.append({
            "model": model, "lang_pair": lp,
            "delta_shuffled_median": float(np.median(d_shuf)),
            "delta_original_noise_median": d_orig_median,
            "diff": float(np.median(d_shuf)) - d_orig_median,
        })
    cmp_df = pd.DataFrame(rows)
    print(cmp_df.pivot_table(index="model", columns="lang_pair", values="delta_shuffled_median").round(4).to_string())
    print()
    print("乱文本减原噪声句（负=乱文本更不像，即原噪声句的表层共性贡献正向）:")
    print(cmp_df.pivot_table(index="model", columns="lang_pair", values="diff").round(4).to_string())
    cmp_df.to_csv(os.path.join(config.RESULTS_DIR, "shuffle_compare.csv"), index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "analyze":
        analyze()
    else:
        run_inference()
