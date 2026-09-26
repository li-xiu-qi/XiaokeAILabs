# -*- coding: utf-8 -*-
"""补做五（2026-09-26）：段落饱和点与功能词的等长归因。

补做四留下两个未闭合，本补做一次做完。

一、段落饱和点。补做四的段落只到 4 段，稀释曲线未找饱和点，而饱和点
   直接决定分块长度上限。本次把粒度推到 1、2、4、8、16 段，测
   「同主题核心 + N 个无关句子」的稀释曲线。选这个构造是因为它对应真实
   检索场景：命中的相关段落被大量不相关上下文包裹。用其他主题的句子做
   尾巴而不是重复同一主题的句子，避免重复 token 造成的假性饱和。

二、功能词的等长归因。补做三的 mask_func 删功能词同时缩短了句子，长度
   与去功能词两个效应方向相反，可能互相抵消，因此不能单独归因。本次改
   为等长替换：把功能词换成等量的无语义占位符（中文「某」，英文 xx），
   长度不变，只动功能词一项。

用法（DGX Spark）：
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ~/xinfer-env/bin/python supplement4.py
  ~/xinfer-env/bin/python supplement4.py analyze
"""

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
import lab  # noqa: E402
from supplement2 import FUNC_ZH, FUNC_EN  # noqa: E402
from themes import THEMES  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

# 段落粒度。16 段中文约 200 字，需确认未被截断，run 阶段打印 token 数。
SEGMENTS = [1, 2, 4, 8, 16]
# 功能词等长替换的替换个数
SWAPS = [1, 2, 4]


# ------------------------------------------------------------ 实验一：段落饱和

def build_saturation_rows():
    """同主题核心 + N 个无关尾巴，加跨主题负对照。

    core 取本主题 a 句，tail 从其余 19 个主题的 a 句按序取，排除自身。
    负对照 rival core 取下一个主题的 a 句，配同一批尾巴。
    """
    rows = []
    n = len(THEMES)
    for pos, t in enumerate(THEMES):
        others = [THEMES[(pos + 1 + i) % n] for i in range(n - 1)]
        for lang, k, joiner in (("zh", "", "。"), ("en", "en_", " ")):
            core = t[k + "a"].rstrip("。")
            tails = [o[k + "a"].rstrip("。") for o in others]
            for nseg in SEGMENTS:
                body = joiner.join([core] + tails[: nseg - 1]) if nseg > 1 else core
                rows.append({"theme_id": t["id"], "lang": lang, "cond": f"core{nseg}",
                             "seg": nseg, "text": body})
            # 负对照：换成 rival 的 core，尾巴同一批
            rival = THEMES[(pos + 1) % n]
            rcore = rival[k + "a"].rstrip("。")
            for nseg in SEGMENTS:
                body = joiner.join([rcore] + tails[: nseg - 1]) if nseg > 1 else rcore
                rows.append({"theme_id": t["id"], "lang": lang, "cond": f"rival{nseg}",
                             "seg": nseg, "text": body})
    return rows


# ------------------------------------------------------- 实验二：功能词等长替换

def swap_func_zh(s: str, n: int):
    """把前 n 个功能词字符替换为「某」，长度不变。返回 (文本, 实际替换数)。

    实际替换数可能小于 n：短句的功能词没那么多。调用方必须按实际替换数分组，
    否则 swap4 在某些主题上只换了 1 个，条件强度不一致。
    """
    out = list(s)
    done = 0
    for i, ch in enumerate(out):
        if done >= n:
            break
        if ch in FUNC_ZH:
            out[i] = "某"
            done += 1
    return "".join(out), done


def swap_func_en(s: str, n: int):
    """把前 n 个功能词替换为 xx，词数不变。返回 (文本, 实际替换数)。"""
    toks = s.split()
    done = 0
    for i, w in enumerate(toks):
        if done >= n:
            break
        if w.lower().strip(".,!?;:") in FUNC_EN:
            toks[i] = "xx"
            done += 1
    return " ".join(toks), done


def build_swap_rows():
    """等长替换行。只动 a 侧，b 侧固定，与 base 比较取 Δ。

    记录 actual_n（真实替换数），实际为 0 的行丢弃：那些句子没有功能词可换，
    留着会让 swap4 与 base 无法区分。
    """
    rows = []
    for t in THEMES:
        for lang, k in (("zh", ""), ("en", "en_")):
            base = t[k + "a"]
            for n in SWAPS:
                if lang == "zh":
                    swapped, actual = swap_func_zh(base, n)
                else:
                    swapped, actual = swap_func_en(base, n)
                if actual == 0:
                    continue
                rows.append({"theme_id": t["id"], "lang": lang, "cond": f"swap{n}",
                             "actual_n": actual, "text_x": swapped, "text_y": t[k + "b"]})
    return rows


# ------------------------------------------------------------------ 计算

def run_saturation(model_name, model, rows):
    """核心对同粒度、核心对各粒度 rival，以及 rival 对各粒度 rival。"""
    by_key = {}
    for r in rows:
        by_key[(r["theme_id"], r["lang"], r["cond"])] = r["text"]
    out = []
    for (tid, lang), g in pd.DataFrame(rows).groupby(["theme_id", "lang"]):
        for nseg in SEGMENTS:
            core = by_key[(tid, lang, f"core{nseg}")]
            # 核心 vs 自身下一档（同主题，长度不等）
            for nseg2 in SEGMENTS:
                if nseg2 <= nseg:
                    continue
                other = by_key[(tid, lang, f"core{nseg2}")]
                out.append({"model": model_name, "lang": lang, "theme_id": tid,
                            "pair": f"same_core{nseg}-core{nseg2}",
                            "seg_a": nseg, "seg_b": nseg2,
                            "text_x": core, "text_y": other})
            # 核心 vs rival 同粒度（纯主题 gap，长度相等）
            rival = by_key[(tid, lang, f"rival{nseg}")]
            out.append({"model": model_name, "lang": lang, "theme_id": tid,
                        "pair": f"gap_core{nseg}-rival{nseg}", "seg_a": nseg, "seg_b": nseg,
                        "text_x": core, "text_y": rival})
    df = pd.DataFrame(out)
    emb = lab.encode(model, list(df["text_x"]) + list(df["text_y"]))
    m = len(df)
    va, vb = emb[:m], emb[m:]
    denom = np.clip(np.linalg.norm(va, axis=1) * np.linalg.norm(vb, axis=1), 1e-12, None)
    df["sim"] = np.sum(va * vb, axis=1) / denom
    return df.drop(columns=["text_x", "text_y"])


def run_pairs(model_name, model, rows):
    """通用文本对打分。"""
    if not rows:
        return pd.DataFrame()
    emb = lab.encode(model, [r["text_x"] for r in rows] + [r["text_y"] for r in rows])
    n = len(rows)
    va, vb = emb[:n], emb[n:]
    denom = np.clip(np.linalg.norm(va, axis=1) * np.linalg.norm(vb, axis=1), 1e-12, None)
    out = []
    for r, s in zip(rows, np.sum(va * vb, axis=1) / denom):
        rec = {"model": model_name, "lang": r["lang"], "theme_id": r["theme_id"],
               "cond": r["cond"], "sim": float(s)}
        if "actual_n" in r:
            rec["actual_n"] = r["actual_n"]
        out.append(rec)
    return pd.DataFrame(out)


def run_swap(model_name, model, rows):
    return run_pairs(model_name, model, rows)


def build_base_rows():
    """base 行：原句对原句，与 swap 条件的 b 侧一致，用于算 Δ。

    不读主实验的 raw_sims.csv：那份数据没同步到 DGX，跨仓取数会让本脚本
    依赖外部状态。base 只有 40 对，重算比搬文件可靠。
    """
    rows = []
    for t in THEMES:
        for lang, k in (("zh", ""), ("en", "en_")):
            rows.append({"theme_id": t["id"], "lang": lang, "cond": "base",
                         "text_x": t[k + "a"], "text_y": t[k + "b"]})
    return rows


def check_truncation(model_name, model, rows):
    """16 段文本是否被模型截断。截断则饱和度读数不可信，必须标注。"""
    long_texts = [r["text"] for r in rows if r["seg"] == 16][:3]
    if not long_texts:
        return None
    tok = getattr(model, "tokenizer", None)
    if tok is None:
        return None
    limit = getattr(model, "max_seq_length", 512)
    counts = [len(tok(t)["input_ids"]) for t in long_texts]
    return {"model": model_name, "limit": limit,
            "max_tokens": max(counts), "truncated": max(counts) > limit}


def run_all():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    sat_rows = build_saturation_rows()
    swap_rows = build_swap_rows()
    base_rows = build_base_rows()
    print(f"段落行数 {len(sat_rows)}，等长替换行数 {len(swap_rows)}，base 行数 {len(base_rows)}",
          flush=True)

    sat_frames, swap_frames, base_frames, trunc_info = [], [], [], []
    for m in config.MODELS:
        name = m["name"]
        t0 = time.time()
        print(f"[{name}] 加载 ...", flush=True)
        model = lab.load_model(m["hf_id"])
        try:
            info = check_truncation(name, model, sat_rows)
            if info:
                trunc_info.append(info)
                print(f"[{name}] 16 段最长 {info['max_tokens']} token，上限 {info['limit']}，"
                      f"{'已截断' if info['truncated'] else '未截断'}", flush=True)
            print(f"[{name}] 实验一 段落饱和 ...", flush=True)
            sat_frames.append(run_saturation(name, model, sat_rows))
            print(f"[{name}] 实验二 功能词等长替换（含 base）...", flush=True)
            swap_frames.append(run_swap(name, model, swap_rows))
            base_frames.append(run_pairs(name, model, base_rows))
            print(f"[{name}] 完成 {time.time() - t0:.0f}s", flush=True)
        finally:
            del model

    pd.concat(sat_frames, ignore_index=True).to_csv(
        os.path.join(RESULTS_DIR, "raw_saturation.csv"), index=False, encoding="utf-8-sig")
    pd.concat(swap_frames, ignore_index=True).to_csv(
        os.path.join(RESULTS_DIR, "raw_func_swap.csv"), index=False, encoding="utf-8-sig")
    pd.concat(base_frames, ignore_index=True).to_csv(
        os.path.join(RESULTS_DIR, "raw_func_base.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(trunc_info).to_csv(
        os.path.join(RESULTS_DIR, "truncation_check.csv"), index=False, encoding="utf-8-sig")
    print("写出四张表", flush=True)


def _med(series):
    return float(np.median(series.values)) if len(series) else float("nan")


def analyze():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "AR PL UMing CN", "SimHei", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False

    sat = pd.read_csv(os.path.join(RESULTS_DIR, "raw_saturation.csv"))
    swp = pd.read_csv(os.path.join(RESULTS_DIR, "raw_func_swap.csv"))
    trunc = pd.read_csv(os.path.join(RESULTS_DIR, "truncation_check.csv"))

    print("=== 截断自检 ===")
    print(trunc.to_string(index=False) if len(trunc) else "无数据")

    # 稀释曲线：core1 对各档
    print("\n=== 段落饱和：core1 对 coreK 的相似度中位 ===")
    c1 = sat[sat["pair"].str.match(r"same_core1-core")].copy()
    c1["k"] = c1["pair"].str.extract(r"core(\d+)$").astype(int)
    tab = c1.pivot_table(index="model", columns=["lang", "k"], values="sim", aggfunc=_med)
    print(tab.round(3).to_string())
    # 逐档边际下降，找饱和点（边际降幅小于 0.01 视为饱和）
    print("\n=== 逐档边际下降（负值越小越接近饱和）===")
    marg = []
    for (model, lang), g in c1.groupby(["model", "lang"]):
        g = g.sort_values("k")
        ks = list(g["k"])
        vs = list(g["sim"])
        for i in range(1, len(ks)):
            marg.append({"model": model, "lang": lang, "from": ks[i - 1], "to": ks[i],
                         "marginal": vs[i] - vs[i - 1]})
    mdf = pd.DataFrame(marg)
    print(mdf.pivot_table(index="model", columns=["lang", "to"], values="marginal").round(4).to_string())
    mdf.to_csv(os.path.join(RESULTS_DIR, "saturation_marginal.csv"),
               index=False, encoding="utf-8-sig")

    # 纯主题 gap 随段数变化
    print("\n=== 纯主题 gap（core K 减 rival K，长度相等）===")
    gp = sat[sat["pair"].str.match(r"gap_core")].copy()
    gp["k"] = gp["pair"].str.extract(r"core(\d+)").astype(int)
    print(gp.pivot_table(index="model", columns=["lang", "k"], values="sim", aggfunc=_med)
          .round(3).to_string())
    gp.to_csv(os.path.join(RESULTS_DIR, "saturation_gap.csv"),
              index=False, encoding="utf-8-sig")

    # 功能词等长替换。按实际替换数分组，因为请求替换数在不同主题上
    # 真正换掉的个数不同（短句功能词不够），混在一起会得到假的梯度。
    print("\n=== 可替换功能词数的分布（决定各档实际样本量）===")
    print(swp.groupby(["lang", "cond", "actual_n"]).size().to_string())

    print("\n=== 等长替换的绝对相似度 ===")
    for n in SWAPS:
        sub = swp[swp["cond"] == f"swap{n}"]
        line = "  ".join(f"{m}/{l}:{_med(g['sim']):.3f}"
                         for (m, l), g in sub.groupby(["model", "lang"]))
        print(f"swap{n}: {line}")

    # Δ 需要 base。base 由本脚本自算（raw_func_base.csv），不跨仓取数
    delta_rows = []
    base_csv = os.path.join(RESULTS_DIR, "raw_func_base.csv")
    if os.path.exists(base_csv):
        bs = pd.read_csv(base_csv)
        bmap = {(r["model"], r["lang"], r["theme_id"]): r["sim"] for _, r in bs.iterrows()}
        for _, r in swp.iterrows():
            key = (r["model"], r["lang"], r["theme_id"])
            if key in bmap:
                delta_rows.append({"model": r["model"], "lang": r["lang"],
                                   "actual_n": r["actual_n"],
                                   "delta": r["sim"] - bmap[key]})
    else:
        print("\n未找到 raw_func_base.csv，先跑推理", flush=True)
    ddf = pd.DataFrame(delta_rows)
    if not ddf.empty:
        print("\n=== 等长替换相对原句的 Δ（按实际替换数）===")
        print(ddf.pivot_table(index="model", columns=["lang", "actual_n"],
                              values="delta", aggfunc=_med).round(4).to_string())
        ddf.to_csv(os.path.join(RESULTS_DIR, "func_swap_delta.csv"),
                   index=False, encoding="utf-8-sig")
    else:
        print("\n未找到主实验 base 数据，Δ 无法计算", flush=True)

    # ---- 图：饱和曲线。必须先取中位数再画：原始行每 k 有 20 个主题，
    #      直接 plot 会画成一簇散点，看不出曲线形状。
    c1_med = c1.pivot_table(index=["model", "lang"], columns="k", values="sim", aggfunc=_med)
    gp_med = gp.pivot_table(index=["model", "lang"], columns="k", values="sim", aggfunc=_med)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, lang in zip(axes, ["zh", "en"]):
        for mi, m in enumerate(config.MODELS):
            name = m["name"]
            key = (name, lang)
            if key in c1_med.index:
                ax.plot(c1_med.columns, c1_med.loc[key].values, marker="o",
                        color=plt.cm.tab10(mi), label=name, linewidth=1.6)
            if key in gp_med.index:
                ax.plot(gp_med.columns, gp_med.loc[key].values, ls="--",
                        color=plt.cm.tab10(mi), alpha=0.6, linewidth=1.2)
        ax.set_xscale("log", base=2)
        ax.set_xticks(SEGMENTS)
        ax.set_xticklabels(SEGMENTS)
        ax.set_xlabel("段数（同主题核心 + 无关尾巴）")
        ax.set_ylabel("相似度中位数")
        ax.set_title(f"{'中文' if lang == 'zh' else '英文'}：实线 core1-coreK，虚线 coreK-rivalK",
                     fontsize=11)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("段落稀释曲线推至 16 段：实线降是稀释，虚线升是主题可分性", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(FIGURES_DIR, "fig9_saturation.png"), dpi=150)
    plt.close(fig)

    # ---- 图：功能词等长替换
    if not ddf.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        width = 0.16
        actuals = sorted(ddf["actual_n"].unique())
        xs = np.arange(len(actuals))
        for mi, m in enumerate(config.MODELS):
            name = m["name"]
            for li, lang in enumerate(["zh", "en"]):
                vals = []
                for a in actuals:
                    v = ddf[(ddf["model"] == name) & (ddf["lang"] == lang) &
                            (ddf["actual_n"] == a)]["delta"]
                    vals.append(_med(v) if len(v) else np.nan)
                ax.bar(xs + (mi * 2 + li - 4.5) * width, vals, width=width,
                       color=plt.cm.tab20(mi * 2 + li), label=f"{name} {lang}")
        ax.axhline(0, color="#4a5568", linewidth=0.8)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"实际替换 {a} 个" for a in actuals], fontsize=10)
        ax.set_ylabel("相对原句的相似度变化")
        ax.set_title("功能词的等长归因：长度不变，只把功能词换成占位符", fontsize=12)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=7.5, ncol=5, frameon=False, loc="lower left")
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, "fig10_func_swap.png"), dpi=150)
        plt.close(fig)

    print("\n图已写出", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "analyze":
        analyze()
    else:
        run_all()
