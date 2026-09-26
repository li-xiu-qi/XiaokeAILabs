# -*- coding: utf-8 -*-
"""补做三（2026-09-26）：用词因素。

动机：主实验把长度与内容绑在一起（irr_both 同时改变两者），补做二只探了
退火机制。用词因素在主实验与两次补做里都是空白，而它是相似度分数最直接
的可控输入之一。本补做把「长度」与「内容」拆开，并补上功能词、同义替换、
关键词重叠三个用词维度。

设计：只动 a 侧，b 侧固定，全部条件与 base(a vs b) 比较取 Δ。

  mask_func  a 删去功能词                      验证 2212.09170 功能词搭车量
  syn        a 的核心实词换同义词（词级，等长）  测用词保真度
  kw1/2/3    a 追加 b 的 1/2/3 个实词（词形重叠） 关键词重叠的边际效应
  func2/4/8  a 追加 2/4/8 个纯功能词            只有长度变化、无新语义
  noise4     a 追加 4 个实词构成的无关片段       与 func4 等词数、有新语义

核心对照：
  1. func4 vs noise4：词数相同，一个有语义一个无语义。两者 Δ 的差就是
     「内容」从「长度」里分离出来的部分，主实验缺失的正是这个分离。
  2. kw1 到 kw3 的梯度：词形重叠能抵消多少语义差异。
  3. mask_func：功能词对相似度的贡献方向与幅度。

局限：syn 的替换对是人工构造的语义近似词，不等价于严格同义；kw 注入是
非自然拼接，作为探针使用，结论限于「词形重叠存在边际效应」这一层。

用法（DGX Spark）：
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ~/xinfer-env/bin/python supplement2.py
  ~/xinfer-env/bin/python supplement2.py analyze
"""

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
import lab  # noqa: E402

# DGX 上脚本与 results 同级（config 的 RESULTS_DIR 按仓内 code/ 布局解析，此处覆盖）
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# 功能词表。中文按字删，英文按词删。
FUNC_ZH = list("的了是在有和就都也很与及并而被把对从到上下里外这那什么如果因为所以但是而且虽然")
FUNC_EN = [
    "the", "a", "an", "is", "are", "of", "to", "in", "on", "and", "or", "but",
    "with", "for", "from", "by", "at", "as", "that", "this", "it", "its",
]

# 每主题一个核心实词的同义替换（人工构造，语义近似，不引入新主题词）。
SYN = {
    "pet": ("小狗", "犬类", "dog", "canine"),
    "programming": ("编程", "写代码", "programming", "coding"),
    "exam": ("考试", "测验", "exam", "test"),
    "health": ("运动", "锻炼", "Exercise", "workout"),
    "weather": ("下雨", "降雨", "raining", "showering"),
    "transit": ("地铁", "轨道交通", "subway", "metro"),
    "gadget": ("手机", "移动电话", "phone", "cellphone"),
    "books": ("情节", "故事", "plot", "storyline"),
    "work": ("项目", "任务", "project", "task"),
    "music_art": ("曲子", "乐曲", "melody", "tune"),
    "travel": ("风景", "景色", "scenery", "landscape"),
    "cooking": ("菜", "菜品", "dish", "course"),
    "fitness": ("跑步", "奔跑", "Running", "Jogging"),
    "plants": ("茂盛", "繁盛", "blooming", "flourishing"),
    "debug": ("bug", "缺陷", "bug", "defect"),
    "classroom": ("老师", "教师", "teacher", "instructor"),
    "checkup": ("体检", "健康检查", "checkups", "screenings"),
    "eco": ("关灯", "熄灯", "Switching off", "Turning off"),
    "cinema": ("电影", "影片", "movie", "film"),
    "gaming": ("游戏", "电玩", "game", "videogame"),
}

# 无关实词片段池（从主实验噪声池截取，4 个词一组，与全部主题无关）。
NOISE_WORDS_ZH = ["会议", "交通", "天气", "音乐", "植物", "烹饪", "旅行", "健身"]
NOISE_WORDS_EN = ["meeting", "traffic", "weather", "music", "plants", "cooking", "travel", "workout"]


def mask_func_zh(s: str) -> str:
    return "".join(c for c in s if c not in FUNC_ZH and not c.isspace())


def mask_func_en(s: str) -> str:
    keep = [w for w in s.strip().rstrip(".").split() if w.lower() not in FUNC_EN]
    return " ".join(keep) + "."


def content_words_en(s: str) -> list:
    return [w for w in s.strip().rstrip(".").split() if w.lower() not in FUNC_EN]


def content_chars_zh(s: str) -> str:
    return "".join(c for c in s if c not in FUNC_ZH and not c.isspace())


def build_rows() -> list:
    """每个主题每个语言对产出全部条件的文本对。"""
    from themes import THEMES

    rows = []
    for idx, t in enumerate(THEMES):
        syn = SYN.get(t["id"], ("", "", "", ""))
        # 中文侧
        a_zh, b_zh = t["a"], t["b"]
        kw_pool_zh = content_chars_zh(b_zh)
        zh_variants = {
            "base": a_zh,
            "mask_func": mask_func_zh(a_zh),
            "syn": a_zh.replace(syn[0], syn[1]) if syn[0] else a_zh,
            "kw1": a_zh + kw_pool_zh[:2],
            "kw2": a_zh + kw_pool_zh[:4],
            "kw3": a_zh + kw_pool_zh[:6],
            "func2": a_zh + "".join(FUNC_ZH[:2]),
            "func4": a_zh + "".join(FUNC_ZH[:4]),
            "func8": a_zh + "".join(FUNC_ZH[:8]),
            "noise4": a_zh + "".join(NOISE_WORDS_ZH[(idx + k) % len(NOISE_WORDS_ZH)] for k in range(4)),
        }
        # 英文侧
        a_en, b_en = t["en_a"], t["en_b"]
        kw_pool_en = content_words_en(b_en)
        en_variants = {
            "base": a_en,
            "mask_func": mask_func_en(a_en),
            "syn": a_en.replace(syn[2], syn[3]) if syn[2] else a_en,
            "kw1": a_en + " " + " ".join(kw_pool_en[:1]),
            "kw2": a_en + " " + " ".join(kw_pool_en[:2]),
            "kw3": a_en + " " + " ".join(kw_pool_en[:3]),
            "func2": a_en + " " + " ".join(FUNC_EN[:2]),
            "func4": a_en + " " + " ".join(FUNC_EN[:4]),
            "func8": a_en + " " + " ".join(FUNC_EN[:8]),
            "noise4": a_en + " " + " ".join(NOISE_WORDS_EN[(idx + k) % len(NOISE_WORDS_EN)] for k in range(4)),
        }
        for lp, (first, second) in {
            "zh-zh": (zh_variants, b_zh),
            "en-en": (en_variants, b_en),
            "zh-en": (zh_variants, b_en),
        }.items():
            for cond, text in first.items():
                rows.append({
                    "theme_id": t["id"], "lang_pair": lp, "condition": cond,
                    "text_a": text, "text_b": second,
                })
    return rows


def run_inference() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = build_rows()
    print(f"wording rows: {len(rows)}", flush=True)

    texts_a = [r["text_a"] for r in rows]
    texts_b = [r["text_b"] for r in rows]

    out_rows = []
    for spec in config.MODELS:
        t0 = time.time()
        try:
            model = lab.load_model(spec["hf_id"])
            e_a = lab.encode(model, texts_a)
            e_b = lab.encode(model, texts_b)
            sims = np.sum(e_a * e_b, axis=1) / np.clip(
                np.linalg.norm(e_a, axis=1) * np.linalg.norm(e_b, axis=1), 1e-12, None
            )
            for r, s in zip(rows, sims):
                out_rows.append({
                    "model": spec["name"], "lang": spec["lang"],
                    "lang_pair": r["lang_pair"], "theme_id": r["theme_id"],
                    "condition": r["condition"], "sim": float(s),
                    "text_a": r["text_a"], "text_b": r["text_b"],
                })
            print(f"{spec['name']:14s} done in {time.time() - t0:.1f}s", flush=True)
            del model, e_a, e_b
        except Exception as exc:
            print(f"{spec['name']:14s} FAILED: {exc}", flush=True)

    pd.DataFrame(out_rows).to_csv(
        os.path.join(RESULTS_DIR, "raw_wording.csv"), index=False, encoding="utf-8-sig")
    print("saved raw_wording.csv")


def analyze() -> None:
    df = pd.read_csv(os.path.join(RESULTS_DIR, "raw_wording.csv"))
    base = df[df.condition == "base"].set_index(["model", "lang_pair", "theme_id"])["sim"]

    df["sim_base"] = df.set_index(["model", "lang_pair", "theme_id"]).index.map(base)
    df["delta"] = df["sim"] - df["sim_base"]

    print("=== 补做三：用词因素，各条件 Δ 中位数（相对 base，只动 a 侧）===")
    order = ["mask_func", "syn", "kw1", "kw2", "kw3",
             "func2", "func4", "func8", "noise4"]
    for lp in config.LANG_PAIRS:
        sub = df[(df.lang_pair == lp) & (df.condition != "base")]
        piv = sub.pivot_table(index="model", columns="condition", values="delta", aggfunc="median")
        piv = piv[[c for c in order if c in piv.columns]]
        print(f"\n--- {lp} ---")
        print(piv.round(4).to_string())

    print("\n=== 核心对照：func4（纯功能词追加）减 noise4（实词噪声追加）===")
    print("负值表示实词噪声比功能词更能拉低相似度，即内容效应大于纯长度效应")
    for lp in config.LANG_PAIRS:
        sub = df[(df.lang_pair == lp) & (df.condition.isin(["func4", "noise4"]))]
        piv = sub.pivot_table(index="model", columns="condition", values="delta", aggfunc="median")
        if {"func4", "noise4"}.issubset(piv.columns):
            diff = (piv["func4"] - piv["noise4"]).round(4)
            print(f"\n--- {lp} ---")
            print(diff.to_string())

    summ = df[df.condition != "base"].groupby(
        ["model", "lang_pair", "condition"])["delta"].median().reset_index()
    summ.to_csv(os.path.join(RESULTS_DIR, "wording_summary.csv"),
                index=False, encoding="utf-8-sig")
    print("\nsaved wording_summary.csv")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "analyze":
        analyze()
    else:
        run_inference()
