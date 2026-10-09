# -*- coding: utf-8 -*-
"""两个损失函数的数学对比（Obsidian 笔记配图，白底学术风）

图 A：strictly proper scoring rule 的理论性质。奖励对真实结果 Y 求期望时，
      期望得分在 q = p（真实概率）处唯一取最大。这是 RLCD 校准目标的理论根基。
图 B：laya 实现里 proper_reward 的 target 是教师软分布 t（不是对 Y 求期望），
      reward 三项（log score + spherical + rps）全部在 q = t 处最优，
      与 soft CE 的最优解重合。策略梯度因此只是同一目标的随机重加权。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---------------------------------------------------------------- 字体
FONT_PATH = ("C:/Users/ke/Documents/projects/obsidian_projects/pkm-hub-skills/"
             "skills/_global/figure-routing/examples/fonts/PingFangSC-Regular.ttf")
font_manager.fontManager.addfont(FONT_PATH)

plt.rcParams.update({
    "font.family": font_manager.FontProperties(fname=FONT_PATH).get_name(),
    "mathtext.fontset": "dejavusans",
    "mathtext.default": "regular",
    "axes.unicode_minus": False,
    "figure.dpi": 150,
    "savefig.dpi": 150,
})

INK = "#2a2f3a"
BLUE = "#3b82c4"
RED = "#c0392b"
GREEN = "#2e8b57"
GREY_TXT = "#7a828e"

OUT = ("C:/Users/ke/Documents/projects/CodeProjects/Windows-Native/XiaokeAILabs-suite/"
       "XiaokeAILabs/experiments/test_jev_open_source/laya_rlcd_train/_diagrams/")

q = np.linspace(0.02, 0.98, 400)


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    ax.tick_params(colors=INK, labelsize=10)
    ax.grid(True, color="#e8eaee", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


# ================================================================ 图 A
# 真实概率 p，奖励对 Y ~ Bernoulli(p) 求期望
p = 0.7
log_score = p * np.log(q) + (1 - p) * np.log(1 - q)
sph = (p * q + (1 - p) * (1 - q)) / np.sqrt(q ** 2 + (1 - q) ** 2)
brier_neg = -((1 - q) ** 2 * p + q ** 2 * (1 - p))

fig, ax = plt.subplots(figsize=(8.6, 5.0))
style_ax(ax)
ax.plot(q, log_score, color=BLUE, lw=2.0, zorder=3,
        label=r"对数得分  $\mathbb{E}[\,\ln q_Y\,] = p\ln q + (1-p)\ln(1-q)$")
ax.plot(q, sph, color=RED, lw=2.0, zorder=3,
        label=r"球形得分  $\mathbb{E}[\,q_Y/\|q\|\,]$")
ax.plot(q, brier_neg, color=GREEN, lw=2.0, zorder=3,
        label=r"负 Brier 得分  $-\mathbb{E}[(q_Y - q)^2]$")

ax.axvline(p, color=GREY_TXT, lw=1.2, ls=(0, (4, 3)), zorder=2)
ax.annotate(r"$q^{*} = p$", xy=(p, ax.get_ylim()[1] * 0.97),
            xytext=(p + 0.06, ax.get_ylim()[1] * 0.97),
            fontsize=12, color=INK, va="top")
for y, name, c in [(log_score[np.argmin(np.abs(q - p))], "三条同点", BLUE)]:
    pass
ax.plot([p], [p * np.log(p) + (1 - p) * np.log(1 - p)],
        marker="o", ms=7, color=INK, zorder=5)
ax.plot([p], [(p * p + (1 - p) * (1 - p)) / np.sqrt(2 * p ** 2)],
        marker="o", ms=7, color=INK, zorder=5)
ax.plot([p], [-(2 * p * (1 - p))], marker="o", ms=7, color=INK, zorder=5)

ax.set_xlabel(r"模型报告的概率  $q$", fontsize=11.5, color=INK)
ax.set_ylabel(r"期望得分（越大越好）", fontsize=11.5, color=INK)
ax.set_title(r"严格适当评分规则：期望得分在 $q=p$ 处唯一取最大（真实概率 $p=0.7$）",
             fontsize=12.5, color=INK, pad=12)
ax.legend(fontsize=10, frameon=False, loc="lower left")
ax.set_xlim(0, 1)
fig.text(0.5, 0.015,
         "strictly proper 的数学定义：对任意 p，期望得分作为 q 的函数在 q=p 处有唯一全局最大。这是 RLCD「校准」目标的理论根基。",
         ha="center", va="bottom", fontsize=9.5, color=GREY_TXT)
fig.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig(OUT + "proper-scoring-rule期望得分曲线.png", facecolor="white")
plt.close(fig)
print("saved A")

# ================================================================ 图 B
# 教师软分布 t。laya 的 reward 主项 log_score = Σ tᵢln qᵢ，正是 soft CE 损失取负：
# 同一函数、同一最优解。soft CE 最小化它，reward 最大化它。
# 辅助项 spherical（权重 0.75）与 rps（权重 1.0）只微调曲线形状，不改变主项的主导地位。
t = 0.9
log_score_reward = t * np.log(q) + (1 - t) * np.log(1 - q)   # reward 主项（越大越好）
soft_ce_loss = -log_score_reward                              # soft CE 损失（越小越好）
q_star = t

fig, ax = plt.subplots(figsize=(8.6, 5.0))
style_ax(ax)
ax.plot(q, soft_ce_loss, color=BLUE, lw=2.2, zorder=3,
        label=r"soft CE 损失  $-\sum t_i \ln q_i$（越小越好）")
ax.plot(q, log_score_reward, color=RED, lw=2.2, zorder=3,
        label=r"reward 主项  $\sum t_i \ln q_i$（越大越好）")
ax.set_xlabel(r"模型输出的概率  $q$", fontsize=11.5, color=INK)
ax.set_ylabel(r"函数值", fontsize=11.5, color=INK)
ax.set_xlim(0, 1)
ax.axhline(0, color="#c8ccd4", lw=1.0, zorder=1)

ax.axvline(q_star, color=INK, lw=1.4, ls=(0, (4, 3)), zorder=2)
ax.annotate(r"两条曲线关于 0 轴对称：同一函数差一个负号", xy=(q_star, 0),
            xytext=(0.06, -1.6), fontsize=11.5, color=INK,
            arrowprops=dict(arrowstyle="->", color=INK, lw=1.1))
ax.annotate(r"$q^{*}=t$", xy=(q_star, ax.get_ylim()[1] * 0.9),
            xytext=(q_star + 0.05, ax.get_ylim()[1] * 0.9),
            fontsize=12, color=INK, va="top")
ax.plot([q_star], [0], marker="o", ms=8, color=INK, zorder=5)

ax.legend(fontsize=10, frameon=False, loc="upper left")
ax.set_title(r"laya 的 reward 主项就是 soft CE 取负：同一目标函数，最优解严格重合（$t=0.9$）",
             fontsize=12.5, color=INK, pad=12)
fig.text(0.5, 0.015,
         "proper_reward 的 target 是教师软分布（参数）而非真实结果的采样期望；spherical（0.75）与 rps（1.0）为辅助项，只微调形状不改主项主导。",
         ha="center", va="bottom", fontsize=9.5, color=GREY_TXT)
fig.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig(OUT + "laya-reward与soft-CE同最优解.png", facecolor="white")
plt.close(fig)
print("saved B")
