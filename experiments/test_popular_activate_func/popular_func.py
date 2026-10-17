#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
常用激活函数可视化与详解
============================

以脚本形式复现「激活函数曲线 + 导数 + 特性对比」的教学内容。

涵盖 8 个常用激活函数及其导数：
    Linear / Sigmoid / Tanh / ReLU / Leaky ReLU / ELU / Swish / GELU

逐个绘制函数图像与导数图像，末尾打印「激活函数特性对比表」并给出选择指南。

依赖: numpy, matplotlib, scipy, pandas
运行: python popular_func.py
      （需要图形界面查看图像；若在无显示环境运行，把各 plot_pair 里的
       plt.show() 改为 fig.savefig("figures/xxx.png") 即可落盘）
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# 中文字体与本图样式（Windows 下用 SimHei；其他系统可改为 "DejaVu Sans" 或 "Arial"）
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 12

X = np.linspace(-5, 5, 1000)


# ---------- 激活函数 ----------
def linear(x):
    return x


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def tanh(x):
    return np.tanh(x)


def relu(x):
    return np.maximum(0, x)


def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)


def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))


def swish(x):
    return x * sigmoid(x)


def gelu(x):
    return 0.5 * x * (1 + erf(x / np.sqrt(2)))


# ---------- 导数 ----------
def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2


def relu_derivative(x):
    return np.where(x > 0, 1, 0)


def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)


def elu_derivative(x, alpha=1.0):
    return np.where(x > 0, 1, alpha * np.exp(x))


def swish_derivative(x):
    s = sigmoid(x)
    return s * (1 + x * (1 - s))


def gelu_derivative(x, h=1e-7):
    # GELU 无解析导数，用中心差分
    return (gelu(x + h) - gelu(x - h)) / (2 * h)


def plot_pair(func, dfunc, fname, ftitle, dtitle, ylim_f=None, ylim_d=(-0.1, 1.2), extra_lines=None):
    """绘制「函数图像 + 导数图像」并展示。

    extra_lines: 可选，列表 of (xs, ys, style, label)，叠加到函数图像子图。
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.plot(X, func(X), linewidth=2, label=fname)
    if extra_lines:
        for ex, ey, estyle, elabel in extra_lines:
            ax1.plot(ex, ey, estyle, linewidth=2, label=elabel)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel("输入 x")
    ax1.set_ylabel("输出 f(x)")
    ax1.set_title(ftitle)
    ax1.legend()
    ax1.axhline(0, color="k", alpha=0.3)
    ax1.axvline(0, color="k", alpha=0.3)
    if ylim_f:
        ax1.set_ylim(*ylim_f)

    ax2.plot(X, dfunc(X), linewidth=2, label="f'(x)")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel("输入 x")
    ax2.set_ylabel("导数 f'(x)")
    ax2.set_title(dtitle)
    ax2.legend()
    ax2.axhline(0, color="k", alpha=0.3)
    ax2.axvline(0, color="k", alpha=0.3)
    ax2.set_ylim(*ylim_d)

    fig.tight_layout()
    plt.show()


def main():
    # Linear
    plot_pair(linear, lambda x: np.ones_like(x), "f(x)=x", "线性激活函数", "线性激活函数的导数", ylim_f=(-5, 5))

    # Sigmoid
    plot_pair(sigmoid, sigmoid_derivative, "1/(1+e^-x)", "Sigmoid激活函数", "Sigmoid激活函数的导数", ylim_f=(-0.1, 1.1))

    # Tanh
    plot_pair(tanh, tanh_derivative, "tanh(x)", "Tanh激活函数", "Tanh激活函数的导数", ylim_f=(-1.2, 1.2))

    # ReLU
    plot_pair(relu, relu_derivative, "max(0,x)", "ReLU激活函数", "ReLU激活函数的导数", ylim_f=(-1, 5), ylim_d=(-0.2, 1.2))

    # Leaky ReLU
    a = 0.01
    plot_pair(lambda x: leaky_relu(x, a), lambda x: leaky_relu_derivative(x, a),
              f"max({a}x, x)", "Leaky ReLU激活函数", "Leaky ReLU激活函数的导数", ylim_f=(-0.2, 5), ylim_d=(-0.1, 1.2))

    # ELU
    ae = 1.0
    plot_pair(lambda x: elu(x, ae), lambda x: elu_derivative(x, ae),
              f"ELU(x, α={ae})", "ELU激活函数", "ELU激活函数的导数", ylim_f=(-1.2, 5), ylim_d=(-0.1, 1.2))

    # Swish（与 ReLU 对照）
    plot_pair(swish, swish_derivative, "x·sigmoid(x)", "Swish激活函数", "Swish激活函数的导数",
              ylim_f=(-0.5, 5), ylim_d=(-0.2, 1.2),
              extra_lines=[(X, relu(X), "purple", "ReLU (参考)")])

    # GELU（与现代激活函数对照）
    plot_pair(gelu, gelu_derivative, "x·Φ(x)", "GELU激活函数", "GELU激活函数的导数",
              ylim_f=(-0.5, 5), ylim_d=(-0.2, 1.2),
              extra_lines=[(X, relu(X), "purple", "ReLU (参考)"),
                           (X, swish(X), "darkred", "Swish (参考)"),
                           (X, elu(X), "navy", "ELU (参考)")])

    # 综合对比图
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    ax1, ax2, ax3, ax4 = axes.flatten()

    ax1.plot(X, linear(X), "b-", label="Linear")
    ax1.plot(X, sigmoid(X), "g-", label="Sigmoid")
    ax1.plot(X, tanh(X), "r-", label="Tanh")
    ax1.set_title("经典激活函数")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_ylim(-2, 5)

    ax2.plot(X, relu(X), "purple", label="ReLU")
    ax2.plot(X, leaky_relu(X), "cyan", label="Leaky ReLU")
    ax2.plot(X, elu(X), "navy", label="ELU")
    ax2.set_title("ReLU系列激活函数")
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_ylim(-1.5, 5)

    ax3.plot(X, swish(X), "darkred", label="Swish")
    ax3.plot(X, gelu(X), "darkblue", label="GELU")
    ax3.plot(X, relu(X), "purple", alpha=0.5, label="ReLU (参考)")
    ax3.set_title("现代激活函数")
    ax3.legend()
    ax3.grid(alpha=0.3)
    ax3.set_ylim(-0.5, 5)

    ax4.plot(X, np.ones_like(X), "b-", label="Linear导数")
    ax4.plot(X, sigmoid_derivative(X), "g-", label="Sigmoid导数")
    ax4.plot(X, tanh_derivative(X), "r-", label="Tanh导数")
    ax4.plot(X, relu_derivative(X), "purple", label="ReLU导数")
    ax4.plot(X, gelu_derivative(X), "darkblue", label="GELU导数")
    ax4.set_title("激活函数导数对比")
    ax4.legend()
    ax4.grid(alpha=0.3)
    ax4.set_ylim(-0.1, 1.2)

    for ax in axes.flatten():
        ax.axhline(0, color="k", alpha=0.3)
        ax.axvline(0, color="k", alpha=0.3)
    fig.tight_layout()
    plt.show()

    # 特性对比表
    comparison = {
        "激活函数": ["Linear", "Sigmoid", "Tanh", "ReLU", "Leaky ReLU", "ELU", "Swish", "GELU"],
        "输出范围": ["(-∞,+∞)", "(0,1)", "(-1,1)", "[0,+∞)", "(-∞,+∞)", "(-α,+∞)", "(-∞,+∞)", "(-∞,+∞)"],
        "零中心化": ["是", "否", "是", "否", "否", "近似", "否", "否"],
        "单调性": ["是", "是", "是", "是", "是", "否", "否", "否"],
        "梯度消失": ["无", "严重", "中等", "无", "无", "轻微", "轻微", "轻微"],
        "计算复杂度": ["极低", "中等", "中等", "极低", "极低", "中等", "高", "高"],
        "死神经元": ["无", "无", "无", "有", "无", "无", "无", "无"],
        "主要应用": ["回归输出", "二分类输出", "RNN门控", "深度网络", "改进ReLU", "稳定训练", "深度网络", "Transformer"],
    }
    df = pd.DataFrame(comparison)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)
    print("激活函数特性对比表：")
    print("=" * 120)
    print(df.to_string(index=False))
    print("=" * 120)
    print("\n选择指南：CNN 隐藏层用 ReLU，Transformer 用 GELU；深层网络避免 Sigmoid/Tanh（梯度消失）；"
          "训练不稳定试 ELU/Swish；避免神经元死亡用 Leaky ReLU/ELU。")


if __name__ == "__main__":
    main()
