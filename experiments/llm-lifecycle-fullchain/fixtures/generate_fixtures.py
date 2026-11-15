#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""generate_fixtures.py — 生成可复现的多域 UTF-8 评估语料切片。

每个域固定 100k chars，保证跨 tokenizer 评估口径一致。
输出: fixtures/*.txt
"""

from __future__ import annotations

import random
from pathlib import Path

FIXTURES_DIR = Path(__file__).resolve().parent
FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
SEED = 1234
CHUNK = 100_000  # 每个 fixture 的目标字符数


def _pad(text: str, target: int = CHUNK) -> str:
    """将 text 填充到 target 长度（重复自身），并在末尾截断。"""
    if not text:
        return ""
    while len(text) < target:
        text = text + "\n" + text
    return text[:target]


def make_python_code() -> str:
    """从项目 scripts/ 提取 Python 源码，作为代码域语料。"""
    scripts = [
        "scripts/llm_common.py",
        "scripts/01_train_tokenizer.py",
        "scripts/dataset.py",
        "examples/generate_toy_dataset.py",
        "scripts/08_eval_report.py",
    ]
    parts: list[str] = []
    for rel in scripts:
        p = Path(rel)
        if p.exists():
            parts.append(f"# ==== {rel} ====\n{p.read_text(encoding='utf-8', errors='replace')}")
    text = "\n\n".join(parts)
    return _pad(text if text else "# no scripts found\n")


def make_latex_math() -> str:
    """LaTeX 数学公式域：微分几何、线性代数、概率统计。"""
    blocks = [
        r"\begin{equation} \nabla \cdot \mathbf{E} = \frac{\rho}{\varepsilon_0} \end{equation}",
        r"\begin{equation} \int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi} \end{equation}",
        r"\begin{equation} \mathcal{L}(\theta) = -\sum_{i=1}^{N} y_i \log p(y_i \mid x_i; \theta) + \lambda \|\theta\|_2^2 \end{equation}",
        r"\begin{equation} \frac{\partial f}{\partial x} = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h} \end{equation}",
        r"\begin{equation} \mathbf{A} \mathbf{x} = \lambda \mathbf{x}, \quad \mathbf{A} \in \mathbb{R}^{n \times n} \end{equation}",
        r"\begin{equation} P(A \mid B) = \frac{P(B \mid A) P(A)}{P(B)} \end{equation}",
        r"\begin{equation} \sum_{k=0}^{\infty} \frac{x^k}{k!} = e^x \end{equation}",
        r"\begin{equation} \oint_{\partial \Omega} \mathbf{F} \cdot d\mathbf{r} = \iint_{\Omega} (\nabla \times \mathbf{F}) \cdot d\mathbf{S} \end{equation}",
        r"\begin{equation} \sigma^2 = \frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2 \end{equation}",
        r"\begin{equation} \frac{d}{dx} \left( \int_a^x f(t) dt \right) = f(x) \end{equation}",
        r"\begin{equation} (a + b)^n = \sum_{k=0}^{n} \binom{n}{k} a^{n-k} b^k \end{equation}",
        r"\begin{equation} \mathbf{v}(t) = \mathbf{v}_0 + \int_0^t \mathbf{a}(\tau) d\tau \end{equation}",
        r"\begin{equation} \chi^2 = \sum_{i=1}^{k} \frac{(O_i - E_i)^2}{E_i} \end{equation}",
        r"\begin{equation} \sin^2 \theta + \cos^2 \theta = 1 \end{equation}",
        r"\begin{equation} \lim_{n \to \infty} \left(1 + \frac{x}{n}\right)^n = e^x \end{equation}",
    ]
    text = "\n\n".join(blocks)
    return _pad(text)


def make_chinese_tech() -> str:
    """中文技术散文：LLM 训练、分词器、分布式系统。"""
    paras = [
        "大语言模型的训练过程可以分为预训练、监督微调和人类反馈强化学习三个阶段。预训练阶段，模型在海量无标注文本上学习语言统计规律；微调阶段，通过指令数据让模型学会遵循人类意图；强化学习阶段则进一步对齐模型输出与人类偏好。",
        "分词器是连接文本与模型的第一道桥梁。字节级 BPE（Byte-Level BPE）能够以较小的词表覆盖几乎全部 Unicode 字符，避免未登录词问题。GPT-4 采用的分词策略结合了正则预分词与 ByteLevel 编码，在压缩率和语义完整性之间取得了良好平衡。",
        "分布式训练框架通常采用数据并行策略。在单机多卡场景下，NCCL 后端能够提供最高的集合通信带宽；而在跨节点场景中，InfiniBand 或 RoCEv2 网络拓扑对通信效率有决定性影响。PyTorch 的 DistributedDataParallel 封装了梯度同步细节，使模型代码与并行逻辑解耦。",
        "模型参数量的估算公式为：对于 Transformer 架构，词嵌入层参数量为 vocab_size × d_model，每层自注意力层为 4 × d_model²，前馈网络为 2 × d_model × d_ff。因此总参数量主要由 d_model、d_ff 和层数决定。",
        "学习率调度是训练稳定性的关键因素。余弦退火调度在预训练后期逐渐降低学习率，有助于模型收敛到更平坦的极小值。Muon 优化器通过 Newton-Schulz 正交化更新隐层权重矩阵，在小批量设置下表现出优于 AdamW 的收敛特性。",
        "KVCache 机制通过缓存已计算的键值向量来避免推理时的重复计算。在自回归生成过程中，每一步只需计算当前 token 的注意力输出，历史 KV 直接从缓存读取，将解码复杂度从 O(T²) 降至 O(T)。",
        "RoPE（Rotary Position Embedding）将绝对位置编码转换为相对位置编码的旋转矩阵形式。其优势在于：外推长度随序列增长呈线性衰减，且注意力分数天然具备相对距离敏感性。DeepSeek 系列模型采用扩展 RoPE（YaRN）以支持更长上下文。",
        "混合精度训练通过自动类型转换在保持数值稳定性的同时加速计算并节省显存。BF16 在 GPU 上的动态范围与 FP32 一致，无需 Loss Scaling；FP16 则需要梯度缩放以防止下溢，但计算吞吐通常更高。",
        "GQA（Grouped-Query Attention）将多个查询头共享同一组键值头，在保持接近 MHA 质量的同时显著降低 KV Cache 显存占用。MQA 是 GQA 的极端形式（所有头共享一组 KV），但质量损失较大。",
        "评估指标 BPB（Bits Per Byte）通过与词表大小无关的字节级负对数似然来度量压缩效率。相比 PPL，BPB 能够在不同词表大小的分词器之间进行公平对比，是衡量分词器跨域鲁棒性的核心指标。",
    ]
    rng = random.Random(SEED)
    rng.shuffle(paras)
    text = "\n\n".join(paras)
    return _pad(text)


def make_mixed_zh_en_code() -> str:
    """中英文混合与代码穿插：常见于技术博客与开源项目 README。"""
    blocks = [
        "# Quick Start\n",
        "在本节中，我们将快速搭建一个最小化的 Transformer 语言模型训练流水线。首先克隆仓库并创建虚拟环境：\n\n",
        "```bash\ngit clone https://github.com/example/llm-lifecycle.git\ncd llm-lifecycle\npython -m venv .venv\nsource .venv/bin/activate\npip install -r requirements.txt\n```\n\n",
        "上述命令会在当前目录创建 `.venv` 并安装 `torch`、`tokenizers` 和 `matplotlib`。注意：Windows 用户请使用 `python -m venv .venv` 并运行 `.venv\\Scripts\\activate`。\n\n",
        "## 分词器训练 (Tokenizer Training)\n\n",
        "分词器训练是构建语言模型的第一步。我们使用 `tokenizers` 库中的 BPE 实现，并采用 GPT-4 风格的正则预分词模式。关键代码片段如下：\n\n",
        "```python\nfrom tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers\n\nGPT4_SPLIT_PATTERN = (\n    \"(?i:'s|'t|'re|'ve|'m|'ll|'d)\"\n    \"|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+\"\n    \"|\\p{N}{1,3}\"\n    \"| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*\"\n    \"|\\s*[\\r\\n]+\"\n    \"|\\s+(?!\\S)\"\n    \"|\\s+\"\n)\n```\n\n",
        "上述正则表达式将文本拆分为单词、数字和符号片段，随后由 ByteLevel 编码器将其映射为字节序列。这种两阶段分词策略保证了罕见字符和表情符号的可处理性。\n\n",
        "## 模型架构 (Model Architecture)\n\n",
        "我们实现的 CausalLM 基于 Pre-Norm 残差结构，包含以下核心组件：\n\n",
        "1. **RMSNorm**：无学习参数的均方根归一化，公式为 $\\hat{x}_i = x_i / \\sqrt{\frac{1}{d}\\sum_j x_j^2 + \\epsilon}$。\n",
        "2. **RoPE**：旋转位置编码，通过复数旋转矩阵 $e^{i m \\theta}$ 将绝对位置转换为相对注意力偏置。\n",
        "3. **Muon 优化器**：对二维权重矩阵应用 Newton-Schulz 正交化迭代，提升小批量训练的泛化稳定性。\n\n",
        "在 gpu-srv-1 上，模型配置为 `d_model=320`、`n_layers=6`、`n_heads=8`、`d_ff=1280`，总参数量约为 12.6M。词嵌入层贡献 `vocab_size × d_model` 个参数，lm_head 贡献 `d_model × vocab_size` 个参数。\n\n",
        "## 分布式训练 (Distributed Training)\n\n",
        "当使用 `torchrun` 启动多卡训练时，每个 rank 独立初始化模型副本，并通过 AllReduce 同步梯度。`init_distributed` 函数自动检测 `WORLD_SIZE` 环境变量并选择后端：CUDA 环境优先使用 NCCL，否则回退到 Gloo。\n\n",
        "```python\nctx = init_distributed(device='auto', backend='auto')\nmodel = wrap_ddp(build_model(cfg, seed=seed), ctx)\n```\n\n",
        "在 Windows 环境下，NCCL 可能未编译进官方 PyTorch wheel，此时需显式指定 `--backend gloo`。虽然带宽较低，但对于小模型和 CPU 训练场景完全可接受。\n\n",
        "## 常见问题 (Troubleshooting)\n\n",
        "- **OOM（Out of Memory）**：减小 `batch_size` 或 `block_size`，或启用 `grad-accum` 累积多步梯度。\n",
        "- **梯度消失**：检查是否使用了 Pre-Norm 和残差连接，并确认学习率未过低。\n",
        "- **分词 roundtrip 失败**：确保 `tok.encode` 与 `tok.decode` 使用相同的特殊 token 处理策略（`skip_special_tokens=False` 用于一致性校验）。\n",
    ]
    text = "".join(blocks)
    return _pad(text)


def make_digits_arithmetic() -> str:
    """纯数字算术：多位数乘除、百分比、科学计数法、货币计算。"""
    rng = random.Random(SEED + 1)
    lines: list[str] = []
    # 多位数加减乘除
    for a in range(100, 999):
        for b in range(1, 50):
            lines.append(f"{a} + {b} = {a + b}")
            lines.append(f"{a} - {b} = {a - b}")
            if a % (b + 1) == 0:
                lines.append(f"{a} / {b} = {a // b}")
            lines.append(f"{a} * {b} = {a * b}")
    # 百分比
    for p in [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95]:
        for base in [100, 200, 500, 1000, 2500, 10000]:
            lines.append(f"{p}% of {base} = {base * p // 100}")
    # 科学计数法
    for coeff in [1.23, 4.56, 7.89, 9.99]:
        for exp in range(-6, 7):
            val = coeff * (10 ** exp)
            lines.append(f"{coeff}e{exp:+d} = {val}")
    # 货币计算
    for price in [9.99, 19.99, 29.99, 49.99, 99.99]:
        for qty in range(1, 11):
            total = price * qty
            lines.append(f"${price} x {qty} = ${total:.2f}")
    # 时间与日期计算
    for hour in range(0, 24):
        for minute in range(0, 60, 15):
            lines.append(f"{hour:02d}:{minute:02d} = {hour * 60 + minute} minutes")
    rng.shuffle(lines)
    text = "\n".join(lines)
    return _pad(text)


def main() -> int:
    fixtures = {
        "python_code": make_python_code(),
        "latex_math": make_latex_math(),
        "chinese_tech": make_chinese_tech(),
        "mixed_zh_en_code": make_mixed_zh_en_code(),
        "digits_arithmetic": make_digits_arithmetic(),
    }
    for name, text in fixtures.items():
        path = FIXTURES_DIR / f"{name}.txt"
        path.write_text(text, encoding="utf-8")
        print(f"[fixtures] wrote {path} ({len(text):,} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
