# 从字节到 token：理解分词、生成和成本

[English](README.en.md)

本目录保存实验代码、配置与结果证据。实验目的、原理、数据、结果和边界见[完整实验文档](../../docs/zh/inference/分词机制与生成成本.md)。

## 运行示例

在本实验目录使用 Python 3 运行，以下两个脚本只用标准库，无需 GPU、安装额外依赖或下载模型；输入样例已写在脚本中：

```bash
python scripts/toy_bpe.py
python scripts/ngrams.py
```

真实编码器对照需要安装 `tiktoken`，首次运行可能需要联网下载词表。可创建独立环境，避免修改其他实验的依赖；下面的激活命令适用于 Linux 和 macOS：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
python scripts/compare_tiktoken.py
```

Windows PowerShell 使用 `.\.venv\Scripts\Activate.ps1` 激活，或直接通过 `.\.venv\Scripts\python.exe` 执行上述 Python 命令。依赖清单未锁定版本，运行时请记录实际安装版本。

运行成功时，玩具 BPE 会打印每次合并及样例 token，并断言编码后可以还原原文；n-gram 脚本打印英文词组与中文字符组合。`compare_tiktoken.py` 也会检查 UTF-8 字节能否还原输入；它输出的版本和 token ID 应随实际安装的 `tiktoken` 一并记录。

2026-11-13 在 NVIDIA DGX Spark 上新建 Python 3.12.3 虚拟环境，按 `requirements.txt` 安装得到 tiktoken 0.14.0，`pip check` 通过。上面三个脚本均运行成功，包括首次获取 `r50k_base` 和 `cl100k_base` 词表。此记录覆盖这组示例和该环境，不代表其他平台或依赖版本均已验证。
