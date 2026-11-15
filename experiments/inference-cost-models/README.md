# 大模型推理成本：从内存预算到性能估算

[English](README.en.md)

本目录保存实验代码、配置与结果证据。实验目的、原理、数据、结果和边界见[完整实验文档](../../docs/zh/inference/推理内存预算与成本估算.md)。

[scripts](scripts/) · [records](records/) · [evidence](evidence/)

以下命令均从本实验目录运行。

## 环境准备

基础算例可在支持 Python 3.10 及以上版本的系统上运行，不依赖特定 CPU 架构或 GPU，也不需要安装第三方库、模型或数据集。先进入本实验目录并检查解释器；隔离环境为可选项：

```bash
cd experiments/inference-cost-models
python --version
python -m venv .venv
source .venv/bin/activate  # Linux / macOS；Windows 使用 .venv\Scripts\Activate.ps1
```

HTTP 计时需要网络可访问的推理服务和第三方 `requests`，下面另列安装方式。`requirements.txt` 固定 `requests==2.34.2`，传递依赖仍由包管理器解析；计时结果取决于服务与运行条件。

## 可运行的示例

需要 Python 3.10+。下例只计算稠密、非因果 full attention 两次矩阵乘的理论操作量，以及单层假设物化的注意力分数矩阵大小；不包含混合架构中的其他层，不预测真实延迟或峰值内存。

```bash
python scripts/attention_model.py
```

内存算例还可运行 `python scripts/hybrid_architecture_model.py`。这两个算例只使用 Python 标准库和仓内模块，不需要模型权重、数据集或 GPU；内存算例不包含其他层状态与运行时开销。

请求级 HTTP 计时需要另行安装 `requests`，并自行准备支持 `/v1/completions` 的模型服务：

```bash
python -m pip install -r requirements.txt
python scripts/compare_engines.py --help
python -m unittest discover -s tests -v
```

实际计时需提供服务 URL、模型名和提示词。结果包含排队、提示词处理、生成与网络耗时，不拆分推理阶段。

基础算例无需安装 requirements；HTTP 请求和离线检查使用的 `requests==2.34.2` 列在 [requirements.txt](requirements.txt)。测试中网络请求由 mock 替代，不需要推理服务。

2026-11-13 在 Windows 11 的 WSL2 Ubuntu 24.04 ARM64、Python 3.12.3 的独立环境按清单安装了 requests 2.34.2，`pip check` 通过；两个基础算例、HTTP 脚本的 `--help` 及 5 项离线测试运行成功。未连接推理服务，也未验证 GPU 基准或历史数值。

2026-11-15 在 Linux aarch64 的新建虚拟环境中按固定依赖安装 `requests==2.34.2`，`pip check` 通过；两个基础算例和 HTTP 脚本的 `--help` 均运行成功。未请求推理服务，未运行 GPU 基准。见[本次环境检查摘要](evidence/clean-env-check-2026-11-15.json)。

历史 `scripts/bench_embedding_spark.py` 是单独的 GPU 和模型基准草稿，需要 CUDA 版 PyTorch、NVIDIA 驱动、`nvidia-smi`、[requirements-gpu.txt](requirements-gpu.txt) 中的 sentence-transformers，以及脚本 `MODELS` 中对应模型；首次加载可能下载模型。先按 [PyTorch 安装说明](https://pytorch.org/get-started/locally/)安装与目标设备兼容的版本，再运行 `python -m pip install -r requirements-gpu.txt` 安装可选依赖。脚本使用合成重复文本，会把结果写入当前用户主目录下的 `emb_bench_*.json`，同名文件会被覆盖。它并非上面的轻量算例，完整依赖版本和测量有效性未锁定，运行前请先阅读[使用边界与审计](records/使用边界与审计.md)。

原理在文首的实验文档。已有其他脚本是研究草稿，其参数与输出尚待逐项审计；运行前请先阅读[使用边界与审计](records/使用边界与审计.md)。

## 审计与历史记录

这个实验的历史脚本和报告包含有价值的推导线索，但经 2026-11-07 的代码审阅，尚不能用作可靠的跨硬件或跨工作负载性能预测。引用任何数字前，先读审计页。

| 文档 | 内容 |
|---|---|
| [使用边界与审计](records/使用边界与审计.md) | 脚本可用范围、已发现的问题，历史结论以此页为准 |
| [LLM-Viewer 交叉验证](records/LLM-Viewer交叉验证.md) | 2026-09-13 的历史对照，含部署命令与踩坑，结论存在已知问题 |
| [迭代日志](records/迭代日志.md) | 建模过程的过程态记录 |
| [vLLM 在 GB10 上的部署](records/vLLM在GB10上的部署.md) · [SGLang 在 GB10 上的部署](records/SGLang在GB10上的部署.md) | 两个引擎的部署步骤与排错 |
