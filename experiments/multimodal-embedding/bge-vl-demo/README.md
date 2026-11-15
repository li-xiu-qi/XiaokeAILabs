# BGE-VL：图片与图文组合查询

[English](README.en.md)

给一张参考图片，再用文字描述希望改变的条件，模型可以将这组输入编码成查询向量，并与候选图片的向量比较。这个过程可用于探索组合图像检索，例如按参考图寻找背景更暗的照片。

[`test_bge_vl.ipynb`](test_bge_vl.ipynb) 基于 [BGE-VL 官方模型示例](https://huggingface.co/BAAI/BGE-VL-large) 使用 Transformers 接口：先调用 `set_processor`，再分别编码查询和候选图片，以向量点积计算分数。notebook 包括图片查图，以及图片加文字的组合查询；它没有纯文本查图的代码。

## 环境准备

示例使用 `BAAI/BGE-VL-large`，需安装兼容的 PyTorch、Transformers、Pillow 和 FlagEmbedding；模型的自定义代码由 `trust_remote_code=True` 加载。`requirements.txt` 固定 Transformers 4.46.3 和 FlagEmbedding 1.4.2，其余依赖及 large 模型版本未锁定；运行前请查阅上游模型卡并确认代码来源。notebook 会在输出中显示当前支持的模型映射；在 Jupyter 中保存会改写 notebook 文件，因此下面在独立工作目录运行副本。

以下命令以 Linux 和 Bash 为例；操作系统、CPU 架构与依赖版本组合尚未做完整验证，内存和磁盘最低要求也未测定。notebook 没有把模型显式移动到 GPU；不要将现有示例的运行结果当作 GPU 推理验证。首次运行需连接模型源，并为模型权重及缓存留出空间。

先创建独立 Python 环境，按 [PyTorch 安装指南](https://pytorch.org/get-started/locally/)安装适配设备的版本，再安装 notebook 和示例所需的包：

```bash
python3 -m venv .venv
source .venv/bin/activate
# 按 PyTorch 安装指南安装适配设备的 torch
python -m pip install -r requirements.txt
```

这是依赖安装入口，除上述两个版本外未锁定完整依赖，也尚未验证全新环境能否直接运行。FlagEmbedding 1.4.2 的依赖声明允许 Transformers 4.44.2 至 6.0 以下；在已有 Python 和 PyTorch 的环境中，`pip install --dry-run -r requirements.txt` 已完成依赖解析，但未安装或运行完整组合，notebook 映射单元也尚未实测。可先检查包导入，然后创建独立运行目录并复制 notebook；将 `RUNS_DIR` 替换为工作目录，在内核选择中选择当前环境：

```bash
python -c 'import torch, transformers, PIL, FlagEmbedding; print("torch", torch.__version__, "transformers", transformers.__version__)'
python -m pip check
export RUNS_DIR=/path/to/experiment-runs
mkdir -p "$RUNS_DIR"
BGE_RUN_DIR="$(mktemp -d "$RUNS_DIR/bge-vl-XXXXXXXX")" || exit 1
export BGE_RUN_DIR
cp test_bge_vl.ipynb "$BGE_RUN_DIR/"
mkdir -p "$BGE_RUN_DIR/.work"
# 在 $BGE_RUN_DIR/.work 放好 cat.png、dog.png 后再启动
cd "$BGE_RUN_DIR"
python -m jupyter lab test_bge_vl.ipynb
```

首次加载时，notebook 的 `AutoModel.from_pretrained` 和 `set_processor` 从 [模型仓库](https://huggingface.co/BAAI/BGE-VL-large)获取模型及处理器文件，缓存位置遵循 Hugging Face 配置；模型仓库或依赖升级可能改变接口。若要离线运行，可先按模型源说明准备本地文件，并将 notebook 的 `MODEL_NAME` 改为包含模型和处理器文件的本地目录；large 模型的离线路径尚未验证。

运行前请在新建运行目录的 `.work/` 下准备两张有权使用的图片，命名为 `cat.png` 和 `dog.png`，或修改 notebook 中的图片路径。示例只用于查看接口和输出形状，不能作为模型效果评测。

## CUDA 接口检查

以下单独的检查脚本使用较小的 `BAAI/BGE-VL-base`，通过同一 `AutoModel`、`set_processor` 和 `encode` 接口，分别对单图查询和图文组合查询计算两张候选图的分数。它生成两张纯色图片，需要 CUDA，不运行 notebook 中用于查看 FlagEmbedding 模型映射的单元。先在独立环境中按 PyTorch 安装指南安装相互匹配的 CUDA 版 PyTorch 和 torchvision，再执行：

```bash
python -m pip install 'transformers==4.46.3' pillow
export RUNS_DIR=/path/to/experiment-runs
mkdir -p "$RUNS_DIR"
BGE_RUN_DIR="$(mktemp -d "$RUNS_DIR/bge-vl-smoke-XXXXXXXX")" || exit 1
export BGE_RUN_DIR
export MODEL_DIR="$BGE_RUN_DIR/model"
# 首次下载权重，请预留磁盘空间；local_dir 不要指向已有文件。
python -c 'import os; from huggingface_hub import snapshot_download; snapshot_download(repo_id="BAAI/BGE-VL-base", revision="cc4c733ed997dbee4ac70ccffb911e70c9c24b93", local_dir=os.environ["MODEL_DIR"])'
python smoke_bge_vl.py --model "$MODEL_DIR" --output-dir "$BGE_RUN_DIR/results"
```

`results/` 保存生成的图片和 `result.json`。2026-11-13 在 Linux ARM64、NVIDIA GB10 上，使用 Python 3.12.3、PyTorch 2.15.0+cu132、Transformers 4.46.3、Pillow 13.0.0、torchvision 0.29.1+cu130 和上述模型 revision，实测离线加载及 CUDA 两种查询均完成，候选向量形状为 `[2, 512]`。依赖来自已有环境与隔离依赖目录，尚未在全新环境验证上述安装命令。纯色图的分数只用于检查接口，不能衡量检索效果；该检查不代表 large 模型或完整 notebook 已验证。

同一模型 revision 的 `AutoModel` 自定义代码在本次 Transformers 5.16.1 和 5.22.2 环境中出现位置索引异常，导致 CUDA 推理失败；请优先使用上面实测的 4.46.3，并在升级依赖后重新检查。不同模型加载接口可能走不同实现，不能用其他接口的成功结果代替此处检查。
