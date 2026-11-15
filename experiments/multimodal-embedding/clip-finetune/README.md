# CLIP 图文微调：理解数据、损失与检索评估

[English](README.en.md)

当检索对象从日常照片变成产品图片或专业图表，通用图文模型的相似度不一定符合目标场景。使用场景内的图片和描述继续训练，可以尝试调整图文向量空间，再用留出的查询与图片比较微调前后的检索效果。

本目录展示 CLIP 双向对比训练与冻结部分层的代码。已有固定版本的真实 Flickr30k 小样本 GPU 训练和同一测试集的前后对照；完整数据训练尚未验证。

## 数据：一张图对应多条描述

脚本读取 [WhereIsAI/flickr30k-v2](https://huggingface.co/datasets/WhereIsAI/flickr30k-v2)，其数据行包含图像、描述列表、图片 ID 和划分字段。图像来自 Flickr30k，每张图片有多条英语描述；数据集页面未提供完整的数据使用许可说明，使用前应核实图片及标注的来源与适用条件。

一条原始记录与展开后的训练对可以表示为：

```text
原始记录：image = 图片，img_id = 123，caption = [描述 A, 描述 B, …]
展开后：  (图片, 描述 A, 123)，(图片, 描述 B, 123)，…
```

脚本的数据加载逻辑为每条描述复制一个图文对。训练时 `CLIPProcessor` 把图片转为像素张量，把文本转为 token ID 和注意力掩码；模型分别输出图文相似度矩阵。检索时，图片和文字分别编码为向量，再通过向量相似度计算排名。

数据集外层只有 `train`，每行的 `split` 字段才指明 `train`、`val` 或 `test`。数据准备按这一字段分组，再展开描述；同时检查图片 ID 存在、划分名称有效、三个集合非空，并拒绝同一图片 ID 跨集合的记录。其他来源即使沿用 Flickr30k 名称，也应先确认版本和划分相同。

## 损失函数如何训练图文对应关系

设一个批次有 $N$ 组图文对，模型输出归一化后的图像向量 $\mathbf{v}_i$ 与文本向量 $\mathbf{t}_j$。以可学习的缩放系数 $\alpha$ 计算相似度：

$$
s_{ij}=\alpha\,\mathbf{v}_i^{\mathsf T}\mathbf{t}_j.
$$

训练脚本用两个方向的交叉熵平均作为损失，标签均为批次对角线：

$$
\mathcal{L}=\frac{1}{2}\left[
-\frac{1}{N}\sum_{i=1}^{N}\log\frac{e^{s_{ii}}}{\sum_{j=1}^{N}e^{s_{ij}}}
-\frac{1}{N}\sum_{i=1}^{N}\log\frac{e^{s_{ii}}}{\sum_{j=1}^{N}e^{s_{ji}}}
\right].
$$

第一项让每张图片接近对应描述，第二项让每条描述接近对应图片。脚本默认冻结其余层，只解冻视觉编码器最后 2 层、文本编码器最后 4 层，以及两个投影层和缩放参数。

如果同一张图片的两个描述进入同一批次，对角线损失会错误地将另一条正确描述视为负例。因此训练采样器每轮先按图片 ID 分组，从每张图的描述中随机选择一条，再组批；同一轮图片不会重复，只有一条记录的末批会被丢弃。随机种子固定后结果可重现，不同轮可以选中不同描述。不同图片也可能在语义上同样相关，这种假负例仍需通过数据和标注处理。

## 代码与评估范围

- `finetune_clip_with_freeze.py`：构建批次、冻结参数、训练及选择检查点。
- `data_common.py`：读取数据、展开图文对、为训练批次选择不同图片，并计算图搜文与文搜图的 Recall@1/5/10。图搜文时，同一图片的多条描述均可视为相关；文搜图时，每条描述对应一张标注图片。
- `get_dataset.py` 与探索 notebook：数据加载和查看。数据、权重和训练输出不随仓库提供。

配置默认从本地目录 `models/CLIP-ViT-B-32-laion2B-s34B-b79K` 加载模型，并使用较大的批次；需要根据可用设备修改脚本内 `CONFIG`。训练输出默认写入 Git 忽略的 `.work/clip_output`；也可以设置 `XL_OUT_DIR` 指定外部输出目录，避免将模型或训练结果放入版本库。目前训练循环只接受 `grad_accum_steps=1`。`requirements.txt` 提供训练依赖入口，PyTorch 需按平台单独安装；原有不完整的环境快照保存在 `requirements-historical.txt`。模型来源见 [LAION 模型卡](https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K)。

## 真实数据小样本复跑

2026-11-14 在 Linux ARM64、NVIDIA GB10 上，使用模型 `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` 的 revision `1a25a446712ba5ee05982a381eed697ef9b435cf`，权重 SHA-256 为 `74813fbcdc750f235c9784c367ca1394d2a5c25eb0aac92761752ac239db7cff`。Flickr30k revision `a2ac1f0bcee3132331324f1c435bcea4527c08f1` 按原始顺序从训练、验证、测试划分分别取 256、64、64 张图，展开为 1280、320、320 个图文对。独立复制环境为 Python 3.12.3、PyTorch 2.15.0+cu132、Transformers 5.16.1、Datasets 5.2.0、Matplotlib 3.11.2，按清单补装依赖后 `pip check` 通过，不是从零安装。

BF16、批次 16、1 个 epoch 共 16 步，测试集中图搜文 Recall@1 从 0.9844 降为 0.9688，文搜图从 0.9375 降为 0.8781。候选只有 64 张图、单次种子，且未排除模型预训练与数据重叠，结果不能外推到完整数据集；并未证明微调改善效果。

以下小样本入口会按上述固定 revision 加载数据并选择各划分靠前的记录。它会下载数据，输出目录必须为空。`CLIP_MODEL_DIR` 和 `RUNS_DIR` 按下方环境准备设置：

```bash
CLIP_RUN="$(mktemp -d "$RUNS_DIR/clip-real-XXXXXXXX")"
python run_real_subset.py --model "$CLIP_MODEL_DIR" --out "$CLIP_RUN"
```

本次实际使用相同 revision 预先准备的本地数据子集，运行上述入口时另指定了 `--local-dataset <本地子集路径>`；该参数不自动验证本地数据来源。直接从 Hub 加载数据的入口未端到端实测。训练输出包含模型和配置，汇总指标打印在日志中；请自行保留日志。

## 环境准备

以下命令以 Linux 和 Bash 为例；系统、CPU 架构及依赖组合尚未做完整兼容性验证。训练脚本检测到 CUDA 就使用 GPU，否则会回退到 CPU；微调前应确认选用了支持所装 PyTorch CUDA 版本的 NVIDIA GPU。默认配置中 `bf16=True`，因此 GPU 还需支持 BF16；若更改精度须同步记录配置。默认训练 batch size 为 300、评估 batch size 为 640，显存、内存与磁盘最低需求未测定，请结合设备容量调整并为数据缓存、模型和输出预留空间。

先创建独立 Python 环境，按 [PyTorch 安装指南](https://pytorch.org/get-started/locally/)安装适配设备及驱动的 CUDA 版本，再安装训练脚本直接使用的其他依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
# 按 PyTorch 安装指南安装适配设备的 CUDA 版本
python -m pip install -r requirements.txt
```

上述命令提供依赖安装入口，未锁定版本，也尚未在全新环境中验证；`requirements.txt` 不包含平台相关的 PyTorch，且没有锁定全部版本；`requirements-historical.txt` 是不完整的历史环境快照，不应用作安装入口。

模型默认从 `./models/CLIP-ViT-B-32-laion2B-s34B-b79K` 加载，该相对路径以启动脚本时的当前目录为基准。可下载 [模型仓库](https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K)到新目录，并将 `CONFIG["model_name"]` 改为该目录；下列下载会占用磁盘空间，先将 `MODEL_ROOT` 替换为有足够空间的位置：

```bash
export MODEL_ROOT=/path/to/model-assets
mkdir -p "$MODEL_ROOT"
CLIP_MODEL_DIR="$(mktemp -d "$MODEL_ROOT/clip-model-XXXXXXXX")" || exit 1
export CLIP_MODEL_DIR
hf download laion/CLIP-ViT-B-32-laion2B-s34B-b79K --revision 1a25a446712ba5ee05982a381eed697ef9b435cf --local-dir "$CLIP_MODEL_DIR"
```

脚本使用 Transformers 的 `CLIPModel` 和 `CLIPProcessor`，本地目录需要相应的模型权重、配置和处理器文件；不能仅放 OpenCLIP 格式的权重。也可将 `model_name` 改为上述 Hub 模型 ID，让加载时下载缓存。数据从 [Flickr30k 数据集页](https://huggingface.co/datasets/WhereIsAI/flickr30k-v2)通过 `datasets.load_dataset` 获取，默认存入用户的 Hugging Face 数据缓存；请检查缓存位置、网络访问及使用条件。首次加载会下载数据。设置 `FLICKR_REVISION` 为数据集仓库已核对的完整 commit SHA 后再运行，加载器会请求该 revision，训练输出的 `train_config.json` 会记录请求值；未设置时该值为 `null`，加载默认版本且不会自动记录实际解析到的提交，无法据此锁定或复原数据。探索入口 `get_dataset.py` 也接受此变量，但不会写训练配置。

准备后，在同一环境检查 CUDA、BF16 和依赖导入：

```bash
python -c 'import torch, transformers, datasets, matplotlib, PIL; print("torch", torch.__version__, "CUDA", torch.version.cuda, "available", torch.cuda.is_available()); assert torch.cuda.is_available(); print("GPU", torch.cuda.get_device_name(0), "BF16", torch.cuda.is_bf16_supported()); assert torch.cuda.is_bf16_supported()'
```

通过检查不等于完整训练已验证。准备好模型和数据访问权限、安装依赖并修改 `CONFIG` 后，为每次训练指定一个新的输出目录。脚本在加载数据与模型前检查该目录，非空时拒绝启动；已有的空目录也可使用。启动时会创建 `.training-started` 标记，防止两次运行同时使用同一目录；失败后重新运行请使用新目录。`RUNS_DIR` 应改为有足够磁盘空间的位置：

```bash
export RUNS_DIR=/path/to/experiment-runs
mkdir -p "$RUNS_DIR"
XL_OUT_DIR="$(mktemp -d "$RUNS_DIR/clip-XXXXXXXX")" || exit 1
export XL_OUT_DIR
export FLICKR_REVISION="<已核对的数据集完整 commit SHA>"
python finetune_clip_with_freeze.py
```

历史数据 revision 与依赖版本未固定，且尚未验证完整数据训练，不能保证复现历史效果。小规模评估时若候选数少于 K，Recall@K 使用全部现有候选，这种指标不应与完整候选集直接比较。

如需在 Linux aarch64 NVIDIA GPU 上重建上方干净环境烟雾检查，可从本目录执行以下命令。它会新建独立 venv 和外部数据/输出目录；PyTorch 与 torchvision 固定为已验证的 CUDA 13.2 配对版本，其余依赖按 `requirements.txt` 安装。每次运行使用新的 `XL_RUN_ID`：

```bash
export XL_RUN_ID="clip-clean-smoke-$(date +%Y%m%d-%H%M%S)"
export XL_DATA_DIR=/path/to/external/clip-data
export XL_OUT_DIR=/path/to/external/clip-output
export XL_REQUIRE_EXTERNAL_ENV=1
bash scripts/clean_env_cuda_smoke.sh
```

此入口只验证环境和合成数据训练链路，不下载预训练权重或 Flickr30k；它不替代真实数据小样本运行。

安装 PyTorch、Transformers、Datasets 和 Matplotlib 后，可在本目录执行下列 CPU 检查。测试用合成字段及随机初始化的小模型，覆盖划分、跨集合 ID 冲突、批次内不重复和一次前向反向；它不下载 Flickr30k，也不评价检索效果。

```bash
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -v
```

也可以先用随机初始化的小型 CLIP 和合成图文数据，在 CUDA 上检查真实训练入口。下面命令不需要下载预训练权重或 Flickr30k；它执行两步训练、验证、测试和保存，并检查投影层权重发生变化。示例必须在本目录运行，`RUNS_DIR` 改为有写入权限的目录：

```bash
export RUNS_DIR=/path/to/experiment-runs
mkdir -p "$RUNS_DIR"
XL_OUT_DIR="$(mktemp -d "$RUNS_DIR/clip-smoke-XXXXXXXX")" || exit 1
export XL_OUT_DIR
python tests/smoke_cuda_train.py
```

输出在 `$XL_OUT_DIR/synthetic-clip-training/`，包括训练配置、检查点、曲线和最终模型；模拟处理器只写入占位文件，最终模型目录不是可直接用于实际推理的完整模型包。2026-11-13 在 Linux ARM64、NVIDIA GB10 上完成上述两步 CUDA 检查及 10 项 CPU 测试；使用 Python 3.12.3、PyTorch 2.15.0+cu132、Transformers 5.22.2、Datasets 5.2.0、Matplotlib 3.11.2、Pillow 13.0.0 和 torchvision 0.29.1+cu130，依赖来自已有环境及额外依赖目录，不代表全新环境安装已验证。该检查替换了数据加载、处理器和模型加载，使用随机模型、合成数据及 FP32，不能证明真实数据处理、预训练权重加载、BF16 或历史效果可复现；输出指标也没有模型效果意义。

2026-11-15 在 Linux aarch64、NVIDIA GB10 上，使用 [全新环境 CUDA 检查脚本](scripts/clean_env_cuda_smoke.sh)从空虚拟环境安装并通过 `pip check`。环境为 Python 3.12.3、PyTorch 2.15.0+cu132、torchvision 0.29.1+cu132、Transformers 5.23.0、Datasets 5.2.0、Matplotlib 3.11.2 和 Pillow 13.0.0；CUDA 13.2 可用并支持 BF16。两步 CUDA 训练、验证、测试和保存均通过，且模型投影权重发生变化。测试替换了处理器、模型加载和数据准备，只用随机初始化模型、合成数据及 FP32；它验证干净环境中的训练代码链路，不验证 Flickr30k、预训练权重、BF16 或检索效果。摘要见 [干净环境 CUDA 检查记录](evidence/clean-env-smoke-2026-11-15.json)。

脚本会将微调模型和原模型放在同一测试候选集上，分别计算双向 Recall。该指标受候选集合、划分和标注覆盖影响；只有实际执行并保存配置、结果与逐项证据，才能判断微调是否改变检索效果。本次小样本记录保留了运行配置和汇总日志，未保存逐查询证据。
