# 中文文本 Embedding 微调：用查询与相关文档训练检索模型

[English](README.en.md)

[完整实验讲解](../../docs/zh/retrieval/embedding/Embedding监督微调.md)说明问题、原理和结果；以下是代码与复现入口。请在工作副本中运行，数据与输出不要写入公开仓。

## 环境准备与快速运行

以下命令以 Linux Bash 为例，在实验目录执行。正式训练需要可用的 CUDA GPU；实际显存随模型、批次和文本长度变化，批次至少为 2。创建仓库外的独立环境，按 [PyTorch 安装说明](https://pytorch.org/get-started/locally/)选择匹配系统架构、驱动与 GPU 的构建，再安装脚本直接依赖：

```bash
python3 -m venv /path/to/venvs/retrieval
source /path/to/venvs/retrieval/bin/activate
# 按 https://pytorch.org/get-started/locally/ 安装与系统、架构和 GPU 匹配的 PyTorch
python -m pip install -r requirements.txt
python -m pip check
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA build:', torch.version.cuda, 'GPU:', torch.cuda.is_available()); assert torch.cuda.is_available(), '训练需要可用的 CUDA GPU'"
python -c "import transformers, sentence_transformers, tqdm; print('依赖导入成功')"
```

[`requirements.txt`](requirements.txt) 和 [`requirements-eval.txt`](requirements-eval.txt) 固定了本次验证的直接依赖；PyTorch 仍需按目标 GPU 单独安装。2026-11-15 在 Linux ARM64 / NVIDIA GB10 上从新建虚拟环境安装并完成真实数据 GPU 微调和评估，版本与范围见下方记录。首次使用默认模型会从 [BAAI/bge-base-zh-v1.5](https://huggingface.co/BAAI/bge-base-zh-v1.5) 下载；也可预先获取并核验模型后用本地路径。需要网络及模型、数据、检查点的磁盘空间；脚本启用 `trust_remote_code=True`，加载前应检查所用模型的代码和许可。

要从空目录执行这次验证的快速流程，可在实验目录运行下列脚本。它会新建独立虚拟环境、检查 CUDA 与依赖、校验数据文件、训练并评估；每次运行必须使用新的 `XL_RUN_ID`，数据和输出目录必须位于仓库外。默认 PyTorch 配置对应已验证的 GB10；其他设备需设置匹配的 `TORCH_INDEX_URL` 与 `TORCH_PACKAGE`。

```bash
export XL_REQUIRE_EXTERNAL_ENV=1
export XL_RUN_ID="$(date +%Y%m%d-%H%M%S)"
export XL_DATA_DIR="$HOME/xl-data"
export XL_OUT_DIR="$HOME/xl-output"
# 网络环境需要镜像时设置：export HF_ENDPOINT=https://hf-mirror.com
# 若继承的代理不可用，再加：export XL_CLEAR_PROXY=1
bash scripts/clean_env_gpu_smoke.sh
```

脚本默认从固定 revision 下载 T2Retrieval；若已有该版本的数据，可通过 `T2RETRIEVAL_DATA_DIR` 指向数据目录，脚本仍会逐文件校验 SHA-256。模型从固定 revision 下载。该流程使用小规模开发集，不替代完整训练或正式测试。

将数据和输出指定到仓库外，并让 `--output_dir` 指向不存在的新目录或空目录；已有文件、非空目录或位于输出目录内的训练数据会被拒绝，以保护已有结果。下面运行随附演示数据（若指定路径是缺失的默认 `demo_data.jsonl`，脚本会创建它；为避免向仓库写入，先切到仓外工作目录，并使用脚本的绝对路径）：

```bash
REPO=/path/to/XiaokeAILabs
WORK=/path/to/new-working-directory
mkdir -p "$WORK"
cd "$WORK"
python "$REPO/experiments/embedding-finetuning/scripts/train_embedding.py" --output_dir "$WORK/model"
```

可在实验目录运行 `python -m unittest discover -s tests -v` 检查输出目录保护；测试不下载模型，也不训练。

默认加载 `BAAI/bge-base-zh-v1.5`，首次运行可能需要下载模型；也可通过 `--model_name_or_path` 指向本地模型。脚本自动选择 CUDA，若不可用则使用 CPU。默认训练 3 轮、批次大小为 4，每轮保存检查点；上例最终模型写入 `$WORK/model/final_model`。

默认数据文件是 `demo_data.jsonl`；该文件缺失时脚本创建内置演示数据。自定义数据路径缺失时直接报错，避免误把演示数据当作真实训练集。批次和数据集至少需要两条记录；若末批只有一条，脚本丢弃该末批，避免无负例的零损失更新。

2026-11-13 在一台 DGX Spark 上完成一次 CUDA 冒烟：以已有 Python 3.12.3、PyTorch 2.15.0+cu132、Transformers 5.22.2 环境为基础，为本次运行隔离添加 Sentence Transformers 6.3.0、scikit-learn 1.9.1、Pillow 13.0.0、joblib 1.6.0、threadpoolctl 3.8.0，以及 cloudpickle、narwhals。使用 tiny-random-bert 和 4 条内置合成样本，批次 4、1 轮共 1 个批次，训练退出码为 0，并保存了检查点和 `final_model`。这不是按本目录清单从零安装的环境，也不是默认模型、真实数据或效果测试；基础环境的 `pip check` 报告 `nvidia-cusparselt-cu13 0.8.1 is not supported on this platform`，因此不能视为完整依赖检查通过。

2026-11-14 在复制并调整的独立环境中，使用默认模型 `BAAI/bge-base-zh-v1.5` 的 revision `f03589ceff5aac7111bd60cfc7d497ca17ecac65` 和 4 条内置样本，按批次 4 训练 1 轮并保存检查点和最终模型，CUDA 运行退出码为 0。环境使用 Python 3.12.3、PyTorch 2.15.0+cu132、Transformers 5.16.1 和 Sentence Transformers 6.0.1，`pip check` 通过。模型通过镜像获取；该环境并非从零安装，只有一个训练批次，未做真实数据或独立检索评估。

## 真实中文数据的小规模验证

2026-11-14 在复制并调整的独立环境使用固定版本的 [T2Retrieval](https://huggingface.co/datasets/mteb/T2Retrieval) 中文数据：从原 `dev` 划出 256 条训练查询和 100 条测试查询，候选池共 409 篇文档；查询文本与训练正例文档 ID 和测试相关文档 ID 不重叠。默认模型 CUDA 训练 1 轮、32 步，Hit@10 从 0.99 到 0.99，MRR 从 0.9753 到 0.9736。

2026-11-15 在 Linux ARM64 / NVIDIA GB10 的新建虚拟环境中，按上述依赖版本从零安装，`pip check` 通过。使用固定 revision 的 BGE 模型和 T2Retrieval 数据完成 1 轮、32 步 CUDA 微调：100 条测试查询的 Hit@10 为 0.99（训练前后相同），MRR 从 0.9753 到 0.9805。此次数据来自已有数据副本，三个 Parquet 文件的 SHA-256 均与下方记录一致；模型在本次运行中从镜像下载。运行摘要见 [`evidence/clean-env-gpu-smoke-2026-11-15.json`](evidence/clean-env-gpu-smoke-2026-11-15.json)。

两次运行的候选池都较小，基线接近满分，且只运行一个随机种子；训练脚本未固定随机种子，也未排除语义近重复或模型预训练数据重叠。指标变化不能证明微调带来可推广的提升，也不是官方测试集成绩。

以下流程需要额外的 `pandas`、`pyarrow` 和 Hugging Face 下载工具；固定数据版本的三个 Parquet 文件分别是 `corpus/dev-00000-of-00001.parquet`、`data/dev-00000-of-00001.parquet` 和 `queries/dev-00000-of-00001.parquet`，其 SHA-256 依次为 `8f0400530c6ba664e580e2ae7bb5ed41a908b8abc17c3b3934f42cc65992ac1e`、`c4c6ffd715faf151e0832a26dfcf2d7ec1edc17d9620a9aa99e721cc2436d7d2`、`186b4d0a13cc4d8339eccfc6d465d1b2d4d1f6572defa8d39f3c8da3b66b0210`。在实验目录执行，`WORK` 须是仓库外的新目录：

```bash
python -m pip install -r requirements-eval.txt
python -m pip check
WORK=/path/to/new-work-directory
mkdir -p "$WORK"
hf download mteb/T2Retrieval --repo-type dataset --revision 921dd3af6e78d1ae7ee0368aa8d7eaee02c8f08e --include '*.parquet' --local-dir "$WORK/dataset"
hf download BAAI/bge-base-zh-v1.5 --revision f03589ceff5aac7111bd60cfc7d497ca17ecac65 --local-dir "$WORK/base-model"
sha256sum "$WORK/dataset"/*/*.parquet
python scripts/t2retrieval_subset.py prepare --source "$WORK/dataset" --output "$WORK/subset"
python scripts/t2retrieval_subset.py evaluate --data "$WORK/subset/eval.json" --model "$WORK/base-model" --output "$WORK/baseline.json"
python scripts/train_embedding.py --model_name_or_path "$WORK/base-model" --train_dataset "$WORK/subset/train.jsonl" --output_dir "$WORK/model" --epochs 1 --batch_size 8
python scripts/t2retrieval_subset.py evaluate --data "$WORK/subset/eval.json" --model "$WORK/model/checkpoint-epoch-1" --output "$WORK/finetuned.json"
```

数据划分固定随机种子 42；训练脚本未固定随机种子，数值不保证一致。候选池包含测试查询的已标注正例及 200 个随机文档，`Hit@10` 表示前十篇至少命中一个已标注相关文档的查询比例。

## 使用自己的数据

准备 UTF-8 编码的 JSONL 文件，每行一个 JSON 对象，包含 `query` 和 `pos_doc`：

```jsonl
{"query":"如何学习 Python？","pos_doc":"学习 Python 可以从安装环境、变量和循环等基础开始。"}
{"query":"什么是机器学习？","pos_doc":"机器学习研究如何让计算机从数据中学习。"}
```

将查询和能够回答该查询的文档段落配对；尽量覆盖真实查询、领域表达和容易混淆的候选内容。训练集和评估集应事先分开，并检查重复或高度相似的查询与文档，减少数据泄漏。文档和查询分别按 `--passage_max_len`、`--query_max_len` 截断，数据准备时需要确认关键内容没有被截掉。

训练时，模型分别编码查询和正例段落，得到各自的向量；损失函数使用同批次向量之间的相似度更新模型。检索时，先用训练后的模型将候选段落分别编码并建立向量索引；收到新查询后再将查询编码，与候选向量计算相似度，返回分数较高的段落。模型输出的是向量，检索结果的排序来自后续的相似度计算，而不是模型直接生成答案。

```bash
python "$REPO/experiments/embedding-finetuning/scripts/train_embedding.py" \
  --model_name_or_path BAAI/bge-base-zh-v1.5 \
  --train_dataset path/to/train.jsonl \
  --output_dir path/to/output \
  --epochs 3 \
  --batch_size 4
```

其余可配置参数为 `--lr`、`--query_max_len` 和 `--passage_max_len`；完整默认值见 [`train_embedding.py`](scripts/train_embedding.py)。脚本不支持显式难负例、验证集评估或自动选择最佳检查点，需要时应另行实现。

## 怎样判断微调是否有用

在独立评估集上，使用同一文档库、切分方式、查询和检索配置，分别测试基础模型和微调后的模型。根据任务比较 Recall@K、MRR 或 NDCG 等指标，并检查真实失败案例：相关文档是否进入候选集、排在什么位置、专业问法是否改善，以及原本表现好的查询有没有退化。选定指标和 K 值时应结合下游实际会读取多少条候选文档。

脚本显示的 `acc` 是**当前训练批次内**正例的排名准确率，`loss` 也是训练批次上的损失；两者都不是独立检索评估结果，不能据此宣称模型泛化能力或业务效果提升。脚本本身不执行上述评估，实际收益需要自行运行对照实验确认。

## 当前示例的范围

内置数据是 4 条不同主题的演示样本；默认批次大小为 4，所以每轮只有一个批次。它展示了数据读取、对比损失、参数更新和模型保存的代码路径，并不代表具备足够覆盖面的训练集，也不包含独立验证集、基线对照或实际业务检索结果。实际部署前，应先补齐数据、评估及效果对照，再决定是否采用微调模型。
`

## 长文本切分策略对比基准

若需评测长文本不同切分策略（朴素独立切分 Naive Chunking、滑动窗口重叠切分 Overlap Chunking 与全局注意力迟分 Late Chunking）在跨段落代词消解与同文均质化上的表现，可运行基准评测脚本：

```bash
python scripts/benchmark_late_chunking.py
```

该脚本以 768 维随机合成向量模拟文档和注意力，比较三种策略下的代词片段余弦相似度、召回率、同文片段相似度与单篇耗时；它没有加载真实文本或训练好的编码模型，结果不能代表实际检索效果。结果输出至 `evidence/results_late_chunking.json`，重复运行会覆盖该文件；对应原理见[长文切分与迟分](../../docs/zh/retrieval/embedding/长文切分与迟分.md)。
