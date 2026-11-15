# 科学文献检索中的向量模型蒸馏

[English](README.en.md)

本目录保留历史实验与复现代码。脚本现已补充独立开发集师生比较：教师在开发集的 nDCG@10 至少高出学生 0.05，且逐查询配对 bootstrap 95% 区间下限大于零，才继续训练。这个 0.05 是本实验的预设门槛，不是所有蒸馏任务的统一标准；旧实验没有经过这项门槛，历史结果仍按原始运行条件解读。

本目录保存实验代码、配置、运行方式与结果证据。实验目的、原理和结果见[Embedding 蒸馏讲解](../../docs/zh/retrieval/embedding/Embedding蒸馏.md)；以下命令从本目录运行。

[scripts](scripts/) · [evidence](evidence/) · [assets](assets/)

## 复现与历史记录

运行 `python scripts/verify.py` 可直接用随附的每查询指标复算表格与配对 bootstrap；证据包含查询 ID 和指标，不含原始文本、模型或数据集。复算只核对聚合和区间，不能独立验证原始模型推理与标注。公开脚本由设备侧脚本参数化整理，已按下述步骤完成一次真实数据运行；它仍不是一键流程。

在 Linux CUDA 环境中准备 Python 3.11 或更高版本、PyTorch、NumPy 和 Sentence Transformers。数据准备脚本使用 Python 标准库；其候选池配置与重排序实验一致，生成的 `pools.json` 决定训练正例和 BM25 候选。数据集及模型按各自许可从原始来源获取，不包含在本仓库内。BEIR 数据卡标注 CC BY-SA 4.0；原始 SciFact 的标注与语料许可不同，使用前请参照[原始许可](https://github.com/allenai/scifact/blob/master/LICENSE.md)。

先创建独立的 Python 环境，按设备安装匹配的 PyTorch，再安装其余直接依赖：

```bash
python3 -m venv /path/to/venvs/retrieval
source /path/to/venvs/retrieval/bin/activate
# 按 https://pytorch.org/get-started/locally/ 安装与系统、架构和 GPU 匹配的 PyTorch
python -m pip install -r requirements.txt
python -m pip check
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA build:', torch.version.cuda, 'GPU:', torch.cuda.is_available()); assert torch.cuda.is_available(), '训练需要可用的 CUDA GPU'"
python -c "import torch; assert torch.cuda.is_bf16_supported(), '教师需要 BF16 支持'"
python -c "import numpy, sentence_transformers, huggingface_hub; print('依赖导入成功')"
```

2026-11-14 在单独复制并调整的 Python 3.12.3 环境中，`pip check` 通过；以 SciFact 原始数据运行候选池准备脚本，得到 390 条训练、89 条开发和 300 条测试查询，`pools.json` 的 SHA-256 为 `e7cac48e7083bdd948e2773136c1df9672b4ca8f787ee5b774697b32d689c5f8`。学生模型固定版本已准备，复制环境不代表从零安装已验证。

同日还完成一次正式流程：开发集 89 条查询上，教师较初始学生的 nDCG@10 高 0.0704，配对 bootstrap 95% 区间为 [0.0250, 0.1192]，达到预设门槛。监督与蒸馏学生各训练 128 步。300 条测试查询上的 nDCG@10 分别为基线 0.6654、监督 0.6702、蒸馏 0.6712、教师 0.7341；蒸馏减监督为 0.0010，配对 bootstrap 95% 区间为 [-0.0180, 0.0208]，不足以证明稳定收益。此结果依赖固定候选池、单个训练种子和本次环境；测试查询 ID 与训练、开发集不重叠，但部分测试正例文档也出现在训练候选中，不能视作对全新语料的评估。

同日用固定版本的 Qwen3-Embedding-8B 教师和 MiniLM 学生在 CUDA 上运行真实数据短测：教师对 4 条查询及其 68 篇候选文档编码评分，学生对 2 条开发查询的候选完成评分并计算门槛；短测按设计在训练和测试前停止。教师权重文件已逐个校验 SHA-256。两条开发查询不足以判断教师优势或实验效果。

[`requirements.txt`](requirements.txt) 固定了直接依赖版本，PyTorch 需按目标设备单独安装。以下数据准备命令还需 `curl`、`unzip`、`tar` 和 `sha256sum`，并需要为数据、模型和输出预留磁盘空间。

可先运行无需下载模型或数据的 GPU 冒烟检查：

```bash
python scripts/synthetic_gpu_smoke.py
```

该检查生成随机微型模型和合成数据，运行教师评分及学生阶段的开发集门槛；学生阶段的 `--smoke` 会在训练前停止。脚本随后直接调用训练函数，对监督和蒸馏损失分别训练两步并检查参数更新。这不代表正式实验通过门槛，也不复现真实数据上的结果。2026-11-13 在 Linux、NVIDIA GB10、Python 3.12.3、PyTorch 2.15.0+cu132、NumPy 2.5.4、Sentence Transformers 6.3.0、Transformers 5.22.2 和 huggingface_hub 2.4.0 的环境中该检查完成；该环境的 `pip check` 报告 `nvidia-cusparselt-cu13 0.8.1 is not supported on this platform`，因此这只是有限的运行证据，并非完整、可从零重建的依赖组合。

要从新环境安装依赖并运行相同的 GPU 冒烟，可使用：

```bash
export XL_REQUIRE_EXTERNAL_ENV=1
export XL_RUN_ID="$(date +%Y%m%d-%H%M%S)"
export XL_DATA_DIR="$HOME/xl-data"
export XL_OUT_DIR="$HOME/xl-output"
# 仅当继承的代理阻断下载时设置：export XL_CLEAR_PROXY=1
bash scripts/clean_env_gpu_smoke.sh
```

2026-11-15 在 Linux ARM64 / NVIDIA GB10 的新建虚拟环境中按固定依赖安装；`pip check`、CUDA 与 BF16 检查通过，合成 GPU 冒烟完成，监督和蒸馏损失各更新参数 2 步。运行记录见 [`evidence/clean-env-gpu-smoke-2026-11-15.json`](evidence/clean-env-gpu-smoke-2026-11-15.json)。本次没有下载真实模型或数据，也没有运行真实数据的教师门槛、完整蒸馏和测试集评估；从空环境完整复现仍未验证。

从本实验目录运行以下命令；把 `WORK` 改成仓库外的工作目录。数据、模型和输出均写入该目录：

```bash
export WORK=/path/to/new-working-directory
mkdir -p "$WORK/data"
curl -fL https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip -o "$WORK/data/scifact.zip"
curl -fL https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz -o "$WORK/data/scifact-original.tar.gz"
(cd "$WORK/data" && sha256sum -c <<'EOF'
536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165  scifact.zip
11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be  scifact-original.tar.gz
EOF
)
unzip -q "$WORK/data/scifact.zip" -d "$WORK/data"
tar -xOzf "$WORK/data/scifact-original.tar.gz" data/claims_train.jsonl > "$WORK/data/claims_train.jsonl"
python scripts/prepare_scifact.py --data "$WORK/data/scifact" \
  --original-claims "$WORK/data/claims_train.jsonl" --config configs/pool-config.json --out "$WORK/prepared"
DATA="$WORK/data/scifact"
POOLS="$WORK/prepared/pools.json"
```

下载两个模型并固定版本，例如：

```bash
python -c "import os; from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-Embedding-8B', revision='1d8ad4ca9b3dd8059ad90a75d4983776a23d44af', local_dir=os.path.join(os.environ['WORK'], 'models/teacher'))"
python -c "import os; from huggingface_hub import snapshot_download; snapshot_download('sentence-transformers/all-MiniLM-L6-v2', revision='1110a243fdf4706b3f48f1d95db1a4f5529b4d41', local_dir=os.path.join(os.environ['WORK'], 'models/student'))"
```

预先确定 `--min-teacher-gap`，两阶段必须一致，且本实验不得低于 0.05：

```bash
python scripts/train.py --stage teacher --data "$DATA" --pools "$POOLS" --out "$WORK" --teacher "$WORK/models/teacher" --student "$WORK/models/student" --min-teacher-gap 0.05
python scripts/train.py --stage student --data "$DATA" --pools "$POOLS" --out "$WORK" --teacher "$WORK/models/teacher" --student "$WORK/models/student" --min-teacher-gap 0.05
```

教师阶段只计算训练候选和开发集分数；学生阶段先在同一开发集、同一批候选上比较师生。通过门槛后，监督和蒸馏两组均训练完成并保存，再计算测试集分数。`--smoke` 只检查小规模开发集流程，不训练，也不评估测试集。脚本要求 Linux `/proc/meminfo` 和至少约 50 GiB 系统可用内存；正式实验使用 CUDA。真实数据的门槛、训练和测试流程此前已运行一次，但使用的是复制并调整的环境；全新环境验证目前只覆盖合成 GPU 冒烟。

历史流程中教师阶段生成分数，学生阶段评价原模型、监督组和蒸馏组。它会核对语料、查询和候选池的 SHA-256、模型路径与教师输出完整性；同一路径下模型内容发生变化不会被路径检查发现，仍需自行核验模型 revision 和权重。不同硬件、依赖版本与随机性会影响结果。
