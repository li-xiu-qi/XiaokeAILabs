# ColBERT 蒸馏实验：代码与复现

历史结果按原始运行条件解读。现有脚本已加入独立开发集上的师生优势门槛：只有教师的 nDCG@10 比初始学生至少高 0.05，且逐查询配对 bootstrap 95% 区间下限大于 0，才进行训练；两组训练结束后再评估测试集。这是本实验的设定，不是所有任务通用的阈值。新流程已在不完整依赖回退环境下完成一次真实模型训练与测试，具体范围见下文。

[English](README.en.md)

本实验考察单向量 Qwen3-Embedding-8B 教师的候选评分能否改善多向量 ColBERT 学生的排序，并用相同学生、训练样本和预算的标签监督组作对照。教师只参与训练目标的生成；部署学生时不需要教师。蒸馏本身也是训练，教师分数不等于人工真值。动机、原理、数据和结果解读见[完整实验讲解](../../docs/zh/retrieval/colbert/文本多向量检索模型蒸馏.md)。

## 复现步骤

### 环境准备

原实验运行于 DGX Spark 的 Linux ARM64 和 CUDA 环境；当前完整流程依赖 Linux 的 `/proc/meminfo`，训练固定使用 CUDA，教师打分使用 BF16。请准备支持 CUDA 和 BF16 的 NVIDIA GPU、匹配的驱动与 PyTorch，以及足够的显存和内存。脚本在教师 CUDA 打分和正式训练时要求系统可用内存不低于 50 GiB，这是运行保护阈值，并非计算出的通用最低硬件需求。8B 教师权重、数据和输出需要额外磁盘空间。

在独立环境中执行以下命令，本依赖组合使用 Python 3.12 或以上（数据准备脚本自身要求 3.11 以上，已记录的 NumPy、SciPy 版本要求 3.12 以上）；在安装其他包之前，先按设备和 CUDA 驱动安装兼容的 PyTorch，确保 `torch.cuda.is_available()` 为真：

```bash
python3 -m venv .venv
source .venv/bin/activate
# 在此安装适用于本设备的 CUDA PyTorch：https://pytorch.org/get-started/locally/
python -c "import importlib.metadata as m; print('torch==' + m.version('torch'))" > torch-constraint.txt
python -m pip install -c torch-constraint.txt -r requirements.txt
python -m pip check
python -c "import torch, numpy, sentence_transformers, pylate; print('torch:', torch.__version__, 'CUDA:', torch.version.cuda, 'available:', torch.cuda.is_available()); print('numpy:', numpy.__version__)"
```

安装前生成的约束文件固定已安装的 PyTorch 版本，避免 pip 为满足依赖而替换 CUDA 构建。2026-11-15 在 Linux ARM64 的全新虚拟环境中从零安装验证：PyTorch 2.15.0+cu132 下 `requirements-base.txt` 的固定版本全部可解析安装，原先未固定的 pandas、sqlitedict、voyager 分别解析为 3.1.1、2.1.0、2.2.3，已按验证结果固定。该文件不是完整锁文件，其余传递依赖仍随解析变化。`pip check` 与导入检查都不能代替真实模型验证。

PyLate 1.4.0 的包元数据将 `fast-plaid` 列为必需依赖，即使本实验只使用固定候选重排。2026-11-15 探针确认：默认索引上没有适用于 Linux ARM64（CPython 3.12）的 fast-plaid 发行版，wheel 与源码包均不可安装（pip 报 `from versions: none`）；各版本还约束 PyTorch 版本，例如历史运行的 PyTorch 2.14.0+cu130 不满足这些约束。正常安装失败时不会自动降级 PyTorch。本实验不构建 PLAID 索引，在 Linux ARM64 上可尝试下面的**不完整依赖回退**：

```bash
python -m pip install -c torch-constraint.txt -r requirements-base.txt
python -m pip install --no-deps pylate==1.4.0
python -m pip check
python -c "import torch, numpy, sentence_transformers, pylate; print(torch.__version__, torch.cuda.is_available())"
```

该回退省略 fast-plaid，`pip check` 会报告它缺失，也可能报告其他依赖冲突；必须检查全部输出。回退仅供本实验固定候选路径排查，并非满足 PyLate 完整依赖的安装，不能据此声称索引功能可用或新环境复现成功。2026-11-15 在 Linux ARM64 全新虚拟环境中从零执行该回退：安装成功，`pip check` 仅报告 `pylate 1.4.0 requires fast-plaid, which is not installed`，`torch`、`numpy`、`sentence_transformers`、`pylate` 导入正常且 CUDA 可见（详见 [env-validation-20261115.json](evidence/env-validation-20261115.json)）。历史设备的 PyLate 路径与 2026-11-14 的导入检查仍按原条件解读。

2026-11-14 在 Linux ARM64 的独立复制环境中按回退清单安装，PyTorch 2.15.0+cu132、Transformers 4.56.2、Sentence Transformers 5.1.1 和 PyLate 1.4.0 可导入，CUDA 可见；`pip check` 报告缺少 `fast-plaid`。这是不完整依赖下的导入检查，不证明真实模型训练或从零安装成功。

同日进一步在该回退环境加载固定 revision `0b9b1e380096a3b72b6b4877f68d59effb62e0f2` 的 ColBERTv2 学生，在 CUDA 上对一条示例查询的两条候选完成编码及重排。它只验证固定候选重排路径；`fast-plaid` 仍缺失，不能据此声称完整依赖或正式训练已通过。

2026-11-14 使用固定版本的 Qwen3-Embedding-8B 教师和 ColBERTv2 学生，按新流程完成 SciFact 真实数据运行。89 条开发查询上，教师 nDCG@10 为 0.7171，学生为 0.6434，差值 0.0737，配对 bootstrap 95% 区间 [0.0232, 0.1318]，达到门槛；监督和蒸馏组各训练 128 步。300 条测试查询上，基线、监督和蒸馏组 nDCG@10 分别为 0.6536、0.6686、0.6657；蒸馏减监督为 -0.0029，配对 bootstrap 95% 区间 [-0.0111, 0.0051]，没有显示稳定优势。测试查询 ID 与训练、开发集不重叠，但部分测试正例文档出现在训练候选；结果仅对应固定候选池、一个训练种子和当前设备。学生在缺少 fast-plaid 的回退环境运行，不证明完整 PyLate 安装或索引功能可用。

### 使用合成数据检查 GPU 训练

若只需检查蒸馏训练函数能否使用 GPU，可在另一个独立环境尝试 PyLate 1.2.0。它没有 PyLate 1.4.0 的 `fast-plaid` 必需依赖，但两版的损失函数实现不同，因此这个检查不能复现 1.4.0 的实验结果，也不能解决上面的 1.4.0 ARM64 依赖问题。

在实验目录运行，先安装适合设备的 CUDA PyTorch，再安装其余依赖。Linux ARM64 与 CUDA 13 设备可用下面这条已验证的安装命令，其他设备按 [PyTorch 官方指南](https://pytorch.org/get-started/locally/) 选择匹配构建并约束其版本：

```bash
python3.12 -m venv .venv-smoke
source .venv-smoke/bin/activate
python -m pip install 'torch==2.15.0' --index-url https://download.pytorch.org/whl/cu132
python -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available()"
python -c "import importlib.metadata as m; print('torch==' + m.version('torch'))" > .venv-smoke/torch-constraint.txt
python -m pip install -c .venv-smoke/torch-constraint.txt -r requirements-smoke.txt
python -m pip check
python scripts/gpu_synthetic_smoke.py
```

安装期间若设备预设了代理环境变量，需确认代理可用或先清除这些变量，否则 pip 会在连接阶段立即失败。

脚本在临时目录随机初始化微型 BERT，构造四条查询、候选文档和教师分数，直接调用 `run.py` 的蒸馏训练函数，在 CUDA 上训练两步；结束后自动清理临时模型。它保留该函数的系统可用内存不低于 50 GiB 的运行保护。脚本不下载模型或数据，也不执行教师打分、正式门槛、监督组或测试集评估。

2026-11-14 的合成训练在 Linux ARM64 的 DGX Spark GPU 上运行，使用 Python 3.12、PyTorch 2.15.0+cu132 和 PyLate 1.2.0，当时通过独立虚拟环境复用已有 PyTorch 等依赖，`pip check` 未报告依赖冲突。2026-11-15 在全新虚拟环境中从零执行上述安装命令并复跑该脚本：`pip check` 报告 `No broken requirements found`，脚本在 CUDA 上完成两步训练（首步 loss 1.922，0.35 秒），解析出的传递依赖版本记录在 [env-validation-20261115.json](evidence/env-validation-20261115.json)。

### 数据与模型

从实验目录执行，以下文件放在个人工作目录，可按需替换路径。下载和解压需要 `curl`、`unzip`、`tar` 与网络；Python 数据准备脚本本身只用标准库。将下面的模型路径分别传给命令中的 `--model`。模型须完整下载，学生模型包括 `1_Dense/` 投影层，运行脚本以本地文件加载：

```bash
mkdir -p output
curl -fL https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip -o output/scifact.zip
curl -fL https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz -o output/scifact-original.tar.gz
unzip -q output/scifact.zip -d output
tar -xOzf output/scifact-original.tar.gz data/claims_train.jsonl > output/claims_train.jsonl
huggingface-cli download Qwen/Qwen3-Embedding-8B --local-dir output/teacher
huggingface-cli download lightonai/colbertv2.0 --revision 0b9b1e380096a3b72b6b4877f68d59effb62e0f2 --local-dir output/student
```

教师来自 [Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B)，学生来自 [ColBERTv2](https://huggingface.co/lightonai/colbertv2.0)。上述学生 revision 是同系列微调实验的快照，教师命令未固定 revision；**原蒸馏运行的模型 revision 和权重哈希尚未确认**，执行时应记录实际下载的 revision 与哈希。数据处理与许可见[实验讲解](../../docs/zh/retrieval/colbert/文本多向量检索模型蒸馏.md)。

下载后可计算源文件哈希，并与 [source-hashes.json](configs/source-hashes.json) 的同名条目逐项对照；原始 claims 对应 `original_claims_train.jsonl`。若不一致，先核对来源和版本。该文件记录数据与候选池哈希，不含模型哈希：

```bash
sha256sum output/scifact/corpus.jsonl output/scifact/queries.jsonl \
  output/scifact/qrels/train.tsv output/scifact/qrels/test.tsv output/claims_train.jsonl
# 准备候选池后，另行比对 pools.json 条目
sha256sum /path/to/prepared/pools.json
```

将实验目录复制到自己的工作目录，在副本中执行以下命令；路径替换为自己的数据、模型和输出位置：

```bash
python scripts/prepare_pools.py --data /path/to/scifact --original-claims /path/to/claims_train.jsonl --config configs/pool_config.json --out /path/to/prepared
python scripts/prepare_teacher.py --data /path/to/scifact --pools /path/to/prepared/pools.json --model /path/to/Qwen3-Embedding-8B --device cuda --out /path/to/source --min-teacher-gap 0.05
python scripts/run.py --data /path/to/scifact --pools /path/to/prepared/pools.json --source /path/to/source --model /path/to/colbert --output /path/to/output --min-teacher-gap 0.05 --smoke
```

第一步准备候选池，第二步只为训练集和开发集生成教师分数并记录预设门槛，第三步检查输入和门槛。去掉 `--smoke` 才会运行开发集师生比较；只有通过门槛才训练监督组和蒸馏组，最后评估测试集。教师准备输出目录须为空，训练时的门槛必须与教师打分前声明的一致。`prepare_teacher.py --synthetic-encoder` 使用伪向量，只适合检查 CPU 接口，不能用于正式训练。`run.py` 的 50 GiB 可用内存下限是原设备的保护设置，不是通用最低要求。

**复现范围：**候选池已在 CPU 上重新生成，`pools.json` 的 SHA-256 与原文件一致。新的门槛流程改变了历史脚本执行路径，不应视为历史结果的逐步骤复现；本次已用真实 8B 教师完成新的训练与测试路径，学生环境缺少 fast-plaid，原始蒸馏运行的确切环境也未确认。2026-11-15 另在全新隔离环境中从零验证了两套依赖安装与合成数据 GPU 冒烟，未下载模型权重或基准数据，范围与限制见 [env-validation-20261115.json](evidence/env-validation-20261115.json)。

## 文件说明

- [prepare_pools.py](scripts/prepare_pools.py)、[pool_config.json](configs/pool_config.json)：准备和配置候选池。
- [prepare_teacher.py](scripts/prepare_teacher.py)：生成训练候选及教师分数。
- [run.py](scripts/run.py)：训练学生并评估。
- [gpu_synthetic_smoke.py](scripts/gpu_synthetic_smoke.py)：用合成数据检查 CUDA 蒸馏训练函数。
- [summary.json](evidence/summary.json)、[per-query-metrics.csv](evidence/per-query-metrics.csv)：汇总结果与逐查询指标；[recompute_metrics.py](scripts/recompute_metrics.py) 用于指标复算。
- [requirements-smoke.txt](requirements-smoke.txt)：合成数据 GPU 冒烟检查的依赖清单；[env-validation-20261115.json](evidence/env-validation-20261115.json)：2026-11-15 全新环境从零安装与冒烟验证记录。

逐查询指标不包含原始文档排序，不能单独用于重新检验每条评分。
