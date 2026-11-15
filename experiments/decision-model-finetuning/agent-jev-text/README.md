# 训练自己的 AgentJev（开源版 Jev 决策模型）

[English](README.en.md)

通用决策模型在具体任务上是否适用，需要用目标场景的数据检验。本目录提供从公开数据和底座出发，全参微调判别式决策模型，并依次选优、校准、测试和启动服务的训练路线。

模型代码来自 `malevrigns/agent-jev`（Apache-2.0），这里只保留训练必需的部分：底座用 Qwen3-0.6B，约 1.5 GB；数据集是 HuggingFace 上的 `LocalLLaMA/typed-decisions`，两个 parquet 加起来不到 0.8 MB。

## 目录结构

```
agent-jev-text/
├── agentjev/           # 模型代码（Qwen3 骨干 + 候选头）
├── jev_service/        # 输入协议、编码（含共享前缀加速）、推理引擎、HTTP 服务
├── scripts/
│   ├── download_data.py       # 下载并校验数据集
│   ├── download_backbone.py   # 下载 Qwen3-0.6B 底座
│   ├── prepare_data.py        # parquet 转 jsonl，按案例可复现切分
│   ├── train.py               # 训练、选优、校准、测试
│   ├── smoke_gpu.py           # 合成数据单步 GPU 冒烟检查
│   ├── serve.py               # 起推理 HTTP 服务
│   └── protocol.json          # 训练超参
├── data/               # 原始 parquet（下载生成）
├── prepared/           # 切分后的 jsonl（切分生成）
└── runs/               # 训练产物（checkpoint、报告）
```

## 环境

需要一张支持 bfloat16 的 CUDA 显卡，训练全程在 GPU 上跑。Python 3.10 以上。建议先创建独立环境，按设备安装兼容的 CUDA 版 PyTorch，再安装其余依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
# 先按设备选择并安装兼容的 CUDA 版 PyTorch
python -m pip install -r requirements.txt
# 本次新环境验证使用该兼容版本
python -m pip install "transformers==5.16.1"
python -c "import torch, transformers, pyarrow; from transformers import Qwen3Model; print('torch:', torch.__version__, 'CUDA:', torch.cuda.is_available(), 'transformers:', transformers.__version__); assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()"
```

`requirements.txt` 含版本下限，并非完整锁定文件。2026-11-15 在新建隔离环境中验证了安装与 GPU 冒烟链路：Python 3.12.3、PyTorch 2.15.0+cu132、Transformers 5.16.1、PyArrow 26.0.1；`pip check` 通过，NVIDIA GB10 上 CUDA 与 bfloat16 均可用。由于 Transformers 最新版本与脚本依赖需显式选定兼容版本，这次验证在安装 requirements 后将 Transformers 固定到 5.16.1。检查仅用合成数据训练一步并确认参数改变，完成初始模型选优、校准和报告；不代表真实底座训练或 600 步训练在此干净环境中复现。该设备本次运行的摘要见 [环境与 GPU 冒烟记录](evidence/clean-env-gpu-smoke-2026-11-15.json)。CUDA/bfloat16 可用也不代表其他显卡有足够显存。下载需预留模型、数据及训练产物的磁盘空间。下面的下载命令使用 `--mirror` 走 hf-mirror；不需要镜像时可去掉该参数。

一次 GPU 冒烟检查在 NVIDIA GB10、Python 3.12、PyTorch 2.15.0+cu132 和 Transformers 5.22.2 的现有环境中通过。检查用随机初始化的微型 Qwen3 和合成数据，实际训练一步并断言 GPU 参数变化，再执行初始模型选优、校准和报告；它不验证真实底座或历史指标。该环境的 `pip check` 对 `nvidia-cusparselt-cu13` 仍报平台兼容问题，不能视为全新环境安装通过。

```bash
python scripts/smoke_gpu.py --out-dir /path/to/new-smoke-output
```

输出目录必须不存在。另一次检查使用了真实 Qwen3-0.6B 底座和 typed-decisions 数据：每个划分取 1 个案例的 5 题，实际 CUDA 训练 1 步并确认参数变化，随后完成选优、校准和测试。底座 revision 为 `c1899de289a04d12100db370d81485cdf75e47ca`；这只验证小样本链路，不代表 600 步训练和历史指标已复现。

在完成下面的数据下载和切分后，可用以下命令检查真实底座与数据。`/tmp/agentjev-mini` 和输出目录须为新路径；如果已按第 1 步下载底座，需确认其 revision 对应上述版本：

```bash
mkdir /tmp/agentjev-mini
for split in train dev calibration test; do
  head -n 5 "prepared/${split}_questions.jsonl" > "/tmp/agentjev-mini/${split}_questions.jsonl"
done
python scripts/smoke_gpu.py --backbone models/Qwen3-0.6B-Base \
  --prepared /tmp/agentjev-mini --out-dir /tmp/agentjev-real-smoke
```

新下载底座时，可在第 1 步的命令末尾追加 `--revision c1899de289a04d12100db370d81485cdf75e47ca`。单步检查因不足 100 步会选第 0 步初始模型进行校准和测试；参数变化断言用于验证训练步骤，不代表选优结果是微调后的模型。

Qwen3 底座要求 Transformers 至少 4.51.0，依据见[上游模型卡](https://huggingface.co/Qwen/Qwen3-0.6B)。这个下限说明模型支持，不代表所有后续版本都已验证。

## 训练流程（四步）

以下命令从 `agent-jev-text/` 目录执行。

### 1. 下载数据集和底座

```bash
python scripts/download_data.py --mirror
python scripts/download_backbone.py --mirror
```

数据集两个文件：train 1200 个案例、test 400 个案例，每个案例 5 道决策题。下载时逐份校验 sha256，不匹配直接报错。底座默认落到 `models/Qwen3-0.6B-Base`，目标目录必须不存在或为空；默认下载上游 `main`；可用 `--revision <commit>` 固定模型版本，并记录它以便复现。

数据默认写入 `data/all/`。下载前会检查两个目标文件，任一已存在时拒绝下载和覆盖；两个文件均校验成功后才写入目标位置。也可用 `python scripts/download_data.py --out-dir /path/to/empty-data-dir` 下载到独立目录。使用独立目录时，通过 `prepare_data.py --data-dir /path/to/data-dir` 指定输入；用 `--out-dir /path/to/empty-output-dir` 指定切分结果目录。

脚本使用与输出目录同一文件系统内的临时文件及硬链接发布数据，目标文件系统须支持硬链接。普通下载错误或 Ctrl+C 会清理临时文件；发布阶段发生错误或中断时，已经发布的完整文件会保留，请检查目标目录或选择新目录重试。若进程被强制终止，还可能留下 `.download-*` 临时目录；重新运行仍会保护已有目标。

### 2. 切分数据

```bash
python scripts/prepare_data.py
```

公开数据的下载、SHA-256 校验和以下案例划分已在 Linux 上实际执行。 切分在案例层面做，先切再展开问题。1200 个训练案例按固定种子哈希后切成训练 960、dev 120、校准 120，test 400 个案例保持原样。输出目录必须不存在或为空，防止覆盖已有文件；切完 `prepared/` 下每个划分有 questions 和 requests 两份 jsonl，同时产出 `prepared/split_manifest.json`，里面有全部案例 id 和源文件哈希，方便核对。

### 3. 训练

```bash
python scripts/train.py \
  --backbone models/Qwen3-0.6B-Base \
  --run-dir runs/agentjev_v1
```

全参微调，不是 LoRA。600 步，骨干学习率 1e-5、候选头 1e-4，bf16。初始模型也参与选优，每 100 步在 dev 上评一次，按未校准的软交叉熵选 checkpoint。若训练未优于初始模型，则保留第 0 步模型。训练输出目录必须不存在或为空。选完才用独立的校准划分拟合每类问题的温度，最后才读 test。顺序由代码固定，test 不会提前泄漏到选型里。

设备日志记录训练约 49 分钟；test 2000 题中，预测最高分选项与参考分布最高项一致率为 78.05%。这不是现实任务正确率：typed-decisions 是合成案例，其软参考分布来自教师模型，并非真实事件结局；2000 题来自 400 个案例，同案例的题目并不独立。记录为单次运行，也不能据此判断训练效果在其他任务上能否保持。训练日志在设备侧，仓库未收录逐题预测，不能仅凭本文独立复算结果。

2026-11-15 使用固定底座 revision `c1899de289a04d12100db370d81485cdf75e47ca` 和数据 revision `ea9306458d6e9563628369a3d1e72e362fb381d2` 重跑上述 600 步：开发集选中第 600 步，2000 道测试题的参考标签一致率为 78.05%；校准前后软交叉熵分别为 0.8584 和 0.8571。训练峰值 CUDA 分配约 33.65 GB。运行使用独立复制环境，Python 3.12、PyTorch 2.15.0+cu132、Transformers 5.16.1，`pip check` 通过，但不是从零安装验证。测试数据已有历史结果，不是新的盲测；这里的校准指标也只针对教师参考分布。

### 4. 起服务验证

```bash
python scripts/serve.py \
  --run-dir runs/agentjev_v1 \
  --backbone models/Qwen3-0.6B-Base \
  --port 8149
```

服务默认只监听本机。POST 一个决策请求到 `http://127.0.0.1:8149/api/evaluate`：

```bash
curl -s http://127.0.0.1:8149/api/evaluate \
  -H 'Content-Type: application/json' \
  -d '{
    "id": "demo-1",
    "state": "我这个月信用卡被扣了两次费，请尽快把多扣的钱退给我",
    "questions": [{
      "id": "route",
      "type": "choice",
      "question": "这个请求应该由哪个部门处理？",
      "options": ["账务", "技术", "销售", "其他"]
    }]
  }'
```

返回每个选项的概率和胜出项，不生成文本。延迟随设备、上下文长度、候选数量和服务配置变化，应在自己的负载上测量。

编码默认 `auto`：同一道题的多个候选共享 `[STATE]+[QUESTION]` 前缀的 KV 计算（一次算前缀，各候选只算自己的后缀）；怀疑编码问题时可用 `--encoder path` 逐路径复跑对照。不同实现和精度下应核对输出误差与实际延迟。

## 换成自己的数据

方法链不依赖官方数据集。一个案例就是一段 state、若干问题、每个问题的候选描述加教师分布，字段和官方行同构。照 `scripts/prepare_data.py` 里的 `convert` 和切分断言改成自己的字段即可。注意三条：target 写成概率分布而不是单点标签，只把 state、问题、候选描述放进输入，自己留一份不参与选型和温度拟合的测试集。

## 上游来源

| 项 | 地址 |
|---|---|
| AgentJev 模型仓库 | https://github.com/malevrigns/agent-jev |
| 数据集 | https://huggingface.co/datasets/LocalLLaMA/typed-decisions |
| 底座 | https://huggingface.co/Qwen/Qwen3-0.6B |
