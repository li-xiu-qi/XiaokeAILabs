# 训练自己的 AgentJev（开源版 Jev 决策模型）

通用决策模型在具体任务上是否适用，需要用目标场景的数据检验。本目录提供从公开数据和底座出发，全参微调判别式决策模型，并依次选优、校准、测试和启动服务的训练路线。

模型代码来自 `malevrigns/agent-jev`（Apache-2.0），这里只保留训练必需的部分：底座用 Qwen3-0.6B，约 1.5 GB；数据集是 HuggingFace 上的 `LocalLLaMA/typed-decisions`，两个 parquet 加起来不到 0.8 MB。

## 目录结构

```
text_jev_train/
├── agentjev/           # 模型代码（Qwen3 骨干 + 候选头）
├── jev_service/        # 输入协议、编码（含共享前缀加速）、推理引擎、HTTP 服务
├── scripts/
│   ├── download_data.py       # 下载并校验数据集
│   ├── download_backbone.py   # 下载 Qwen3-0.6B 底座
│   ├── prepare_data.py        # parquet 转 jsonl，按案例可复现切分
│   ├── train.py               # 训练、选优、校准、测试
│   ├── serve.py               # 起推理 HTTP 服务
│   └── protocol.json          # 训练超参
├── data/               # 原始 parquet（下载生成）
├── prepared/           # 切分后的 jsonl（切分生成）
└── runs/               # 训练产物（checkpoint、报告）
```

## 环境

需要一张 CUDA 显卡，训练全程在 GPU 上跑。Python 3.10 以上，先装依赖：

```bash
pip install -r requirements.txt
# torch 按本机 CUDA 版本自行安装，例如 CUDA 12.x:
# pip install torch --index-url https://download.pytorch.org/whl/cu121
```

国内网络访问 HuggingFace 不稳定，下面的下载命令统一加 `--mirror` 走 hf-mirror。

## 训练流程（四步）

### 1. 下载数据集和底座

```bash
python scripts/download_data.py --mirror
python scripts/download_backbone.py --mirror
```

数据集两个文件：train 1200 个案例、test 400 个案例，每个案例 5 道决策题。下载时逐份校验 sha256，不匹配直接报错。底座默认落到 `models/Qwen3-0.6B-Base`。

### 2. 切分数据

```bash
python scripts/prepare_data.py
```

切分在案例层面做，先切再展开问题。1200 个训练案例按固定种子哈希后切成训练 960、dev 120、校准 120，test 400 个案例保持原样。切完 `prepared/` 下每个划分有 questions 和 requests 两份 jsonl，同时产出 `split_manifest.json`，里面有全部案例 id 和源文件哈希，方便核对。

### 3. 训练

```bash
python scripts/train.py \
  --backbone models/Qwen3-0.6B-Base \
  --run-dir runs/agentjev_v1
```

全参微调，不是 LoRA。600 步，骨干学习率 1e-5、候选头 1e-4，bf16。每 100 步在 dev 上评一次，按未校准的软交叉熵选 checkpoint。选完才用独立的校准划分拟合每类问题的温度，最后才读 test。顺序由代码固定，test 不会提前泄漏到选型里。

设备日志记录训练约 49 分钟；test 2000 题中，预测最高分选项与参考分布最高项一致率为 78.05%。这不是现实任务正确率：typed-decisions 是合成案例，其软参考分布来自教师模型，并非真实事件结局；2000 题来自 400 个案例，同案例的题目并不独立。记录为单次运行，也不能据此判断训练效果在其他任务上能否保持。训练日志在设备侧，仓库未收录逐题预测，不能仅凭本文独立复算结果。

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
