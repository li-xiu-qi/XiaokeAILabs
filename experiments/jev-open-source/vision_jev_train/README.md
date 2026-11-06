# 训练自己的视觉版 Jev（多模态决策模型）

> **当前结果不能证明独立视觉答题能力**：数据准备把参考答案推理 `solution` 加入模型输入，训练与评估存在目标信息泄漏。下面保留历史流程供检查；应先移除泄漏字段并重新生成数据、训练和评估，才能判断效果。

`text_jev_train` 走的是纯文本路线。这个目录把同样的判别式决策模型搬到视觉上：底座换成本身就懂图像的多模态模型，每条候选同时吃图像、上下文和候选文本，由 score 头给出一个标量分，再在本问题的候选之间做归一。

模型侧没有现成的开源训练实现可抄（AgentJev 是纯文本，Jev-Omni 训练权重不开源），这里是自建路线，参考了 `pngwn/system-one-qwen3.5-4b-scorer` 的架构组织方式（LoRA 加 scalar score 头、按问题归一、温度校准），但数据、协议、训练脚本都独立写。

## 用到的东西

| 项 | 取值 | 说明 |
|---|---|---|
| 底座 | Qwen3.5-2B | Apache 2.0，原生多模态，权重约 4 GB |
| 数据集 | ScienceQA | CC BY-SA 4.0，非 gated，图像内嵌在 parquet 里 |
| 参数高效微调 | LoRA r=16 | 本实验使用 LoRA；设备能否全参训练取决于模型、批次、精度等配置 |
| 损失 | 软交叉熵 + 0.1×sum-candidate Brier | 与文本版一致 |
| 校准 | 单一标量温度，81 点网格 | 只在 calibration 集上拟合 |

底座也可以用 `Qwen3.5-0.8B`（权重 1.7 GB）或 `4B`（权重 8.8 GB），改 `download_backbone.py --size` 和 `protocol.json` 即可。

## 目录结构

```
vision_jev_train/
├── scripts/
│   ├── download_backbone.py   # 下载 Qwen3.5 多模态底座
│   ├── download_data.py       # 下载 ScienceQA test split
│   ├── prepare_data.py        # parquet 转 jsonl，按图片 sha256 去重切分
│   ├── train.py               # 训练、内存闸门、选优、校准、测试
│   ├── evaluate.py            # 独立评测（图像预 resize，避开 processor 单线程瓶颈）
│   ├── serve.py               # 起推理 HTTP 服务
│   └── protocol.json          # 实验协议与超参
├── data/               # 原始 parquet（下载生成）
├── models/             # 底座（下载生成）
├── prepared/           # 切分后的 jsonl（切分生成）
└── runs/               # 训练产物（adapter、报告）
```

## 流程

### 1. 下载底座和数据

```bash
python scripts/download_backbone.py --size 2b --mirror
python scripts/download_data.py
```

### 2. 切分数据

```bash
python scripts/prepare_data.py data/all/test-00000-of-00001.parquet
```

产出 `prepared/{train,dev,calibration,test}.jsonl` 和 `split_manifest.json`。

切分主键是图片 SHA-256，不是行 ID；同图不跨划分有助于降低图片泄漏，但不能消除题目、文字或语义重复，也无法修复下述参考答案推理进入输入的问题。实际行数和丢弃数以生成的 `split_manifest.json` 为准。

每行字段：

| 字段 | 含义 |
|---|---|
| `image_png_b64` | 图像 base64 |
| `state` | 题目上下文，讲解加备选答案加提示加参考答案推理 |
| `question` | 判别问题 |
| `options` | 候选列表 |
| `gold` | 概率分布，正确项 1.0 其余 0 |

### 3. 训练

```bash
python scripts/train.py \
  --model models/Qwen3.5-2B \
  --data prepared \
  --out runs/v2 \
  --steps 1200 \
  --bs 4
```

训练前会先读 `/proc/meminfo` 的 `MemAvailable`，低于 `--mem-min-start`（默认 25 GB）直接拒绝启动。训练中每步采样内存，可用量跌破阈值或峰值超过 `--mem-cap`（默认 70 GB）就停训、保存现场、写 `aborted.json` 然后退出，不进入评估。

这个闸门不是可选项。Qwen3.5 是 Gated DeltaNet 加门控注意力的混合架构，激活开销模型和普通 transformer 不一样，`gradient_checkpointing_enable()` 默认的 reentrant 实现在这类非线性结构上可能静默失效。脚本里强制传了 `use_reentrant=False`，但如果模型或 batch 再放大，仍然可能把机器拖死，所以显式留一道闸。

### 4. 评测

训练脚本内置的评测会把同一张图交给 processor 处理多次（每个候选一次），
processor 的 smart_resize 是单线程 CPU 密集操作，1000 条测试集能跑十几分钟。
独立评测脚本先按长边和总像素双约束把图缩小，processor 只剩归一化可做：

```bash
python scripts/evaluate.py \
  --model models/Qwen3.5-2B \
  --ckpt runs/v1/final \
  --data prepared \
  --out runs/v1
```

温度网格是 0.5 到 4.5 共 41 点，只在 calibration 集上拟合。train.py 内置的
网格上限是 2.0，实测最优温度会触顶，需要更宽范围。

### 5. 起服务

```bash
python scripts/serve.py --ckpt runs/v2/final --model models/Qwen3.5-2B --port 8010
```

```bash
curl -X POST http://127.0.0.1:8010/decide -H 'Content-Type: application/json' -d '{
  "image_png_b64": "<base64>",
  "state": "讲解：...\n\n备选答案：\nA. ...\nB. ...",
  "question": "图中正在发生什么？请选出正确备选答案。",
  "options": ["A. ...", "B. ..."]
}'
```

返回 `probs`、`prediction`、`confidence`。

## 与 text_jev_train 的差异

| 维度 | text_jev_train | vision_jev_train |
|---|---|---|
| 底座 | Qwen3-0.6B | Qwen3.5-2B |
| 数据 | typed-decisions | ScienceQA |
| 微调方式 | 全参 | LoRA r=16 |
| 输入 | 上下文 + 候选文本 | 图像 + 上下文 + 候选文本 |
| 图像占位符 | 无 | `<|vision_start|><|image_pad|><|vision_end|>` |
| 切分主键 | case id | 图片 sha256 |
| 内存管理 | 无 | 启动体检 + 运行硬闸 |

## 已知边界

当前数据准备脚本把 ScienceQA 的 `solution`（参考答案推理）放进模型输入，训练和评估都存在目标信息泄漏。即使图片按 sha256 去重，现有结果也不能证明模型能在没有答案推理的条件下独立按图答题。需要移除答案相关字段、重新生成数据并重新训练和评测后，才能判断该能力。`gold` 是选择题标准答案，也不能直接代表真实行动的成功概率。

Qwen3.5 系列 2026 年 2 月起改为统一多模态基座，早期的 Qwen-VL 产品线已停更，不要用 `Qwen2-VL` 或 `Qwen2.5-VL` 的占位符和 processor 写法套这里。
