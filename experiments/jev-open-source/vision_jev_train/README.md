# 训练自己的视觉版 Jev（多模态决策模型）

`text_jev_train` 走的是纯文本路线。这个目录把同样的判别式决策模型搬到视觉上：底座换成本身就懂图像的多模态模型，每条候选同时吃图像、上下文和候选文本，由 score 头给出一个标量分，再在本问题的候选之间做归一。

模型侧没有现成的开源训练实现可抄（AgentJev 是纯文本，Jev-Omni 训练权重不开源），这里是自建路线，参考了 `pngwn/system-one-qwen3.5-4b-scorer` 的架构组织方式（LoRA 加 scalar score 头、按问题归一、温度校准），但数据、协议、训练脚本都独立写。

## 用到的东西

| 项 | 取值 | 说明 |
|---|---|---|
| 底座 | Qwen3.5-2B | Apache 2.0，原生多模态，权重约 4 GB |
| 数据集 | ScienceQA | CC BY-SA 4.0，非 gated，图像内嵌在 parquet 里 |
| 参数高效微调 | LoRA r=16 | 4B 全参在 128 GB 统一内存上跑不动，LoRA 是必选项 |
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

切分的主键是图片 sha256，不是行 id。ScienceQA 的图在自己的 train 和 test 之间大量复用，实测同目录下 test 一开始有 4241 行，按图去重后能干净切出 train 3000、test 1000，同时丢弃 2529 张跨划分重复图。如果只按行 id 切，同一张图会同时出现在训练集和测试集里，退化成背图，准确率虚高。

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

gold 是选择题标准答案，不是教师模型分布，也不是行动成功率实测。所以这个模型学的是"按图选对答案的能力"，不直接等价于"选对动作的概率"。要做后者，需要把 gold 换成带实测打分的样本，本目录不覆盖。

Qwen3.5 系列 2026 年 2 月起改为统一多模态基座，早期的 Qwen-VL 产品线已停更，不要用 `Qwen2-VL` 或 `Qwen2.5-VL` 的占位符和 processor 写法套这里。
