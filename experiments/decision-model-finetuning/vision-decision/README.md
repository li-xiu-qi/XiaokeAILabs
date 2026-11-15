# 训练自己的视觉版 Jev（多模态决策模型）

[English](README.en.md)

> **历史结果不能证明独立视觉答题能力**：旧数据将参考答案推理 `solution` 加入模型输入。现在的数据准备已移除此字段，并完成 100 步小规模训练及测试，但尚未完成原计划的正式训练；旧数据及旧权重不能用于新实验。

`agent-jev-text` 走的是纯文本路线。这个目录把同样的判别式决策模型搬到视觉上：底座换成本身就懂图像的多模态模型，每条候选同时吃图像、上下文和候选文本，由 score 头给出一个标量分，再在本问题的候选之间做归一。

模型侧没有现成的开源训练实现可抄（AgentJev 是纯文本，Jev-Omni 训练权重不开源），这里是自建路线，参考了 `pngwn/system-one-qwen3.5-4b-scorer` 的架构组织方式（LoRA 加 scalar score 头、按问题归一、温度校准），但数据、协议、训练脚本都独立写。

## 用到的东西

| 项 | 取值 | 说明 |
|---|---|---|
| 底座 | Qwen3.5-2B | Apache 2.0，原生多模态，权重约 4 GB |
| 数据集 | ScienceQA | 非 gated，图像内嵌在 parquet 里；许可信息见下文 |
| 参数高效微调 | LoRA r=16 | 本实验使用 LoRA；设备能否全参训练取决于模型、批次、精度等配置 |
| 损失 | 软交叉熵 + 0.1×sum-candidate Brier | 与文本版一致 |
| 校准 | 单一标量温度，81 点网格 | 只在 calibration 集上拟合 |

底座也可以用 `Qwen3.5-0.8B`（权重 1.7 GB）或 `4B`（权重 8.8 GB），改 `download_backbone.py --size` 和 `protocol.json` 即可。

## 目录结构

```
vision-decision/
├── scripts/
│   ├── download_backbone.py   # 下载 Qwen3.5 多模态底座
│   ├── download_data.py       # 下载 ScienceQA test split
│   ├── prepare_data.py        # parquet 转 jsonl，按图片 sha256 去重切分
│   ├── data_schema.py         # 数据和权重的输入版本检查
│   ├── train.py               # 训练、内存闸门、保存最终权重、校准、测试
│   ├── evaluate.py            # 独立评测（图像预 resize，避开 processor 单线程瓶颈）
│   ├── serve.py               # 起推理 HTTP 服务
│   └── protocol.json          # 实验协议与超参
├── data/               # 原始 parquet（下载生成）
├── models/             # 底座（下载生成）
├── prepared-clean-v1/  # 新版数据（切分生成）
└── runs/               # 训练产物（adapter、报告）
```

## 环境准备

历史训练流程依赖 Linux、CUDA 显卡与支持 bfloat16 的环境；训练脚本通过 `/proc/meminfo` 检查系统内存。所需显存和内存随模型、批次和精度变化，不能仅凭权重大小估算。

在本目录创建独立 Python 虚拟环境，并按设备安装匹配的 CUDA 版 PyTorch 和 torchvision。下面示例采用 [PyTorch 官方版本表](https://pytorch.org/get-started/previous-versions/)中的 2.14.0 / 0.29.0、CUDA 13.0；须先确认系统架构、驱动、Python 和 GPU 均有适用 wheel。这个新环境的完整安装尚未验证：

```bash
python3 -m venv .venv
source .venv/bin/activate
# CUDA 13.0 示例，其他平台请按官方版本表选择匹配组合
python -m pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt 'torch==2.14.0' 'torchvision==0.29.0'
python -m pip check
python -c "import torch, torchvision, pyarrow, peft; from transformers import AutoProcessor, Qwen3_5ForSequenceClassification; print('torch:', torch.__version__, 'torchvision:', torchvision.__version__, 'CUDA:', torch.cuda.is_available()); assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()"
```

`requirements.txt` 包含版本下限而非完整锁定版本；它声明了 PyTorch，若需特定 CUDA 构建，应先按设备选择并安装对应版本，再核对安装结果。上面的检查只验证导入和 GPU 条件，不能证明模型加载或训练成功。下载脚本需要网络和额外磁盘空间；模型及数据的许可见各自上游页面。ScienceQA 的[数据集卡片](https://huggingface.co/datasets/derek-thomas/ScienceQA)存在不一致：页面元数据标注 CC BY-SA 4.0，正文许可章节写 CC BY-NC-SA 4.0，正文也称仅限非商业研究；使用前请向上游确认适用条款。

2026-11-13，ScienceQA test parquet 的下载、默认配额切分与图片哈希跨划分检查已运行：test、train、dev、calibration 分别得到 100、400、100、100 条。随后以 Qwen3.5-0.8B 底座、每个划分的首条真实样本，在 CUDA 上完成 LoRA 单步训练、校准和评测；96 个 LoRA B 张量均发生变化，训练时 PyTorch CUDA 张量峰值为 4.86 GiB。底座 revision 为 `2fc06364715b967f1860aea9cf38778875588b17`。单步结果只说明链路能运行，不代表模型效果或完整复现。

本次复用的环境为 Python 3.12.3、PyTorch 2.15.0+cu132、torchvision 0.29.1+cu130、transformers 5.22.2、peft 0.21.3、accelerate 1.15.0、PyArrow 26.0.1、Pillow 12.0.0、huggingface_hub 2.4.0。它由已有环境和隔离安装目录组合而成，`pip check` 报告 `nvidia-cusparselt-cu13 0.8.1 is not supported on this platform`，不能视为从零安装验证。运行时缺少可选加速内核，回退到 PyTorch 实现；transformers 5.22.2 提示图像处理的 `max_pixels` 参数将在 5.23 移除，因此依赖暂限制在 5.23 以下。完整的干净数据训练和评测仍待进行。

2026-11-14 使用上述同一修正划分和固定 revision 的 Qwen3.5-0.8B，在另建的隔离环境完成 CUDA LoRA 100 步、批次 2 的训练，并用各 100 条的 calibration、dev、test 完成校准和评测。测试准确率为 71/100，校准温度为 1.75；测试交叉熵由 0.9741 降至 0.7772，准确率保持 71%。训练时 PyTorch CUDA 张量峰值为 30.06 GB。该环境从已有环境复制后按本实验清单调整，Python 3.12.3、PyTorch 2.15.0+cu132、torchvision 0.29.1+cu130、transformers 5.16.1、peft 0.21.3，`pip check` 通过；不是从零安装。

本次仅训练原计划 1200 步中的 100 步，没有未微调基线，不能判断微调带来的收益。四个划分均来自 ScienceQA 原始 test 文件，按图片哈希去重，不是官方划分，也未排除文本或近似图片重复；该单次 100 题结果不能作为官方测试成绩。按下文准备固定底座和修正数据后，可把训练命令中的 `--model` 改为 0.8B 路径、`--steps` 改为 `100`、`--bs` 设为 `2` 运行同一流程；测试集先前已用于探索，不能视为新的盲测。

在同一次训练的 100 条测试题上补做事后图片对照：独立评估脚本使用原图得到 71/100，把每张图替换为同尺寸白图后得到 56/100。两组使用相同问题、选项和 adapter；结果说明图片替换影响了本次总准确率，但未逐题统计，白图也是分布外输入，不能把差值直接解释为通用视觉能力。按下文的目录约定，在训练完成后可运行：

```bash
python scripts/make_image_ablation.py --data "$VISION_WORK/prepared-clean-v1" --out "$VISION_WORK/test-real"
python scripts/make_image_ablation.py --data "$VISION_WORK/prepared-clean-v1" --out "$VISION_WORK/test-white" --white
python scripts/evaluate.py --model "$VISION_WORK/models/Qwen3.5-0.8B" --ckpt "$VISION_WORK/run/final" --data "$VISION_WORK/test-real" --out "$VISION_WORK/eval-real" --temperature 1.75
python scripts/evaluate.py --model "$VISION_WORK/models/Qwen3.5-0.8B" --ckpt "$VISION_WORK/run/final" --data "$VISION_WORK/test-white" --out "$VISION_WORK/eval-white" --temperature 1.75
```

以上目录均应为新路径；这里直接使用本次校准温度，若重新训练需使用新运行报告的温度。新数据准备入口已核对生成文件哈希与本次一致。

## 流程

以下命令从 `vision-decision/` 目录执行。

### 1. 下载底座和数据

```bash
# 选取仓库外的全新工作目录；下载量较大
VISION_WORK="$(mktemp -d "${HOME}/vision-decision-XXXXXXXX")" || exit 1
export VISION_WORK
python scripts/download_backbone.py --size 2b --dest "$VISION_WORK/models"
python scripts/download_data.py --dest "$VISION_WORK/data"
```

两个下载器默认使用 Hugging Face 官方端点；如确有需要可显式加 `--mirror`，也会尊重已设置的 `HF_ENDPOINT`。数据和模型目标目录已存在时会拒绝覆盖；若中断，请检查并保留现有文件，另选新路径重新下载。数据集及模型 revision 未固定，需自行记录实际版本以便复查。

### 2. 切分数据

ScienceQA 的原始文件位于下载目录下的 `data/`，名称包含哈希。传入下载目录，脚本会递归查找 `test-*.parquet`，只有恰好一个匹配文件时才继续；没有或多个匹配会报错。也可直接传入已核对的具体 parquet 文件：

```bash
python scripts/prepare_data.py "$VISION_WORK/data" "$VISION_WORK/prepared-clean-v1"
```

产出 `$VISION_WORK/prepared-clean-v1/{train,dev,calibration,test}.jsonl` 和 `$VISION_WORK/prepared-clean-v1/split_manifest.json`。目标文件已存在时脚本会停止；要另存一次，可在输入文件后指定新的输出目录。

切分主键是图片 SHA-256，不是行 ID；同图不跨划分有助于降低图片泄漏，但不能消除题目、文字或语义重复。新版本仅使用题目、选项和提示构造输入，不使用 `lecture` 或 `solution`。默认配额为 test 100、train 400、dev 100、calibration 100，按该顺序及源文件行序选取互不重复的图像，没有随机打乱；这些划分均来自上游 test 文件，并非 ScienceQA 官方训练集与测试集，结果不能直接和官方划分的成绩比较。原始 test split 总共 4,241 行，无法满足旧版总计 4,800 个不同图像的配额。可通过 `--test-cases`、`--train-cases`、`--dev-cases`、`--calibration-cases` 调整；数据不足时脚本会提示缩小配额，不生成不完整的划分。实际行数和扫描时的重复、无效命中数以生成的 `prepared-clean-v1/split_manifest.json` 为准；同一源行可能在多个划分的扫描中重复计数。历史 `prepared/` 需保留作旧记录；训练和评测会拒绝缺少新版本标记的数据。

每行字段：

| 字段 | 含义 |
|---|---|
| `image_png_b64` | 图像 base64 |
| `state` | 备选答案和提示 |
| `context_hint`、`input_schema` | 输入提示与数据格式标记 |
| `question` | 判别问题 |
| `options` | 候选列表 |
| `gold` | 概率分布，正确项 1.0 其余 0 |

训练循环的轻量回归检查使用人造输入和微型替身模型，可确认单步更新参数，不涉及底座下载：

```bash
python tests/test_training_update.py
```

该测试需要支持 bfloat16 的 CUDA GPU；不满足时会显示跳过。它不能验证 Qwen、LoRA 或真实图文数据的训练。

如果要先验证真实模型和数据的训练链路，可以下载较小底座并只使用每个划分的第一条数据：

```bash
python scripts/download_backbone.py --size 0.8b --dest "$VISION_WORK/models" --revision 2fc06364715b967f1860aea9cf38778875588b17
python scripts/train.py \
  --model "$VISION_WORK/models/Qwen3.5-0.8B" \
  --data "$VISION_WORK/prepared-clean-v1" \
  --out "$VISION_WORK/smoke" --steps 1 --bs 1 --max-cases-per-split 1
```

该检查会保存一个单步 adapter，得到的指标没有统计意义。输出目录应为新路径；内存阈值可通过 `--mem-min-start`、`--mem-min-run` 和 `--mem-cap` 按设备调整。

### 3. 训练

```bash
python scripts/train.py \
  --model "$VISION_WORK/models/Qwen3.5-2B" \
  --data "$VISION_WORK/prepared-clean-v1" \
  --out "$VISION_WORK/run" \
  --steps 1200 \
  --bs 2
```

训练和独立评测要求输出目录不存在或为空；重复运行请为 `--out` 选择新路径，以免覆盖已有权重和结果。训练脚本保存最后一步权重，不会按开发集指标选择 checkpoint；训练参数固定后才应查看测试结果。

训练前会先读 `/proc/meminfo` 的 `MemAvailable`，低于 `--mem-min-start`（默认 25 GB）直接拒绝启动。训练中每步检查系统可用内存和本进程 PyTorch CUDA 张量的峰值分配量；可用量跌破 `--mem-min-run`（默认 45 GB）或张量峰值超过 `--mem-cap`（默认 70 GB）就停训、保存现场、写 `aborted.json` 然后退出，不进入评估。张量分配量不是整机的 GPU 内存总占用，检查也只在反向传播后发生，不能保证避免所有内存不足。

这个闸门不是可选项。Qwen3.5 是 Gated DeltaNet 加门控注意力的混合架构，激活开销模型和普通 transformer 不一样，`gradient_checkpointing_enable()` 默认的 reentrant 实现在这类非线性结构上可能静默失效。脚本里强制传了 `use_reentrant=False`，但如果模型或 batch 再放大，仍然可能把机器拖死，所以显式留一道闸。

### 4. 评测

训练脚本内置的评测会把同一张图交给 processor 处理多次（每个候选一次），
processor 的 smart_resize 是单线程 CPU 密集操作，1000 条测试集能跑十几分钟。
独立评测脚本先按长边和总像素双约束把图缩小，processor 只剩归一化可做。训练后的内置评测和独立评测均按每次一个问题处理，以控制批次大小；评测阶段仍可能因输入和设备条件耗尽内存：

```bash
python scripts/evaluate.py \
  --model "$VISION_WORK/models/Qwen3.5-2B" \
  --ckpt "$VISION_WORK/run/final" \
  --data "$VISION_WORK/prepared-clean-v1" \
  --out "$VISION_WORK/eval"
```

温度网格是 0.5 到 4.5 共 41 点，只在 calibration 集上拟合。train.py 内置的
网格上限是 2.0，实测最优温度会触顶，需要更宽范围。

### 5. 起服务

```bash
python scripts/serve.py --ckpt "$VISION_WORK/run/final" --model "$VISION_WORK/models/Qwen3.5-2B" --port 8010
```

```bash
curl -X POST http://127.0.0.1:8010/decide -H 'Content-Type: application/json' -d '{
  "image_png_b64": "<base64>",
  "state": "备选答案：\nA. ...\nB. ...\n\n提示：...",
  "question": "图中正在发生什么？请选出正确备选答案。",
  "options": ["A. ...", "B. ..."]
}'
```

返回 `probs`、`prediction`、`confidence`。

## 与 agent-jev-text 的差异

| 维度 | agent-jev-text | vision-decision |
|---|---|---|
| 底座 | Qwen3-0.6B | Qwen3.5-2B |
| 数据 | typed-decisions | ScienceQA |
| 微调方式 | 全参 | LoRA r=16 |
| 输入 | 上下文 + 候选文本 | 图像 + 上下文 + 候选文本 |
| 图像占位符 | 无 | `<|vision_start|><|image_pad|><|vision_end|>` |
| 切分主键 | case id | 图片 sha256 |
| 内存管理 | 无 | 启动体检 + 运行硬闸 |

## 已知边界

旧数据准备脚本曾把 ScienceQA 的 `solution`（参考答案推理）放进模型输入，历史训练和评估均受影响。修正数据已完成小规模训练和分开划分的测试，但原计划训练和独立视觉能力验证仍需完成。归档中的旧公开训练脚本缺少 `optimizer.step()`；历史运行实际使用的脚本快照未找到，无法确认历史权重是否发生更新。训练恢复、独立评测和服务入口会检查权重的数据格式标记，拒绝没有标记的旧权重。图片哈希去重及字段筛选也不能排除题目、提示或近似图片带来的其他泄漏。`gold` 是选择题标准答案，不能直接代表真实行动的成功概率。

Qwen3.5 系列 2026 年 2 月起改为统一多模态基座，早期的 Qwen-VL 产品线已停更，不要用 `Qwen2-VL` 或 `Qwen2.5-VL` 的占位符和 processor 写法套这里。
