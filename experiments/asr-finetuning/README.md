# Whisper 语音识别微调实验

[English](README.en.md)

## 一次 GPU 小样本运行

2026-11-15 在 NVIDIA GB10、Python 3.12、PyTorch 2.15.0+cu132、Transformers 5.16.1 和 Datasets 5.2 环境下，使用 Whisper small 和 LibriSpeech clean，完成了数据准备、基线评估、GPU bf16 微调和再次评估。模型版本为 `973afd24965f72e36ca33b3055d56a652f456b4d`，数据集版本为 `71cacbfb7e2354c4226d01e70d77d5fca3d04ba1`。训练 64 条、测试 16 条，batch size 2、1 epoch、32 个更新步，修正标签后重跑，训练平均 loss 为 0.4451。

修正标签后的同一测试集词错误率为 **3.79% → 3.79%**，本次没有观察到改善。修正前的实现错误屏蔽了结束符，当时得到 5.68%，不作为最终结果。测试样本很少，且 Whisper 预训练可能包含或接触过 LibriSpeech，不能视为全新盲测或推广为模型的普遍效果。评估使用 batch size 2，运行时出现 attention mask 未设置的提示，结果需结合这一限制理解。下方提供的是完整流程和参数，复现结果不保证数值相同。

同日还在独立环境使用 `run.sh all` 完成了一次端到端冒烟：8 条训练、4 条测试，4 次 GPU 更新，基线与微调后 WER 均为 0.87%。该次仅验证一键流程，样本太少，不用于判断微调效果。

训练运行使用独立虚拟环境；其最初创建方式没有留档。后来在 NVIDIA GB10（Linux aarch64）上另建全新虚拟环境，按下方命令安装并验证：Python 3.12.3、PyTorch 2.15.0+cu132、Transformers 5.16.1、Datasets 5.2.0、SoundFile 0.14.0、jiwer 4.0.0、NumPy 2.5.4、tqdm 4.70.1；`pip check` 无依赖冲突，导入正常，CUDA 可见 NVIDIA GB10。干净安装验证没有下载模型或数据，也没有运行训练。

用公开的 `openai/whisper-small`（或 `whisper-tiny`）与 `openslr/librispeech_asr` 的 clean 训练/测试分片，做一次可复现的英文 ASR 微调，并在同一测试集上比较微调前后的词错误率（WER）。全部脚本位于本目录，只读写本目录与通过环境变量指定的目录，不依赖仓库其他部分。


## 目录结构

```
asr-finetuning/
├── README.md / README.en.md   # 本文档与英文版
├── requirements.txt           # 固定版本的直接依赖（PyTorch 单独安装）
├── run.sh                     # 一键入口：prepare / train / eval / all
├── scripts/
│   ├── asr_common.py          # 运行契约、归一化、WER、manifest 与产物保护
│   ├── prepare_librispeech.py # 固定 revision 数据准备（16 kHz、时长过滤、集合隔离）
│   ├── train_whisper.py       # 微调训练循环（CUDA/CPU、bf16、每轮存档）
│   ├── eval_wer.py            # 测试集 WER 评估（基线或微调模型）
│   └── compare_wer.py         # 两次评估结果对比
└── tests/                     # 离线单元测试（不加载模型、不下载数据）
```

## 实验范围与架构

- 数据：LibriSpeech ASR clean 配置，训练用 `train.100`，测试用 `test`（可用参数改 split）。
- 输入处理：统一重采样为 16 kHz 单声道 PCM_16 WAV；文本经 `normalize_text` 归一化（NFKC、小写、去标点、折叠空白）；按时长过滤（默认 0.5–30 秒，匹配 Whisper 30 秒窗口）。
- 训练：`WhisperForConditionalGeneration` + 自定义 PyTorch 循环（AdamW、梯度裁剪、bf16 autocast、每轮保存检查点），不用 `transformers.Trainer`。
- 评估：贪心或集束解码生成转写，参考与假设同口径归一化后用 jiwer 计算 WER；基线模型与微调模型跑同一测试 manifest，`compare_wer.py` 输出差值。
- 隔离与保护：训练/测试 manifest 的样本 id 不允许重叠；训练前强制校验；输出目录必须是不存在的新目录或空目录，否则拒绝运行，避免覆盖已有产物。

## 硬件与环境

- 实测目标环境：NVIDIA GB10（Grace Blackwell，Linux aarch64），独立 CUDA 虚拟环境。单卡即可；whisper-small 小样本微调显存需求约数 GB，whisper-tiny 更低。
- 其他 Linux x86_64 + CUDA 亦可，未逐一验证。CPU 只能冒烟，速度不实用。
- 依赖：本次使用 Python 3.12.3。为复现上方 GB10 环境，先建立干净虚拟环境，再安装同版 PyTorch CUDA 13.2 wheel 和已锁定的直接依赖。PyTorch 官方索引提供 Linux aarch64 对应 wheel；其他 GPU/操作系统请按 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/) 选择匹配构建。PyTorch 单独安装，不放入本实验 requirements，避免覆盖平台构建。

```bash
python3 -m venv /path/to/venvs/asr
source /path/to/venvs/asr/bin/activate
# GB10 / Linux aarch64：复现本次记录的 PyTorch 版本
python -m pip install --index-url https://download.pytorch.org/whl/cu132 "torch==2.15.0+cu132"
python -m pip install -r requirements.txt
python -m pip check
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
python -c "import transformers, datasets, soundfile, jiwer; print('imports ok')"
```

- 从零安装核对：2026-11-15 在 NVIDIA GB10（Linux aarch64）上新建独立虚拟环境，使用本节 PyTorch 索引与 `requirements.txt` 完成依赖安装；`pip check` 通过，导入和 CUDA 检查通过。该验证未下载模型或数据，也未运行训练。
- Transformers 的 Whisper generate 参数（language/task 与 forced_decoder_ids）在 4.x 与 5.x 之间有差异；本清单固定到已验证版本，升级时应重新检查生成结果。

## 模型与数据来源

- 模型：`openai/whisper-small`（Hugging Face，权重许可证 Apache-2.0；OpenAI Whisper 代码 MIT）。默认要求通过 `ASR_MODEL_REVISION` 固定 commit；也可指向本地已下载目录。加载前请阅读模型卡确认许可。
- 数据：`openslr/librispeech_asr`（LibriSpeech 的 Hugging Face 数据集副本，clean 配置）。LibriSpeech 语料许可证为 CC BY 4.0，引用见 <https://www.openslr.org/12/>；数据集卡以 Hugging Face 页面为准。同样默认要求固定 `ASR_DATASET_REVISION`。
- 固定 revision 的查询方法（示例）：

```bash
python -c "from huggingface_hub import HfApi; print(HfApi().model_info('openai/whisper-small').sha)"
python -c "from huggingface_hub import HfApi; print(HfApi().dataset_info('openslr/librispeech_asr').sha)"
```

- 确认可复现性影响后，可用 `ASR_REQUIRE_REVISION=0` 或各脚本的 `--allow-floating-revision` 关闭强制固定；不建议。

## 运行命令

以下命令在任意目录执行均可（脚本按自身位置解析）。默认小样本：训练 200 条、测试 50 条、3 个 epoch。

### 方式一：一键入口

```bash
REPO=/path/to/XiaokeAILabs
EXP="$REPO/experiments/asr-finetuning"
WORK=/path/to/asr-work
mkdir -p "$WORK"

export XL_OUT_DIR="$WORK/out/run-001"
export XL_DATA_DIR="$WORK/data"
export XL_RUN_ID="run-001"
export ASR_MODEL="openai/whisper-small"
export ASR_MODEL_REVISION="<上面查到的模型 commit>"
export ASR_DATASET="openslr/librispeech_asr"
export ASR_DATASET_REVISION="<上面查到的数据集 commit>"

bash "$EXP/run.sh" all
```

要使用本页 GPU 运行的同一模型和数据版本与样本规模，在执行前设置：

```bash
export ASR_MODEL_REVISION=973afd24965f72e36ca33b3055d56a652f456b4d
export ASR_DATASET_REVISION=71cacbfb7e2354c4226d01e70d77d5fca3d04ba1
export ASR_MAX_TRAIN_SAMPLES=64 ASR_MAX_TEST_SAMPLES=16 ASR_SHUFFLE_BUFFER=64
export ASR_EPOCHS=1 ASR_BATCH_SIZE=2
```

### 方式二：分步执行（便于只重跑某一阶段）

```bash
# 1) 数据准备：streaming 下载所需样本，落 16 kHz WAV 与 manifest
python "$EXP/scripts/prepare_librispeech.py" \
  --dataset openslr/librispeech_asr --dataset-revision "<数据集 commit>" \
  --train-split train.100 --test-split test \
  --max-train-samples 200 --max-test-samples 50 \
  --output-dir "$XL_DATA_DIR/manifests/$XL_RUN_ID"

# 2) 训练（输出目录必须是不存在的新目录）
python "$EXP/scripts/train_whisper.py" \
  --train-manifest "$XL_DATA_DIR/manifests/$XL_RUN_ID/train.jsonl" \
  --test-manifest "$XL_DATA_DIR/manifests/$XL_RUN_ID/test.jsonl" \
  --model openai/whisper-small --model-revision "<模型 commit>" \
  --output-dir "$XL_OUT_DIR/model" --epochs 3 --batch-size 8

# 3) 基线 WER（微调前，同一测试集）
python "$EXP/scripts/eval_wer.py" \
  --model openai/whisper-small --model-revision "<模型 commit>" \
  --manifest "$XL_DATA_DIR/manifests/$XL_RUN_ID/test.jsonl" \
  --output "$XL_OUT_DIR/wer-baseline.json"

# 4) 微调后 WER（同一测试集）
python "$EXP/scripts/eval_wer.py" \
  --model "$XL_OUT_DIR/model/final-model" --model-revision "" --allow-floating-revision \
  --manifest "$XL_DATA_DIR/manifests/$XL_RUN_ID/test.jsonl" \
  --output "$XL_OUT_DIR/wer-finetuned.json"

# 5) 训练前后对比
python "$EXP/scripts/compare_wer.py" \
  --baseline "$XL_OUT_DIR/wer-baseline.json" \
  --finetuned "$XL_OUT_DIR/wer-finetuned.json" \
  --output "$XL_OUT_DIR/wer-comparison.json"
```

### 冒烟档（最小验证）

```bash
ASR_MODEL=openai/whisper-tiny ASR_MODEL_REVISION="$(python -c "from huggingface_hub import HfApi; print(HfApi().model_info('openai/whisper-tiny').sha)")" \
ASR_MAX_TRAIN_SAMPLES=8 ASR_MAX_TEST_SAMPLES=8 ASR_EPOCHS=1 ASR_BATCH_SIZE=2 \
bash "$EXP/run.sh" all
```

冒烟档会查询 tiny 模型的当前版本，仍需事先设置 `ASR_DATASET_REVISION`；模型版本与 small 不同，不能沿用 small 的 commit。该命令需要下载模型和少量数据，建议在目标 GPU 机器上执行。

## 环境变量

| 变量 | 作用 | 缺省 |
|---|---|---|
| `XL_OUT_DIR` | 输出根目录 | 实验目录下 `.work/out/$XL_RUN_ID` |
| `XL_DATA_DIR` | 数据目录（manifest、WAV） | 实验目录下 `.work/data` |
| `XL_RUN_ID` | 运行标识，进 manifest 目录与 metrics | `local` |
| `XL_REQUIRE_EXTERNAL_ENV` | `=1` 时禁止 `.work/` 回退，未注入即报错 | 未设置 |
| `ASR_MODEL` | 模型 id 或本地路径 | `openai/whisper-small` |
| `ASR_MODEL_REVISION` | 模型固定 commit | 无（默认强制） |
| `ASR_DATASET` | 数据集 id 或本地路径 | `openslr/librispeech_asr` |
| `ASR_DATASET_REVISION` | 数据集固定 commit | 无（默认强制） |
| `ASR_REQUIRE_REVISION` | `=0` 关闭强制固定 revision | `1` |
| `ASR_TRAIN_SPLIT` / `ASR_TEST_SPLIT` | 训练/测试 split | `train.100` / `test` |
| `ASR_MAX_TRAIN_SAMPLES` / `ASR_MAX_TEST_SAMPLES` | 样本上限，0 为全量 | `200` / `50` |
| `ASR_SHUFFLE_BUFFER` | 流式采样的打乱缓冲区大小 | `1000` |
| `ASR_EPOCHS` / `ASR_BATCH_SIZE` | 训练轮数 / 批大小；一键流程的评估也使用该批大小 | `3` / `8` |

受控环境（CI、算力机）建议设 `XL_REQUIRE_EXTERNAL_ENV=1` 并显式注入 `XL_OUT_DIR` / `XL_DATA_DIR` / `XL_RUN_ID`，防止误写在本地。

## 检查方法

```bash
# 离线单元测试（不需要 GPU、不下载模型与数据）
cd "$EXP" && python -m unittest discover -s tests -v

# 运行产物核对
python -c "import json; print(json.load(open('$XL_OUT_DIR/wer-baseline.json'))['wer_percent'])"
python -c "import json; print(json.load(open('$XL_OUT_DIR/wer-comparison.json')))"
```

关键产物：`prepare_summary.json`（数据与过滤参数）、`model/metrics.json`（loss 曲线，非 WER）、`wer-*.json`（WER 与最差样本）、`wer-comparison.json`（前后差值）。

## 已知限制与未验证项

- 直接依赖已锁定到 GB10 验证版本；其传递依赖未生成完整 lock 文件。其他平台的 PyTorch 构建需按各自硬件选择。干净环境验证只覆盖 Linux aarch64 / Python 3.12.3 / CUDA 13.2 这组组合。
- WER 归一化是简化版（不做数字读法、缩略语展开）；跨论文数字对比时口径可能不同。
- `prepare` 使用 streaming + 有限缓冲区 shuffle，样本上限大于缓冲区时随机性偏弱（脚本会警告）。
- 训练/测试隔离只按样本 id 检查，不识别同说话人跨集合泄漏；LibriSpeech 官方 split 本身按说话人划分，改用自定义 split 时需自行确认。
- 单卡单进程；未做混合精度 GradScaler（fp16 不支持，请用 bf16）、梯度累积、学习率调度与早停；`metrics.json` 的 loss 是训练集批次损失，不能当泛化指标。
- 未测试 Windows；`run.sh` 需要 Bash（Git Bash 可用）。
