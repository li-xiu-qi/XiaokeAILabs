# 重排序模型蒸馏：小模型能学到教师的排序偏好吗？

[English](README.en.md)

运行蒸馏前，需使用 `--min-teacher-gap` 预先指定教师在开发集 nDCG@10 上相对学生的最小绝对优势；还要求按查询配对 bootstrap 的 95% 区间下限大于零。未指定、门槛未通过或使用冒烟数据时不启动训练。历史结果按原始运行时的门槛解读。

本目录保存实验代码、配置、运行方式与结果证据。实验目的、原理和结果见[重排序蒸馏讲解](../../docs/zh/retrieval/reranking/重排序蒸馏.md)；以下命令从仓库外的工作目录运行。

[scripts](scripts/) · [assets](assets/) · [evidence](evidence/)

在固定 SciFact 候选上的一次实验中，Qwen3-Reranker-8B 教师测试 nDCG@10 为 0.7635；同一 TinyBERT 学生的监督组为 0.6336，蒸馏组为 0.6356。蒸馏与监督的差值区间包含零，不能据此确认蒸馏带来额外收益。[本轮逐查询证据](evidence/qwen3-8b/)可复算统计。开发集和测试集均曾用于旧实验，本次不属于全新盲测。

## 复现与历史记录

### 五组训练信号对照

在固定候选与 180 步训练预算下，分别使用原始标注、教师硬排序、教师软分数，以及原始标注与硬排序或软分数的混合信号。每组运行三个训练随机种子，训练样本抽样使用同一个固定种子；具体数据与限制见[结果讲解](../../docs/zh/retrieval/reranking/重排序蒸馏.md#教师硬排序和软分数有什么不同)。[逐查询、逐候选分数、训练损失和配置](evidence/hard-soft-five-arms/)均可供检查。测试集曾曝光，本次为探索性对照。

五组对照需要先完成下方准备和 Qwen 教师打分，再执行后面的五组运行命令。

需要 Linux、CUDA PyTorch、Transformers、NumPy 和 Hugging Face Hub。Qwen 8B 教师以 BF16 加载时，权重本身约占 16 GB，还需要额外内存容纳运行时和批次；实际需求随设备和配置变化。先在独立 Python 环境中安装适配设备的 CUDA PyTorch，再安装其余依赖：

```bash
REPO=/path/to/XiaokeAILabs
python3 -m venv /path/to/venvs/retrieval
source /path/to/venvs/retrieval/bin/activate
# 按 https://pytorch.org/get-started/locally/ 安装与系统、架构和 GPU 匹配的 PyTorch
python -m pip install -r "$REPO/experiments/reranker-distillation/requirements.txt"
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA build:', torch.version.cuda, 'GPU:', torch.cuda.is_available()); assert torch.cuda.is_available(), '训练需要可用的 CUDA GPU'"
python -c "import numpy, transformers, accelerate, huggingface_hub; print('依赖导入成功')"
hf --help
```

先把 `REPO` 改成仓库的绝对路径。[依赖清单](requirements.txt)将 Transformers 固定到 Qwen 教师的历史版本 4.56.2，并包含加载 Qwen 教师 `device_map` 所需的 Accelerate；其余依赖未锁定，因此并非完整锁文件，也尚未从零验证。`hf download` 使用 Hugging Face Hub 的命令行工具。先按 [重排序微调教程的数据准备步骤](../reranker-finetuning/README.md#动手训练准备数据检查流程并运行)准备 `scifact/` 和 `prepared/pools.json`，不必运行那里的训练。该候选文件的 SHA-256 应为 `e7cac48e7083bdd948e2773136c1df9672b4ca8f787ee5b774697b32d689c5f8`；不同版本或候选池不能直接对比本文数字。

将 `REPO`、`WORK` 和 `DATA_ROOT` 替换成仓库、仓外工作目录和已准备数据的绝对路径；模型和结果均保存在 `WORK`：

```bash
WORK=/path/to/private-reranker-work
DATA_ROOT=/path/to/prepared-reranker-data  # 包含 scifact/ 和 prepared/pools.json
mkdir -p "$WORK"
cd "$WORK"
hf download cross-encoder/ms-marco-MiniLM-L12-v2 --revision 7b0235231ca2674cb8ca8f022859a6eba2b1c968 --local-dir "$WORK/teacher"
hf download cross-encoder/ms-marco-TinyBERT-L2-v2 --revision 81d1926f67cb8eee2c2be17ca9f793c7c3bd20cc --local-dir "$WORK/student"
python "$REPO/experiments/reranker-distillation/scripts/run.py" \
  --data-root "$DATA_ROOT" --teacher "$WORK/teacher" \
  --student "$WORK/student" --out "$WORK/run" --min-teacher-gap 0.05
```

2026-11-14 使用上述固定版本的 MiniLM 教师、TinyBERT 学生和 SciFact 候选池，在 Linux ARM64、NVIDIA GB10 上执行了真实数据的教师评分与开发集门槛检查。89 条开发查询的 nDCG@10 分别为 0.6370 和 0.6138，教师优势为 0.0232，低于预设的 0.05；逐查询配对 bootstrap 95% 区间下限为 -0.0071，因此程序按设计停止，未训练或测试。模型权重哈希已核对；Python 3.12.3、PyTorch 2.15.0+cu132、Transformers 4.56.2，独立复制并调整的环境通过 `pip check`，并非从零安装。该结果不代替下方 Qwen 教师的复现。

要使用本轮的 Qwen 教师，可改用以下教师下载与运行方式，学生仍使用上面下载的 TinyBERT：

```bash
hf download Qwen/Qwen3-Reranker-8B --revision 77d193c791ed757ca307ee72715aa132723da912 --local-dir "$WORK/qwen-teacher"
python "$REPO/experiments/reranker-distillation/scripts/run.py" \
  --data-root "$DATA_ROOT" --teacher "$WORK/qwen-teacher" \
  --teacher-type qwen3 --student "$WORK/student" --out "$WORK/qwen-run" --min-teacher-gap 0.05
```

2026-11-14 在 Linux ARM64、NVIDIA GB10 上以固定 Qwen 教师和 TinyBERT 学生运行公开入口，模型权重已核对。89 条开发查询的教师 nDCG@10 为 0.7363，学生为 0.6138，优势 0.1225，超过预设的 0.05 且配对 bootstrap 下限为 0.0749；监督与蒸馏各 180 步后，在 300 条测试查询上完成最终评测：教师、原始学生、监督组、蒸馏组 nDCG@10 分别为 0.7618、0.6267、0.6336、0.6356；蒸馏减监督为 0.0019，逐查询配对 bootstrap 95% 区间为 [-0.0028, 0.0068]，不足以证明稳定提升。此数据集此前用于历史实验，测试查询与训练、开发查询不重合，但部分测试正例文档出现在训练候选中；只运行一个种子。独立复制环境为 Python 3.12.3、PyTorch 2.15.0+cu132、Transformers 4.56.2，通过 `pip check`；非从零安装。

Qwen 教师默认使用最大 512 token、批次 4，可用 `--qwen-max-length` 和 `--qwen-batch-size` 调整；学生最大 192 token。教师使用 `yes` 与 `no` 原始 logit 之差，按实际训练时抽取的候选打分。此次运行使用 DGX Spark、BF16 教师、FP32 学生、PyTorch 2.14.0+cu130 与 Transformers 4.56.2。新公开入口根据设备侧脚本整理，已完成 CPU 模拟、旧教师冒烟检查和下述合成数据 GPU 检查，现已用 Qwen 8B 完成上述真实数据端到端复跑；它会等两组训练结束后再计算测试结果，而本轮原始运行在训练前生成了 Qwen 测试分数，未将其用于教师选择或训练。配置、截断或执行顺序变化时，结果应重新报告。

默认 `--stage all` 依次进行教师打分和学生实验；也可先运行 `--stage teacher`，再以相同参数运行 `--stage student`，复用教师分数。脚本核对候选池、语料和查询文件的 SHA-256、模型路径及教师输出的完整性；模型目录内容仍需自行核验版本，单凭路径不能证明权重一致。现在重新运行时须事先指定教师在开发集上的最小绝对优势，并要求逐查询配对 bootstrap 的 95% 区间下限大于零，否则停止训练；示例的 0.05 是本例的选择，不是通用阈值。历史结果使用当时“教师高于学生”的门槛，不能声称通过了新门槛。门槛通过并完成训练后，输出包含逐查询指标、各模型逐候选分数、训练损失、参数清单和模型权重；未通过时只保留已完成的教师和开发集评估以及停止原因。设备、软件版本和非确定性计算会影响结果。

### 使用已有教师分数重跑五组对照

以下入口复用事先生成的教师打分，重新检查开发集教师优势后再训练。需要与本轮相同的训练与开发候选分数；可用前述 Qwen 运行的教师打分阶段生成，或使用同一配置产生的分数文件。教师分数文件的哈希会被记录，但不单凭哈希证明其模型来源。

```bash
python "$REPO/experiments/reranker-distillation/scripts/compare_signals.py" \
  --data-root "$DATA_ROOT" --student "$WORK/student" \
  --teacher-scores "$WORK/qwen-run/teacher-scores.json" \
  --out "$WORK/five-arm-run" --min-teacher-gap 0.05
```

五组对照入口要求 `--min-teacher-gap` 至少为 0.05；这个限制只适用于该入口。Linux 设备可追加 `--min-available-gib 50` 使可用内存低于 50 GiB 时停止。不要使用已有结果目录；复现实验的教师、学生版本与候选池须与本节其余说明一致。该公开入口在同一 DGX Spark 上复用已有教师分数完成了五组复跑，15 个测试指标均与原运行一致到小数点后九位；尚未验证从 Qwen 权重开始重新生成教师分数并完成端到端运行。

两阶段命令均可追加 `--device cpu --smoke` 检查打分流程：每个划分只取两个查询，需使用**独立输出目录**，不能覆盖正式结果。冒烟数据不足以判断教师优势，因此不会进入训练；这不是效果比较。正式运行应使用上面指定的模型版本与完整数据。

也可不下载模型或数据，使用随机初始化的微型 BERT 与合成数据检查 GPU：

```bash
python "$REPO/experiments/reranker-distillation/scripts/smoke_gpu.py" \
  --out "$WORK/synthetic-gpu-check"
```

输出目录必须为空。该检查先运行公开入口并确认冒烟数据在教师优势门槛处停止，再单独调用教师打分与学生训练函数，监督和蒸馏各执行两步并检查权重变化；这不是正式门槛通过，也不说明真实数据或 Qwen 教师已经复现。一次检查使用 Linux ARM64、NVIDIA GB10、Python 3.12.3、PyTorch 2.15.0+cu132、Transformers 4.56.2、NumPy 2.5.4。运行时使用独立依赖覆盖目录和已有 PyTorch 环境，覆盖目录的 Hugging Face Hub 与基础环境中的 `datasets` 存在版本冲突，不能把这次检查视为从零建立并验证完整依赖环境。

[旧 MiniLM 教师实验](evidence/)与[本轮 Qwen 教师实验](evidence/qwen3-8b/)分开保留逐查询指标。本轮可用 `python "$REPO/experiments/reranker-distillation/scripts/recompute-qwen3.py"` 复算蒸馏与监督的配对区间；这些记录不包含学生逐候选分数，无法从排名独立重算指标。BEIR 数据卡标注 CC BY-SA 4.0；原始 SciFact 标注与摘要语料有各自许可，参见[原始许可](https://github.com/allenai/scifact/blob/master/LICENSE.md)。公开证据不包含原始文献正文或模型权重。
