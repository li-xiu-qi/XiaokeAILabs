# 用 ColSmol 检索文档页面：多向量匹配实测

[English](README.en.md)

历史视觉蒸馏脚本缺少独立开发集上的师生优势门槛，因此会在训练前停止；需要先补齐同条件师生评估和事先确定的优势标准。已完成的历史结果仍按原始运行条件解读。

本目录保存实验代码、数据准备、运行方式与结果证据。视觉检索的用途和方法见[视觉多向量检索](../../docs/zh/retrieval/colbert/视觉多向量检索.md)；微调与蒸馏的训练目标分别见[监督微调](../../docs/zh/retrieval/colbert/视觉多向量检索模型微调.md)和[蒸馏讲解](../../docs/zh/retrieval/colbert/视觉多向量检索模型蒸馏.md)。以下命令从本目录运行。

## 数据与基线

[ViDoRe DocVQA 测试子集](https://huggingface.co/datasets/vidore/docvqa_test_subsampled) 包含文档页图像及针对页面的问题，涵盖表格、报告和表单。按数据顺序选前 100 个不同的 `image_filename`，各取首次出现的问题；每个问题对同一组 100 页排序，以其配对页作为唯一已知正例。

| 指标 | 结果 |
|---|---:|
| Recall@1 | 0.55 |
| Recall@5 | 0.78 |
| nDCG@5 | 0.6664 |

数据可能存在其他相关但未标注的页面，候选范围仅为选出的 100 页，不代表全库召回。原模型直接推理，没有做本次微调。模型卡称其训练集与 ViDoRe 未使用相同的多页 PDF，但不能排除基础模型预训练阶段的数据重叠。

### 环境准备

原运行使用 Linux ARM64、NVIDIA GB10、BF16、PyTorch 2.14.0+cu130、Transformers 4.56.2、ColPali Engine 0.3.13、PEFT 0.17.1 和 Datasets 4.1.1。基线和训练脚本读取 Linux 的 `/proc/meminfo`，并固定在 CUDA 上以 BF16 加载模型，因此需要支持 BF16 的 NVIDIA GPU、匹配驱动与 CUDA PyTorch；当前脚本不提供 CPU 运行路径。脚本要求系统可用内存至少 50 GiB，运行期间跌破该值会退出；这是原设备的保护设定，并非模型的通用最低硬件要求。模型、数据集和运行输出需要额外磁盘空间。

原记录中的 PyTorch 2.14.0 与 [ColPali Engine 0.3.13 的发布元数据](https://pypi.org/project/colpali-engine/0.3.13/)所声明的 `torch>=2.2,<2.9` 不一致；直接安装下面的清单可能替换原 PyTorch，因此不能把它当作重建历史环境的步骤。历史组合为何可运行尚未查明。

下面给出一个符合包版本约束的候选组合：Linux x86_64、CUDA 12.8 wheel、PyTorch 2.8.0 和 torchvision 0.23.0，版本配对见 [PyTorch 官方表](https://pytorch.org/get-started/previous-versions/)。须确认设备架构、Python、驱动和 GPU 均支持相应 wheel；截至核对时 PyTorch 官方 CUDA 12.8 索引的该版本没有 Linux aarch64 wheel，不能用于历史 GB10；ARM64 需要另寻符合约束且适配设备的构建。该组合尚未在本实验从零安装或 GPU 验证，也不是历史环境。数据集、底模和 adapter 会从上游下载，不随本仓分发：

```bash
WORK=/path/to/independent-workdir
mkdir -p "$WORK"
python3 -m venv "$WORK/venv"
source "$WORK/venv/bin/activate"
# Linux x86_64、CUDA 12.8 候选组合；ARM64 请另寻符合约束且适配设备的构建
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
# 显式约束已选版本，避免解析器替换设备对应的 torch
python -m pip install -r requirements.txt 'torch==2.8.0' 'torchvision==0.23.0'
python -c "import torch; print('torch:', torch.__version__, 'CUDA:', torch.version.cuda, 'available:', torch.cuda.is_available()); print('BF16:', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)"
python -m pip check
```

确认 CUDA 和 BF16 均可用，再执行以下步骤。依赖检查通过不代表模型推理或训练已完成验证。`prepare_data.py` 默认使用数据集 revision `49bf8f13e13c41dd8cdb0cae5314e31c1da1e0d6`；首次下载需要网络。

```bash
hf download vidore/colSmol-256M --revision c79b633e17e060cc109b11aa2bf92a1517d3bd6f --local-dir "$WORK/model"
hf download vidore/ColSmolVLM-Instruct-256M-base --revision 99ca96f1f6b95b3a69e6abef74a2416cb738fed0 --local-dir "$WORK/base"
python scripts/prepare_data.py --output "$WORK/data"
python scripts/baseline.py --data "$WORK/data" --base "$WORK/base" --model "$WORK/model" --out "$WORK/smoke.json" --count 10
python scripts/baseline.py --data "$WORK/data" --base "$WORK/base" --model "$WORK/model" --out "$WORK/baseline100.json" --count 100
```

ColSmol 发布权重为 adapter，需先加载指定底模再加载 adapter；底模本身不是可直接使用的检索器。10 页运行只用于检查环境，100 页运行对应基线。模型卡提示 ColPali Engine 当前提示格式与模型训练时的格式存在差异；本次使用 0.3.13，不能直接与模型卡成绩比较。设备与软件差异也可能改变排名。

测试耗时 78.12 秒，包含编码和两两打分，不含模型与数据加载；PyTorch 峰值分配约 2.81 GiB，不等于整机内存占用。脚本保留本次机器使用的 50 GiB 可用内存保护下限，不是模型的通用最低要求。

[匿名逐查询排名](evidence/baseline100.json)记录 100 条查询的配对页名次，不包含原始图片、问题或数据集 ID。运行 `python scripts/recompute.py` 可复算汇总指标，但不能据此验证模型打分或标签。

## 合成报表监督微调

该实验生成 160 张虚构运营报表，每页三个字段，按实体划分为训练 96 页、验证 32 页、测试 32 页；训练前后在 96 条测试问题上分别取得 Recall@1 87/96 和 88/96。共享模板及单训练种子限制结论，详见[微调讲解](../../docs/zh/retrieval/colbert/视觉多向量检索模型微调.md)。

```bash
python scripts/generate_synthetic.py --output "$WORK/synthetic"
HF_HUB_OFFLINE=1 python scripts/train_synthetic.py --root "$WORK/synthetic" --base "$WORK/base" --adapter "$WORK/model" --out "$WORK/train-smoke" --smoke
HF_HUB_OFFLINE=1 python scripts/train_synthetic.py --root "$WORK/synthetic" --base "$WORK/base" --adapter "$WORK/model" --out "$WORK/train-formal" --steps 8
python scripts/recompute_training.py
```

训练使用上述基线模型和软件环境。页面生成脚本通过固定路径读取 `/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf` 和 `/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf`；请在执行前安装对应字体并检查这两个文件，字体版本可能改变页面像素。[synthetic-train-results.json](evidence/synthetic-train-results.json) 保存逐查询名次与损失，不含模型权重；复算只能检查记录一致性，不能仅凭名次核实模型输出。

2026-11-13 的一次通路检查使用 Linux ARM64、NVIDIA GB10、Python 3.12、PyTorch 2.11.0+cu130、torchvision 0.26.0+cu130 及上述固定版本的 Transformers、ColPali Engine 和 PEFT，加载本页列出的固定版本底模与 adapter，在合成页面上完成训练前评估、两步 CUDA 微调、训练后评估与 adapter 保存。额外检查显示两次优化步骤均改变了所有 420 个有梯度的参数张量。该检查只验证训练通路，不代表复现 100 页基线或正式微调结果。

此次检查在已有环境上叠加依赖，`pip check` 未通过：PyTorch 2.11 不满足 ColPali Engine 0.3.13 声明的 `torch<2.9`，还存在缺失 SciPy 和其他包版本冲突。它不构成可从零重建的兼容环境，也不证明该版本组合适用于其他任务；上面的候选安装组合仍待验证。

## 蒸馏实验记录

历史实验将多模态重排序教师分数用于训练，并与相同预算的监督组比较。教师选择时曾查看部分测试查询，结果不能视为完全未触碰测试集的验证；详情见[蒸馏讲解](../../docs/zh/retrieval/colbert/视觉多向量检索模型蒸馏.md)。当前历史训练入口在补齐独立开发集师生评估前停止，不应据此重跑正式蒸馏。

训练后 adapter 不随本仓分发；[独立页面复测](independent-evaluation/README.md)说明所需的三份 adapter、底模、旧数据及复算方法。`distill_synthetic.py` 在入口处直接停止，因此仅凭当前仓库无法生成蒸馏 adapter。

[qwen-train-scores.json](evidence/qwen-train-scores.json) 保存合成训练批次和教师分数；[synthetic-distill-results.json](evidence/synthetic-distill-results.json) 保存匿名逐查询名次。运行 `python scripts/recompute_distillation.py` 可检查历史指标与批次记录，不能还原模型推理。教师和学生使用独立环境，原教师使用 BF16、最大图片像素 65536；教师权重和训练权重不随仓库分发。

如需重新生成训练批次的教师分数，可在另一个 Python 环境安装相互匹配的 CUDA PyTorch 和 torchvision 后，使用 [requirements-teacher.txt](requirements-teacher.txt) 和下列命令。教师环境的 Transformers 版本参考同系列 Qwen 重排序运行，尚未在本实验中从零验证；它与上文学生环境的版本不同，不应混装。`scripts/qwen3_vl_reranker.py` 保留了 Qwen 官方实现的来源和修改说明；上游 Apache-2.0 许可原文见 [LICENSE-QWEN-APACHE-2.0](LICENSE-QWEN-APACHE-2.0)。此步骤只生成训练候选分数，并不实现教师优势门槛或解除蒸馏训练的停止条件。

```bash
python3 -m venv "$WORK/teacher-venv"
source "$WORK/teacher-venv/bin/activate"
# CUDA 13.0 示例：官方对应版本；其他设备按官方指南调整
python -m pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements-teacher.txt
python -m pip check
python -c 'import torch, torchvision, qwen_vl_utils; print("torch:", torch.__version__, "torchvision:", torchvision.__version__, "CUDA:", torch.cuda.is_available())'
hf download Qwen/Qwen3-VL-Reranker-2B --revision 4bd860ac4f15ad1897a214615cccc700f8f71818 --local-dir "$WORK/teacher-model"
python scripts/qwen_train_scores.py --root "$WORK/synthetic" --model "$WORK/teacher-model" --out "$WORK/new-teacher-scores.json"
```

上述 CUDA 13.0 示例来自 [PyTorch 官方对应版本表](https://pytorch.org/get-started/previous-versions/)，请确认本机驱动、Python 与 CPU 架构有适用 wheel；该安装组合尚未在本实验中从零验证。`qwen-vl-utils` 0.0.14 的图像处理代码需要 torchvision，但包元数据未声明，故不能只靠 `pip check`。

下载后应核对权重哈希。`--root` 需要前述生成的 `records.json` 和图片，分数输出包含查询及图片路径，应放在工作目录。

2026-11-14 使用上述固定教师模型，在 Linux ARM64、NVIDIA GB10 上生成 160 张合成页面，并对其中的训练划分完成 8 个批次的 CUDA BF16 教师评分；每批两条查询与两张候选图片，得到 32 个有限分数。独立教师环境使用 Python 3.12.3、PyTorch 2.15.0+cu132、torchvision 0.29.1+cu130、Transformers 5.16.1 和 qwen-vl-utils 0.0.14，`pip check` 与导入检查通过；该环境由已有环境复制调整，并非从零安装。这只验证教师训练批次打分，未建立开发集教师优势，也没有解除正式蒸馏的停止条件。

## 许可

模型及底模仓库元数据标记 MIT，底模卡正文说明骨干为 Apache-2.0、adapter 为 MIT；模型卡许可正文另有 ColQwen2 文案，重新分发权重前应核对具体文件和上游许可。ViDoRe 数据集卡标记 MIT，仍应遵守原始 DocVQA 及源文档的适用条件。本仓库不分发数据集页面、问题文本或模型权重。
