# 图文检索蒸馏：Jina v5 教师与 BGE-VL 学生

[English](README.en.md)

历史实验的教师选择没有事先设定开发集优势门槛。当前脚本增加了独立验证集上的教师检查：教师必须在同一图文候选池上的文本检索图片 R@1 至少领先指定的绝对差值，且逐查询配对 bootstrap 95% 区间的下限大于零，才会训练学生。新流程已在真实图文数据上运行一次，历史结果仍按原始运行条件解读。

本目录保存实验代码、配置、运行方式与结果证据。实验目的、原理和结果见[图文蒸馏讲解](../../docs/zh/retrieval/embedding/多模态Embedding蒸馏.md)；以下命令从本目录运行。

[scripts](scripts/) · [evidence](evidence/) · [assets](assets/)

## 复现与历史记录

**复现状态：**原始设备脚本跑完了[讲解中的历史实验](../../docs/zh/retrieval/embedding/多模态Embedding蒸馏.md)。公开脚本去除了设备私有路径和教师封装，已在复制并调整的独立环境中完成一次真实数据运行。它需要 Linux CUDA 和约 50 GiB 可用系统内存。图文教师代码依赖模型自定义代码，应自行核查后运行。原环境使用 Python 3.12、PyTorch 2.14.0+cu130、Transformers 5.16.1、Sentence Transformers 6.0.1、NumPy 和 Pillow。环境版本、标注顺序和硬件变化都可能影响结果。

先在独立的 Python 环境中安装适配设备的 CUDA PyTorch，再安装其余脚本和模型下载所需的包：

```bash
python3 -m venv /path/to/venvs/retrieval
source /path/to/venvs/retrieval/bin/activate
# 按 https://pytorch.org/get-started/locally/ 安装适配系统架构和 GPU 的 CUDA PyTorch
python -m pip install -r requirements.txt
python -m pip check
python -c "import peft; from peft import PeftMixedModel, PeftConfig; print('PEFT:', peft.__version__)"
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA build:', torch.version.cuda, 'GPU:', torch.cuda.is_available()); assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(), '需要 CUDA GPU 及 BF16 支持'"
```

[requirements.txt](requirements.txt) 固定的版本来自原环境记录，其余包未锁定；固定 revision 的 Jina 教师模型代码直接导入 PEFT，因此清单包含未锁定版本的 `peft`。该安装组合尚未在全新环境中验证。`verify.py` 仅使用 Python 标准库。

2026-11-14 在 Linux ARM64 的独立复制环境按清单补装依赖后，`pip check` 通过；Python 3.12.3、PyTorch 2.15.0+cu132、Transformers 5.16.1、Sentence Transformers 6.0.1、PEFT 0.21.3、datasets 5.2.0 可导入，CUDA 和 BF16 可用。两项数据导出单元测试与历史指标复算通过；这不是从零安装；该环境检查本身没有加载真实图文模型或证明完整复现。

原始图片及标注不随仓库分发，历史实验使用 Flickr30k；可从 [Flickr30k 数据集页面](https://huggingface.co/datasets/WhereIsAI/flickr30k-v2)查找数据并核验图片及标注许可，再自行转换为脚本格式。可用下方脚本将数据集导出为所需 CSV 和图片；原始数据顺序未随仓库提供，因此无法保证重建历史样本。标注 CSV 需要 `filename`、`split`、`raw` 三列；`split` 为 `train`、`val`、`test`，`raw` 是五条描述组成的 Python 列表，格式示例见[数据结构](../../docs/zh/retrieval/embedding/多模态Embedding蒸馏.md#数据结构)。三部分至少分别提供 512、128、100 条；图片文件名相对于图片目录，标注行顺序也会影响抽样。

2026-11-14 从 Flickr30k 数据集 revision `a2ac1f0bcee3132331324f1c435bcea4527c08f1` 导出并准备了 740 张图片：训练、验证、测试分别为 512、128、100 张；图片解码检查通过，选中图片未发现字节完全一致的重复。该检查不排除视觉近重复或来源重叠。

同日用固定版本 Jina v5 教师与 BGE-VL 学生完成真实数据流程。128 条验证查询上，教师和初始学生的 R@1 分别为 0.9844 和 0.9141，差值 0.0703，配对 bootstrap 95% 区间 [0.0156, 0.1250]，达到预设门槛；监督和蒸馏组各训练两轮、64 个批次。在 100 张图的测试候选池中，基线 R@1 为 0.96，监督、蒸馏和教师均为 0.99；两个学生在这 100 条查询的 R@1 结果一致。候选池小且结果接近满分，不能据此认定蒸馏有优势或两种方法普遍等价；运行仅使用一个训练种子，尚未排除图片近重复和预训练数据重叠。

例如，可指定不存在的新目录导出所需数量的样本；`--revision <commit>` 可固定已知的数据集版本，省略则上游版本未固定。首次运行会下载数据并生成图片，需预留磁盘空间。

```bash
python scripts/prepare_flickr30k.py --out /path/to/new-flickr-export
```

无需数据集或权重，可用 `python -m unittest discover -s tests -v` 检查抽样、CSV 和图片导出，以及拒绝覆盖已有输出；测试只需 Pillow。

也可以先在 CUDA GPU 上检查学生训练链路，指定仓库外不存在的新目录：

```bash
python scripts/synthetic_gpu_smoke.py --out /path/to/new-synthetic-smoke
```

该命令现场生成随机初始化的小型 CLIP、合成图片和描述，以及人为构造的教师向量；随后调用 `train.py`，检查教师门槛、监督与蒸馏训练、验证、测试和完成标记。它不下载数据或权重，不运行真实教师，指标也不能用于判断模型效果或复现历史结果。输出目录不可复用。

2026-11-13 在 NVIDIA GB10、Linux、Python 3.12、PyTorch 2.15.0+cu132、Transformers 5.22.2、tokenizers 0.23.3、Pillow 12.0.0、NumPy 2.5.4 上跑通上述合成检查：训练、验证、测试分别为 20、16、8 条，监督与蒸馏各完成 4 个批次。此运行复用了已有环境及单独安装的 Pillow，**不是**对上面固定版本的全新安装验证，也没有运行真实 Jina 教师、BGE-VL 或 Flickr30k。

导出目录含 `annotations.csv`、`images/` 和记录来源、抽样种子及行号的 `source.json`。设 `DATA` 指向导出的 CSV，`IMAGES` 指向图片目录，`WORK` 指向仓库以外的全新工作目录。正式评估前固定师生模型、数据和教师优势阈值；这里的 `0.05` 指 R@1 高出至少 5 个百分点，是本方案的门槛，不是对所有数据集通用的标准。

```bash
DATA=/path/to/annotations.csv
IMAGES=/path/to/images
WORK=/path/to/new-work-directory
python -c "from huggingface_hub import snapshot_download; snapshot_download('jinaai/jina-embeddings-v5-omni-small', revision='5c54692f22e186fc12ea38a9193e3ff9e1cda2a3', local_dir='/path/to/teacher')"
python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/BGE-VL-base', revision='cc4c733ed997dbee4ac70ccffb911e70c9c24b93', local_dir='/path/to/student')"
python scripts/prepare.py --annotations "$DATA" --images "$IMAGES" --out "$WORK"
python scripts/teacher.py --workdir "$WORK" --images "$IMAGES" --teacher /path/to/teacher --student /path/to/student --min-teacher-gap 0.05 --stage development
python scripts/train.py --workdir "$WORK" --images "$IMAGES" --student /path/to/student --min-teacher-gap 0.05
python scripts/teacher.py --workdir "$WORK" --images "$IMAGES" --teacher /path/to/teacher --stage test
python scripts/verify.py
```

数据准备按图片划分抽取 512 个训练样本、128 个验证样本及 100 个测试样本，并检查图片可解码和是否字节重复。教师编码前会将师生模型路径、数据哈希和阈值写入 `teacher-plan.json`；它先仅为训练集与验证集编码。训练脚本核对计划及教师向量哈希，比较教师和初始学生在同一个验证候选池的排序，保存 `teacher-gate.json`。模型路径本身不能证明权重版本未变化，仍须自行核对模型版本。若门槛不通过，训练和测试均停止；通过后按固定预算完成监督与蒸馏两组，最后评估测试集。教师测试集评分需要两组训练完成的标记。训练产物、图文描述与模型不要上传到公开仓库。

验证集没有参与这次脚本的梯度更新，但曾在历史实验中用于评估；如果后续要反复调整模型或阈值，应另外准备新的独立开发集及最终测试集。两组只用一个训练种子，配对 bootstrap 只反映固定查询集合的抽样不确定性，不反映重新训练或数据集选择的波动。

`verify.py` 使用[每查询学生排名](evidence/student-test-ranks.json)独立复算三组学生指标和配对区间，不含图片、描述或身份；教师只保留聚合指标，不能从这份排名文件复算。复算检查的是统计计算，不能替代重新运行模型。图片近重复、预训练数据重叠和同图其他有效描述仍未排除。
