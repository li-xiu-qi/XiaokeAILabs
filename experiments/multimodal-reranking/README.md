# 图文检索后的多模态重排

[English](README.en.md)

本目录保存实验代码、配置、运行方式与结果证据。问题、方法与结果见[图文重排序讲解](../../docs/zh/retrieval/reranking/多模态重排序.md)；以下命令从本目录运行。

[scripts](scripts/) · [evidence](evidence/) · [figures](figures/)

## 复算与复现

无需图片即可复算公开汇总：

```bash
python scripts/summarize.py
```

完整复现需在 Linux/CUDA 准备同版标注、图片以及两份模型。已验证环境：Python 3.12.3、Linux ARM64、NVIDIA GB10、PyTorch 2.14.0+cu130、torchvision 0.29.0+cu130、Transformers 5.16.1、sentence-transformers 6.0.1、qwen-vl-utils 0.0.14、BF16。CUDA 13.0 版本组合见 [PyTorch 官方版本表](https://pytorch.org/get-started/previous-versions/)；安装前仍需确认目标驱动、Python、CPU 架构和 GPU 支持。

```bash
python3.12 -m venv /path/to/venvs/retrieval
source /path/to/venvs/retrieval/bin/activate
python -m pip install pip==24.0
# CUDA 13.0 示例，其他设备按 PyTorch 官方指南调整
python -m pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
python -m pip check
python -c 'import torch, torchvision, qwen_vl_utils; print("torch:", torch.__version__, "torchvision:", torchvision.__version__)'
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA build:', torch.version.cuda, 'GPU:', torch.cuda.is_available()); assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(), '需要 CUDA GPU 及 BF16 支持'"
```

`qwen-vl-utils` 0.0.14 的图像处理代码直接依赖 torchvision，但其包元数据未声明；上面的导入检查不能以 `pip check` 替代。ARM64 上 PyTorch 2.14.0 安装的 `nvidia-cusparselt-cu13==0.8.1` 轮子内部平台标签有误，pip 24.2 及更新版本会让 `pip check` 报不支持当前平台；本次改用 pip 24.0 后，依赖检查、CUDA/BF16 检查和推理均通过。细节见 [PyTorch 问题记录](https://github.com/pytorch/pytorch/issues/202632)。[`requirements.txt`](requirements.txt) 固定了实验直接依赖；CUDA 相关的 torch 与 torchvision 仍按目标设备单独安装。`summarize.py` 与 `anonymize.py` 只使用 Python 标准库。输出目录应放在仓库外，候选及重排文件含原始描述和文件名，切勿直接公开。先按下方说明设置变量、准备数据和模型，再依次运行：

```bash
python scripts/candidates.py --annotations "$ANNOTATIONS_CSV" --images "$IMAGES_DIR" --model "$BGE_MODEL" --output "$PRIVATE_DIR/candidates.json"
python scripts/rerank.py --candidates "$PRIVATE_DIR/candidates.json" --images "$IMAGES_DIR" --model "$QWEN_MODEL" --output "$PRIVATE_DIR/reranked.jsonl"
python scripts/anonymize.py --candidates "$PRIVATE_DIR/candidates.json" --reranked "$PRIVATE_DIR/reranked.jsonl" --output "$PRIVATE_DIR/anonymous.json"
python scripts/summarize.py "$PRIVATE_DIR/anonymous.json"
```

可生成 5 张纯色图片和对应标注用于自检。先设置并创建仓库外的新 `PRIVATE_DIR`，然后执行：

```bash
export PRIVATE_DIR=/path/to/new-work-directory
mkdir -p "$PRIVATE_DIR/synthetic-images"
python - <<'PY'
import csv, os
from pathlib import Path
from PIL import Image
root = Path(os.environ['PRIVATE_DIR'])
with (root / 'synthetic-annotations.csv').open('w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['split', 'raw', 'filename'])
    writer.writeheader()
    for i, color in enumerate(['red', 'green', 'blue', 'yellow', 'purple']):
        name = f'{i}.png'
        Image.new('RGB', (96, 96), color).save(root / 'synthetic-images' / name)
        writer.writerow({'split': 'test', 'raw': repr([f'a solid {color} square']), 'filename': name})
PY
ANNOTATIONS_CSV="$PRIVATE_DIR/synthetic-annotations.csv"
IMAGES_DIR="$PRIVATE_DIR/synthetic-images"
```

先按下文获取模型并设置 `BGE_MODEL`、`QWEN_MODEL`，然后在一套新的输出文件上依次运行以下命令。标注中至少需要 5 条测试集记录；这是运行检查，不用于比较历史指标：

```bash
python scripts/candidates.py --annotations "$ANNOTATIONS_CSV" --images "$IMAGES_DIR" --model "$BGE_MODEL" --sample-size 5 --output "$PRIVATE_DIR/smoke-candidates.json"
python scripts/rerank.py --candidates "$PRIVATE_DIR/smoke-candidates.json" --images "$IMAGES_DIR" --model "$QWEN_MODEL" --output "$PRIVATE_DIR/smoke-reranked.jsonl"
python scripts/anonymize.py --candidates "$PRIVATE_DIR/smoke-candidates.json" --reranked "$PRIVATE_DIR/smoke-reranked.jsonl" --expected-count 5 --output "$PRIVATE_DIR/smoke-anonymous.json"
python scripts/summarize.py "$PRIVATE_DIR/smoke-anonymous.json" --expected-count 5
```

运行前将 `ANNOTATIONS_CSV`、`IMAGES_DIR`、`BGE_MODEL`、`QWEN_MODEL` 和 `PRIVATE_DIR` 分别设为标注 CSV、图片目录、两个本地模型目录及仓库外的新输出目录，创建该输出目录。标注及图片不随仓库分发；历史使用 Flickr30k Karpathy 测试划分，需自行依据 [Flickr30k 数据来源](https://huggingface.co/datasets/WhereIsAI/flickr30k-v2)获取图片与标注并核验许可。下方转换脚本从数据集导出图片和 CSV：默认抽取训练、验证、测试集各 512、128、100 行，按原数据顺序写出，并记录所选索引、种子和数据版本。先核对图片及标注许可。数据顺序与历史私有标注不一定相同，这条路径不能保证重建原样本。`candidates.py` 要求 UTF-8 CSV 含 `split`、`raw`、`filename` 列；至少 100 行 `split=test`，`raw` 为非空 Python 字符串列表且以首条描述作为查询，`filename` 相对于图片目录。格式示例：

```csv
split,raw,filename
test,"['a red car', 'a vehicle']",example.jpg
```

先准备数据，将 `/path/to/new-flickr-export` 改为不存在的新目录；如已知上游 commit，可加 `--revision <commit>` 固定，省略时记录为未固定。首次运行会下载数据并生成图像文件，需要磁盘空间：

```bash
python scripts/prepare_flickr30k.py --out /path/to/new-flickr-export
```

无需数据集或权重，可用 `python -m unittest discover -s tests -v` 检查抽样、CSV 和图片导出、覆盖保护，以及使用模拟模型执行重排、匿名化、汇总和断点续跑校验。测试只需 Pillow，模拟模型测试不验证真实模型或 GPU 推理。

然后将 `ANNOTATIONS_CSV` 和 `IMAGES_DIR` 分别指向导出目录下的 `annotations.csv` 和 `images/`。先通过以下方式取得模型；历史 Qwen revision 尚未核实；下方固定的 revision 的权重已与历史 SHA-256 一致，下载后仍应校验哈希。模型与图片占用空间需提前预留，自定义模型代码启用了远程代码加载，使用前检查来源和许可。

```bash
ANNOTATIONS_CSV=/path/to/annotations.csv
IMAGES_DIR=/path/to/images
PRIVATE_DIR=/path/to/new-work-directory
BGE_MODEL="$PRIVATE_DIR/bge"
QWEN_MODEL="$PRIVATE_DIR/qwen"
mkdir -p "$PRIVATE_DIR"
hf download BAAI/BGE-VL-base --revision cc4c733ed997dbee4ac70ccffb911e70c9c24b93 --local-dir "$BGE_MODEL"
hf download Qwen/Qwen3-VL-Reranker-2B --revision 4bd860ac4f15ad1897a214615cccc700f8f71818 --local-dir "$QWEN_MODEL"
sha256sum "$QWEN_MODEL/model.safetensors"
```

完成模型与数据准备后，可用一个全新环境跑 5 张合成图片的 GPU 全链路自检。该脚本会在 `XL_DATA_DIR/XL_RUN_ID/venv` 新建环境、安装本节固定版本、检查依赖与 CUDA/BF16，再运行候选生成、重排、匿名化和汇总；不会下载或覆盖模型权重：

```bash
export XL_DATA_DIR=/path/to/external-data
export XL_OUT_DIR=/path/to/new-external-output
export XL_RUN_ID=rerank-clean-001
export BGE_MODEL="$PRIVATE_DIR/bge"
export QWEN_MODEL="$PRIVATE_DIR/qwen"
bash scripts/run_clean_gpu_reproduction.sh
```

`XL_RUN_ID` 必须唯一，`XL_OUT_DIR` 放在仓库外；脚本拒绝复用已有环境和覆盖已有摘要。成功时写出不含模型路径的 `environment.json`、`metrics.json` 和 `summary.json`。此自检确认安装与模型推理链路，不替代真实数据集结果。

历史 Qwen `model.safetensors` 的 SHA-256 为 `466ec01961061e9d7f804b4fb1625fb6f406106cd1567e026096d4736fa9d5b9`。脚本中的 Qwen 推理封装来自 [Qwen 官方实现](https://github.com/QwenLM/Qwen3-VL-Embedding)，为本次 Transformers 5.16.1 环境添加多模态 token 类型转 tensor 的兼容检查。

重排支持从已有 JSONL 继续，恢复前会校验已完成记录的查询、候选和模型路径；旧格式记录缺少这些信息，不能安全续跑，请选择新的输出文件。模型路径不包含权重哈希，续跑时还须确保该路径下的模型文件未改变。图片或视频预处理失败会直接报错，修复输入后再重试。

公开重排脚本在图像输入中设置 `min_pixels=4096`、`max_pixels=65536`，并以 BF16 加载模型；这些预处理参数会影响复现结果，修改时应一并记录。

2026-11-13 在 Linux ARM64、NVIDIA GB10 CUDA GPU 和 BF16 上，使用上述两份固定 revision 的模型与 5 张自生成纯色图，实际跑通候选生成、重排、匿名化及汇总。运行使用分层依赖：Python 3.12、PyTorch 2.15.0+cu132、torchvision 0.29.1+cu130、Transformers 5.16.1、sentence-transformers 6.0.1、qwen-vl-utils 0.0.14、huggingface-hub 1.33.0、Pillow 13.0.0。该分层环境并非全新独立安装，未验证整体 `pip check`；这个合成样本只确认流程运行，不代表检索质量或历史结果。

2026-11-14 使用同一组版本，在复制并调整的独立环境安装本实验依赖后，`pip check` 通过；使用上述固定模型和 5 张合成图，公开脚本依次完成候选生成、重排、匿名化和汇总。该环境并非从零安装，合成样本也不能用于判断真实检索质量。

2026-11-15 按 [`scripts/run_clean_gpu_reproduction.sh`](scripts/run_clean_gpu_reproduction.sh) 从空 Python 3.12 venv 安装固定依赖，在 GB10 上通过 `pip check`、TorchVision/Qwen 工具导入、CUDA 和 BF16 检查，并使用设备上缓存的两份固定 revision 模型跑完 5 张合成图的候选生成、重排、匿名化和汇总。具体版本与摘要见 [`evidence/2026-11-15-clean-env-gpu-reproduction.json`](evidence/2026-11-15-clean-env-gpu-reproduction.json)。命中率前后均为 5/5，这是合成小样本的流程自检，不代表真实检索质量；本轮复用已有模型权重，没有验证从空设备缓存下载权重和数据集。

2026-11-14 在同一独立环境使用 Flickr30k 固定 revision `a2ac1f0bcee3132331324f1c435bcea4527c08f1` 导出的测试图片，从中抽取 20 条图文对，运行公开脚本完成候选生成、重排、匿名化和汇总。20 个查询的对应图片在重排前后均排在第 1 位；候选池只有 20 张图，结果接近上限，不能据此判断重排是否改善检索，也不能代表完整数据集或历史结果。

随后在相同数据版本与环境下，以公开脚本默认的 100 条测试图文对重新运行完整流程。100 张图片的候选池中，重排前后 Hit@1 分别为 97/100 和 99/100，3 条改善、1 条变差；差值的逐查询配对 bootstrap 95% 区间为 [-0.02, 0.06]。区间跨过零，且候选池较小、标注只确认对应图片，不能据此认定提升稳定或代表完整 Flickr30k；本次数据顺序也不保证重建历史私有样本。

历史实测使用私有等价脚本，公开汇总脚本已用匿名结果复算。依赖版本、图片预处理、抽样和模型文件差异均可能改变结果。标注来源和模型权重的核验条件见[实验讲解的数据说明](../../docs/zh/retrieval/reranking/多模态重排序.md#数据和实验范围)。
