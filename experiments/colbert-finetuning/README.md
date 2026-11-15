# ColBERT 微调：让科学文献检索的候选排序更准确
[English](README.en.md)

本目录保存实验代码、配置与结果证据。实验目的、原理、数据、结果及复现步骤见[完整实验文档](../../docs/zh/retrieval/colbert/文本多向量检索模型微调.md)；除非文档另有说明，请在本目录运行其中的命令。

[scripts](scripts/) · [config](config/) · [evidence](evidence/) · [assets](assets/)

## 准备与复现

历史运行在 DGX Spark 的 Linux ARM64、CUDA 上完成，训练脚本固定使用 CUDA；需要支持 CUDA 的 NVIDIA GPU、匹配的驱动与 PyTorch，以及容纳模型、数据和训练输出的磁盘空间。数据准备脚本使用 `hashlib.file_digest`，需 Python 3.11 或以上；已记录的 NumPy 和 SciPy 版本要求 Python 3.12 或以上。实验没有给出经测量的最低显存要求。实验使用 PyLate、sentence-transformers、Transformers、datasets 和 accelerate，完整安装记录见[环境清单](evidence/environment-freeze.txt)。PyLate 的完整依赖在 Linux ARM64 上存在安装限制；本实验只使用模型、训练与重排功能，环境需根据目标平台安装并验证。

先创建独立环境，再参照 [PyTorch 安装说明](https://pytorch.org/get-started/locally/)安装适配目标设备与 CUDA 驱动的 PyTorch：

```bash
python3 -m venv .venv
source .venv/bin/activate
# 在此安装匹配设备的 CUDA PyTorch
python -c "import torch; print('torch:', torch.__version__, 'CUDA:', torch.version.cuda, 'available:', torch.cuda.is_available())"
```

确认输出中 `available: True` 再运行训练；不要用与 CUDA 不匹配的通用 PyTorch 覆盖已有环境。

在已安装匹配 CUDA 的 PyTorch 环境中，先固定其版本，再按 PyLate 的包元数据安装完整依赖：

```bash
python -c "import importlib.metadata as m; print('torch==' + m.version('torch'))" > torch-constraint.txt
python -m pip install -c torch-constraint.txt -r requirements.txt
python -m pip check
python -c "import torch, numpy, datasets, sentence_transformers, pylate"
```

约束文件防止 pip 为解决依赖而替换已安装的 PyTorch。版本记录并非完整锁文件；PyLate 1.4.0 完整依赖的从零安装尚未验证。检查通过也不等于原实验已复现。

PyLate 1.4.0 将 `fast-plaid` 列为必需依赖，即使本实验不构建 PLAID 索引。其支持范围内的 PyPI 包没有 Linux ARM64 wheel 或源码包，且各版本约束 PyTorch；历史设备的 PyTorch 2.14.0+cu130 不满足这些版本约束。完整安装失败时，可针对固定候选训练和重排尝试**不完整依赖回退**：

```bash
python -m pip install -c torch-constraint.txt -r requirements-base.txt
python -m pip install --no-deps pylate==1.4.0
python -m pip check
python -c "import torch, numpy, datasets, sentence_transformers, pylate"
```

该回退省略 fast-plaid，`pip check` 会报告它缺失，也可能报告其他依赖冲突；需检查全部输出。这不是满足完整元数据依赖的环境，不能用于声称 PLAID 索引可用。PyLate 1.4.0 的依赖元数据将 fast-plaid 列为必需项；PyPI 上支持该版本范围的 Linux ARM64 发行包缺失。2026-11-15 在全新 Linux ARM64 环境验证了下面的训练和重排路径：安装 `requirements-base.txt` 后以 `--no-deps` 安装 PyLate 1.4.0，`pip check` 只报告 fast-plaid 缺失；导入模型、损失函数、训练工具与重排模块成功，合成数据上的两步 CUDA 训练及训练前后重排完成。该验证没有构建或查询 PLAID 索引，不代表原实验效果已复现。完整过程见[PyLate 1.4.0 ARM64 环境验证](evidence/pylate-1.4-arm64-training-smoke-2026-11-15.json)。

可在 Linux ARM64、Python 3.12、NVIDIA GPU 上用同一脚本重做这项合成数据检查。脚本会创建独立环境，并按已验证的 GB10 配置安装 PyTorch 2.15.0+cu132；其他设备可先按 PyTorch 官方安装说明确认版本，再通过 `XL_TORCH_SPEC` 和 `XL_TORCH_INDEX_URL` 替换安装项。设置三个输出变量后运行：

```bash
export XL_RUN_ID="$(python3 -c 'import uuid; print(uuid.uuid4().hex)')"
export XL_DATA_DIR="$PWD/.work/data"
export XL_OUT_DIR="$PWD/.work/out/$XL_RUN_ID"
# 其他硬件可覆盖默认的 torch==2.15.0+cu132 与 cu132 包源
# export XL_TORCH_SPEC='torch==<适配版本>'
# export XL_TORCH_INDEX_URL='https://download.pytorch.org/whl/<对应CUDA版本>'
bash scripts/verify_arm64_training_install.sh
```

脚本会创建独立虚拟环境、安装依赖、核对 PyLate 1.4.0 的训练与重排导入，并调用两步合成 GPU 冒烟。因 fast-plaid 未安装，`pip check` 会有且仅有这一项缺失提示；如果出现其他冲突，脚本会停止。不要把这条不完整依赖路径用于 PLAID 索引。

### Linux ARM64 的替代版本 GPU 冒烟

如果 PyLate 1.4.0 的完整依赖无法在目标平台安装，可以用 PyLate 1.2.0 检查训练通路。1.2.0 不依赖 fast-plaid，但它的损失函数实现与 1.4.0 不同，**不能用这个环境复现或比较原实验结论**。原版本的 ARM64 完整依赖问题仍未解决。

在独立的 Python 3.12 虚拟环境中，先按上面的步骤安装适合目标 GPU 的 CUDA PyTorch，再执行：

```bash
python -c "import importlib.metadata as m; print('torch==' + m.version('torch'))" > torch-constraint.txt
python -m pip install -c torch-constraint.txt 'pylate==1.2.0' 'datasets==4.1.1' 'accelerate==1.10.1' attrs pillow
python -m pip check
python scripts/synthetic_gpu_smoke.py
```

该脚本临时生成随机初始化的微型 BERT 和合成数据，调用本实验的训练入口，执行两步 CUDA 训练以及训练前后的候选重排；不下载原始模型或数据，输出不用于判断检索效果。2026-11-15 在 Linux ARM64、NVIDIA GB10 上从空 Python 3.12 虚拟环境安装 PyTorch 2.15.0+cu132、PyLate 1.2.0、Datasets 4.1.1、Accelerate 1.10.1、sentence-transformers 4.0.2 和 Transformers 4.48.2；`pip check` 通过，脚本完成两步 GPU 训练及重排。版本和验证范围见[全新环境 GPU 冒烟记录](evidence/clean-env-gpu-smoke-2026-11-15.json)。这验证了替代版本的安装与训练通路，不代表复现了原实验效果。PyLate 1.4.0 的独立环境也已完成同类合成检查，但需省略未提供 ARM64 发行包的 fast-plaid；详见上文。

### 原实验的数据与训练

将本实验目录复制到工作目录，在副本中下载数据，并准备训练查询、固定候选池和原始模型：

```bash
mkdir -p output
curl -fL https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip -o output/scifact.zip
curl -fL https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz -o output/scifact-original.tar.gz
sha256sum -c <<'EOF'
536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165  output/scifact.zip
11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be  output/scifact-original.tar.gz
EOF
unzip -q output/scifact.zip -d output
tar -xOzf output/scifact-original.tar.gz data/claims_train.jsonl > output/claims_train.jsonl
python scripts/prepare_scifact.py --data output/scifact --original-claims output/claims_train.jsonl \
  --config config/data.json --out output/prepared
huggingface-cli download lightonai/colbertv2.0 --revision 0b9b1e380096a3b72b6b4877f68d59effb62e0f2 --local-dir output/model
sha256sum output/prepared/pools.json evidence/training-pairs.jsonl output/model/model.safetensors \
  output/model/1_Dense/model.safetensors output/model/1_Dense/config.json
```

将输出的哈希与[输入校验](evidence/input-sha256.txt)对照。训练对只包含查询和文档 ID，不包含原文。模型必须完整下载，包括 `1_Dense/` 投影层。脚本以离线模式读取模型：

```bash
python scripts/run.py --data output/scifact --prepared output/prepared \
  --pairs evidence/training-pairs.jsonl --model output/model --out output/smoke --smoke
python scripts/run.py --data output/scifact --prepared output/prepared \
  --pairs evidence/training-pairs.jsonl --model output/model --out output/formal
python scripts/analyze.py --results output/formal --pairs evidence/training-pairs.jsonl
```
模型、数据和训练输出不放入版本库。公开的路径参数化脚本已在原设备上完成两步冒烟及完整重跑，重跑得到相同的前后汇总指标；复现时仍需核对环境和输入文件哈希。固定候选池和训练对的哈希见[输入校验](evidence/input-sha256.txt)。训练时若更换采样、模型或参数，结果不能直接视作本次实验。

### 复算已保存指标

不下载模型也能复算已保存的逐查询指标与 bootstrap 区间。下面使用临时目录，避免覆盖仓库内证据：

```bash
tmp_dir=$(mktemp -d)
cp evidence/{before,after}-per-query.json "$tmp_dir"/
python scripts/analyze.py --results "$tmp_dir" --pairs evidence/training-pairs.jsonl
```

该脚本不重新运行模型或复算候选分数；检验排序本身需使用记录中的候选 ID、相关 ID 和排序结果。

### 使用训练后的模型重排

以下示例用正式训练保存的模型，演示对两篇候选排序；示例文本只说明调用方式。

```python
from pylate import models, rank

model = models.ColBERT(
    model_name_or_path="output/formal/fine-tuned",
    device="cuda", query_length=32, document_length=180,
)
query = ["Does the treatment affect the outcome?"]
documents = [["A study of the treatment and outcome.", "An unrelated study."]]
query_vectors = model.encode(query, is_query=True, convert_to_numpy=True)
document_vectors = model.encode(documents[0], is_query=False, convert_to_numpy=True)
results = rank.rerank(
    documents_ids=[["doc-1", "doc-2"]],
    queries_embeddings=query_vectors,
    documents_embeddings=[document_vectors], device="cuda",
)
print(results[0])
```

首次训练后保存权重时，模型卡联网请求未完成；权重已落盘，随后以离线方式重新加载权重完成评测。公开脚本加入离线模式后，已完整重跑训练、保存和评测。代码与数据划分只支持本次英文科学文献的判断，未测试中文、全文档检索、端到端索引构建或线上延迟。


### 残差量化压缩与剪枝检索基准测试

若需评估 ColBERT 多向量在 ColBERTv2 残差量化与 PLAID 质心剪枝下的压缩比与延迟加速比，可运行独立基准评测脚本：

```bash
python scripts/benchmark_residual_compression.py
```

该脚本使用合成向量；倒排索引的质心分配是随机生成的，与用于精排的文档向量并未对应，暴力检索计时由 100 篇样本外推，剪枝路径最多计算 100 个候选，因此这些结果不能用于证明实际召回率或加速比。旧版输出中的 0.982 召回率是写死的示例值，并非实测；现已移除固定值并标注为未测。脚本拒绝覆盖已有的 `evidence/results_residual_compression.json`，请在副本中运行并保留原证据。相关原理见[多向量存储压缩与 PLAID 剪枝检索](../../docs/zh/retrieval/colbert/多向量索引与剪枝加速.md)。
