# 从表格图片恢复行列结构：SLANet+ 实测

[English](README.en.md)

本目录保存实验代码、配置与结果证据。实验目的、原理、数据、结果和边界见[完整实验文档](../../docs/zh/document-understanding/SLANet表格结构识别.md)。

[scripts](scripts/) · [fixtures](fixtures/) · [results](results/) · [figures](figures/)

## 复现

### 1. 结构预测

历史 CPU 运行使用 aarch64、2 个 PyTorch 线程、PyTorch 2.14.0+cu130、Transformers 5.17.0 和 Pillow（具体版本未记录）。CPU 预测不需要 GPU。环境记录不是完整的依赖锁定文件。

[Transformers 5.22.2 的包说明](https://pypi.org/project/transformers/5.22.2/)要求 Python 3.11 及以上，使用 PyTorch 后端时要求 PyTorch 2.6 及以上；这是上游依赖范围，不代表整套组合已通过验证。

在独立 Python 虚拟环境中，先按[PyTorch 安装指南](https://pytorch.org/get-started/locally/)选择适合操作系统和 CPU 或 CUDA 的匹配版本，一同安装 torch 和 torchvision，再安装其余依赖。脚本还需 Linux 上的 `/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf`，Ubuntu 可通过 `sudo apt install fonts-dejavu-core` 安装；其他系统需自行提供此字体路径或调整脚本。以下示例适用于 Linux：

```bash
python -m venv .venv
source .venv/bin/activate
# 按 PyTorch 安装指南安装匹配的 torch 和 torchvision 后：
python -m pip install -r requirements.txt
```

安装后可执行 `python -m pip check` 和 `python -c "import torch, torchvision, transformers; from PIL import Image; print(torch.__version__, torchvision.__version__, transformers.__version__)"`；这只检查依赖与导入，不会加载模型。

[requirements.txt](requirements.txt) 固定了本实验直接使用的 Transformers 5.22.2 和 Pillow 12.0.0，不含需按平台匹配安装的 torch 与 torchvision；后面的组装与鲁棒性脚本只用标准库，不需要安装该清单。传递依赖仍未锁定。脚本只读取本地模型，不会自动下载。可从 [SLANet+ 模型页面](https://huggingface.co/PaddlePaddle/SLANet_plus_safetensors)获取固定版本的完整快照，包括权重、配置和图像处理器文件：

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('PaddlePaddle/SLANet_plus_safetensors', revision='e44d12b3fe695792170dd46623075f63d8544461', local_dir='.work/slanet-model')"
```

2026-11-13 的一次模型快照下载尝试在推理开始前遇到读取超时与 HTTP 503，未取得完整快照，见[当次失败记录](evidence/2026-11-13-model-download-failure.json)。2026-11-15 的干净环境 GPU 复跑成功下载了固定 revision，并在推理前核对权重哈希；两次网络结果分别记录，不把早先失败描述为当前仍无法下载。

该版本 `model.safetensors` 的 SHA-256 为 `4455fa2a535b80738e0e0af2ca899f719efb8d6e1c7e7209cc5126f81607bf06`。以下命令从本实验目录执行，两个 `--out` 必须是尚不存在的目录：

```bash
python scripts/slanet_table_smoke.py --model .work/slanet-model --out .work/slanet-uncropped
python scripts/slanet_table_smoke.py --model .work/slanet-model --out .work/slanet-cropped --crop-table
```

有可用 CUDA 设备时，可在上述命令末尾加 `--device cuda`，并使用新的输出目录。输出目录包含输入图像和 `result.json`；检查其中 `device`、`model_sha256` 及 `result.structure`。例如：

```bash
python scripts/slanet_table_smoke.py --model .work/slanet-model --out .work/slanet-gpu --crop-table --device cuda
python -c "import json; d=json.load(open('.work/slanet-gpu/result.json')); s=d['result']['structure']; print('device:', d['device'], 'rows:', s.count('<tr>'), 'cells:', s.count('<td></td>'))"
```

2026-11-15 在 Linux aarch64、NVIDIA GB10 上，新建虚拟环境并安装 PyTorch 2.15.0+cu132、torchvision 0.29.1+cu132、Transformers 5.22.2 和 Pillow 12.0.0；`pip check`、固定模型快照下载及权重哈希核对均通过。对一张裁剪的合成表格完成 CUDA 推理，输出 4 行、12 个单元格，结构分数 0.99996，首次前向耗时约 0.430 秒。这只验证干净环境中的安装、下载和结构预测流程，不衡量真实表格准确率，也不复现完整评测。使用新外部目录重跑：

```bash
export XL_DATA_DIR=/path/to/slanet-data/run-001
export XL_OUT_DIR=/path/to/slanet-output/run-001
export XL_RUN_ID=run-001
export XL_REQUIRE_EXTERNAL_ENV=1
bash scripts/clean_env_cuda_smoke.sh
```

脚本在 `XL_DATA_DIR` 下建立虚拟环境、缓存和模型快照，在 `XL_OUT_DIR` 下保存结果；会拒绝覆盖已有虚拟环境或输出目录。详细结果见[干净环境 CUDA 冒烟证据](evidence/2026-11-15-clean-env-gpu-smoke.json)。原 CPU 运行使用 Transformers 5.17.0，属于另一份历史环境记录。

### 2. 文字与结构组装对齐与鲁棒性评测

使用 `fixtures/ocr-recognition.json` 中的文字多边形，通过几何空间聚类将文字回填至 SLANet 预测的 HTML 空槽位。该小样例是本仓 `paddle-ocr-detection-recognition` 实验的识别结果副本，不需要依赖另一实验目录。此部分只使用 Python 标准库，不要求安装 PyTorch 或下载 SLANet+ 模型。以下命令均从本实验目录运行：

```bash
# 执行完整空间网格组装
python scripts/assemble_table.py \
  --ocr-json fixtures/ocr-recognition.json \
  --structure-json results/cropped.json \
  --out-dir .work/assembly

# 运行鲁棒性压力测试与级联误差基准
python scripts/eval_table_reconstruction.py --output .work/assembly-robustness.json
```

2026-11-14 使用新建的独立 Python 3.12 环境运行上述两个标准库脚本：样例组装得到 4 行、12 个单元格；去掉一处文字和增加一行结构这两种扰动均被修正后的检查识别为未完全填充及结构不一致。结果只适用于仓库中的这张示例表格，不能代表真实表格的准确率。

历史结果保存在 `results/`。上述命令将新结果写到 Git 忽略的 `.work/`，输出文件已存在时会拒绝覆盖；重新运行请使用新的输出路径。 历史 `assembly_robustness_benchmark.json` 的 `all_slots_filled` 未检查剩余空槽，`structure_exact` 未检查结构预测，不能用于判断缺字或多余结构行；修正后新结果按这张无合并单元格样例的行数和空槽数检查，仍不是通用表格结构等价指标。
## 证据

- [2026-11-15 干净环境 CUDA 冒烟记录](evidence/2026-11-15-clean-env-gpu-smoke.json)：包含固定模型与输入哈希、软件版本、结构得分和单次前向耗时。
- [2026-11-13 GB10 合成表格 CUDA 冒烟记录](evidence/2026-11-13-gpu-synthetic-smoke.json)：较早的分层环境记录。
- [2026-11-13 模型下载失败记录](evidence/2026-11-13-model-download-failure.json)：一次获取固定快照的尝试因超时和 503 在推理前终止；后续成功推理使用了设备已有文件。
