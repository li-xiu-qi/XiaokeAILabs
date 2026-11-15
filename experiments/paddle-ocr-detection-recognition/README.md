# 用 PP-OCRv5 定位并识别表格文字

[English](README.en.md)

本目录保存实验代码、配置与结果证据。实验目的、原理、数据、结果和边界见[完整实验文档](../../docs/zh/document-understanding/PaddleOCR文字检测与识别.md)。

[scripts](scripts/) · [results](results/) · [figures](figures/)

## 复现

实验先检测文字区域，再逐框识别；它不预测表格行列结构。CPU 运行不需要 GPU。原运行记录为 PyTorch 2.14.0+cu130、Transformers 5.17.0、OpenCV 5.0.0 和 Pillow；这不是完整的依赖锁定文件。

### 环境准备

[Transformers 5.17.0 的包说明](https://pypi.org/project/transformers/5.17.0/)要求 Python 3.10 及以上，使用 PyTorch 后端时要求 PyTorch 2.5 及以上；这是上游依赖范围，不代表整套组合已通过验证。

在独立 Python 虚拟环境中，先按[PyTorch 安装指南](https://pytorch.org/get-started/locally/)选择适合操作系统和 CPU 或 CUDA 的匹配版本，一同安装 torch 和 torchvision，再安装其余依赖。例如 Linux / macOS：

```bash
python -m venv .venv
source .venv/bin/activate
# 按 PyTorch 安装指南安装匹配的 torch 和 torchvision 后：
python -m pip install -r requirements.txt
```

安装后可执行 `python -m pip check` 和 `python -c "import torch, torchvision, transformers, cv2; from PIL import Image; print(torch.__version__, torchvision.__version__, transformers.__version__)"`，只检查依赖和导入，不会加载模型。

[requirements.txt](requirements.txt) 固定 Transformers、Pillow 和 OpenCV 的发行包版本，不包含需按平台匹配安装的 torch 与 torchvision，也不是完整锁文件。OpenCV Python 发行包固定为 `5.0.0.93`；导入后的 `cv2.__version__` 显示 `5.0.0`，两者版本号口径不同（[PyPI 发布页](https://pypi.org/project/opencv-python/5.0.0.93/)）。2026-11-15 首次整理时误把 `cv2` 运行时版本当成发行包版本，后续已修正并完成空 venv 安装与 GPU 推理。此次空 venv 使用了设备已有的 pip wheel 缓存和已校验模型文件；不是清空设备缓存后的全新下载验证。

在 Linux ARM64 / GB10 上从空环境安装并运行合成表格 GPU 样例，可用以下脚本。它会创建独立虚拟环境、下载固定模型版本、检查 `pip check` 和 CUDA，并要求 12 个识别文本与样例预期完全一致。数据、模型和结果放在 `.work/` 下；每次使用新的运行标识，避免覆盖：

```bash
export XL_DATA_DIR="$PWD/.work/data"
export XL_OUT_DIR="$PWD/.work/run-$(date +%Y%m%d-%H%M%S)"
export XL_RUN_ID="$(date +%Y%m%d-%H%M%S)"
bash scripts/run_clean_gpu_reproduction.sh
```

脚本固定使用 PyTorch `2.15.0+cu132` 和 torchvision `0.29.1+cu130`，适用于记录中的 GB10 Linux ARM64 组合；其他系统或 GPU 应按设备安装匹配的构建，再按下方通用命令运行。成功时输出 `ppocr_metrics.json` 和含环境、模型及输入哈希的 `summary.json`。如果设备已有模型文件，可设置 `PP_OCR_MODEL_SOURCE_DIR` 指向同时含 `det/`、`rec/` 两个子目录的目录；脚本会复制配置和权重，并校验权重哈希。未设置时脚本从 Hugging Face 下载固定版本。

### 运行

脚本只从本地读取模型，不会自动下载。以下命令从本实验目录执行，将[检测模型](https://huggingface.co/PaddlePaddle/PP-OCRv5_server_det_safetensors)和[识别模型](https://huggingface.co/PaddlePaddle/PP-OCRv5_server_rec_safetensors)的固定版本下载到 `models/`；下载工具随 Transformers 的依赖安装：

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download('PaddlePaddle/PP-OCRv5_server_det_safetensors', revision='cbea9f3c3254c6ff7b0016cfbf90549e1ad4c5bb', local_dir='models/det')
snapshot_download('PaddlePaddle/PP-OCRv5_server_rec_safetensors', revision='542979d7cc3791732bb12af35313a6840952d79f', local_dir='models/rec')
PY
```

运行检测和识别：

```bash
python scripts/run.py \
  --image figures/synthetic-table.png \
  --det-model models/det --rec-model models/rec \
  --det-device cpu --rec-device cpu --threads 2 \
  --output /tmp/paddle-ocr-result.json
```

输出 JSON 文件须不存在。使用 CUDA 时，将两个设备参数改为 `--det-device cuda --rec-device cuda`；需要安装支持所用 GPU 的 PyTorch 和 torchvision。检查输出的识别结果：

```bash
python -c "import json; x=json.load(open('/tmp/paddle-ocr-result.json', encoding='utf-8')); print(len(x), [r['prediction']['text'] for r in x])"
```

2026-11-13 在 Linux aarch64、NVIDIA GB10 上使用固定模型版本和 CUDA 运行这张合成图，得到 12 个文本框，识别文字与 `results/cpu-reproduction.json` 一致。运行环境为 Python 3.12、PyTorch 2.15.0+cu132、torchvision 0.29.1+cu130、Transformers 5.17.0、OpenCV 5.0.0 和 Pillow 13.0.0；测试使用分层依赖环境，尚未验证全新环境的完整安装。模型的配置和权重分别与上述固定版本的文件哈希一致，但原运行没有记录 revision 和权重哈希，无法确认历史模型快照完全相同。单张合成图的结果不能代表真实文档准确率。

2026-11-14 在独立复制环境中按清单安装依赖并通过 `pip check`，使用上述版本组合，在 CUDA 上处理了一份[固定版本的上游 PDF](https://github.com/RapidAI/RapidOCRPDF/blob/bb99e81e397c53fc9b6bba71d49408470c02f09e/tests/test_files/direct_and_image.pdf) 首页顶部截图。检测到 3 个文字区域，识别结果包含论文标题；未逐字标注或测量准确率。此检查使用 PyMuPDF 2.1 将首页顶部 200 PDF 点以 2 倍分辨率渲染为 PNG，再将该图片路径传给上面的 `--image`；环境由已有依赖复制，不是从零安装验证。单张截图也不能代表真实文档的整体效果。

2026-11-15 的首轮干净环境尝试先后被失效代理和 OpenCV 发行包版本写错挡住，均未运行推理，记录见[干净环境安装失败记录](evidence/2026-11-15-clean-install-failure.json)。修正版本后，新建 venv 的安装、`pip check`、CUDA 检查和 GPU 推理全部通过。设备到 Hugging Face Hub 的模型文件列表请求超时，因此本次没有在线下载模型；改用设备已有的模型目录，并将权重 SHA-256 与固定 Hub 版本的文件哈希逐一比对，均一致。合成表格识别出 12 段预期文字。环境与结果见[干净环境 GPU 复现记录](evidence/2026-11-15-clean-gpu-reproduction.json)。这验证了新环境安装和 GPU 推理，但未验证空模型缓存下的在线下载。

准备这张截图的示例（在另一个安装了 `pymupdf==2.1` 的 Python 环境执行，输出路径应为新文件）：

```bash
export PPOCR_TMP="$(mktemp -d)"
curl -fL https://raw.githubusercontent.com/RapidAI/RapidOCRPDF/bb99e81e397c53fc9b6bba71d49408470c02f09e/tests/test_files/direct_and_image.pdf -o "$PPOCR_TMP/source.pdf"
python - <<'PY'
import os, pymupdf
with pymupdf.open(os.path.join(os.environ['PPOCR_TMP'], 'source.pdf')) as doc:
    page = doc[0]
    page.get_pixmap(matrix=pymupdf.Matrix(2, 2), clip=pymupdf.Rect(0, 0, page.rect.width, min(200, page.rect.height))).save(os.path.join(os.environ['PPOCR_TMP'], 'page.png'))
PY
# 在 OCR 环境中运行上面的 scripts/run.py，并指定 --image "$PPOCR_TMP/page.png"
```
