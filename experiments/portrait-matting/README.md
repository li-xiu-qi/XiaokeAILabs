# 用 RVM 评估人物抠像：区域、发丝边界与推理时间

[English](README.en.md)

本实验用 RVM MobileNetV3 处理 30 张人物照片，检查主体范围、边缘、透明度和模型前向耗时。每张照片独立推理，不评估视频的帧间稳定性，也不包含其他模型的同条件对照。实验动机、指标解读和实际结果见[人物抠像讲解](../../docs/zh/multimodal/人物抠像质量与推理速度.md)。

[scripts](scripts/) · [evidence](evidence/) · [assets](assets/)

复现时先准备固定版本的模型与数据，再运行评测；输出中的逐图指标有助于定位平均值掩盖的失败案例。原始照片和输出只在本地使用，发布案例图前需单独核查对应素材的使用条件。

以下命令均从本实验目录运行。

## 安装与准备

准备 Python 3.12、适用于设备的 PyTorch 和 torchvision，并按本目录 [`requirements.txt`](requirements.txt) 安装 `numpy`、`Pillow`。干净环境 GPU 冒烟脚本为 GB10/Linux aarch64 固定了 PyTorch 2.15.0+cu132、torchvision 0.29.1+cu132 和传递依赖约束；其他设备应按 [PyTorch 官方安装指南](https://pytorch.org/get-started/locally/)选择构建。CPU 也可运行但更慢。模型权重与 PPM-100 数据约需数百 MB，另需为虚拟环境和结果留出空间。固定权重 SHA-256 为 `3c7c1d92033f7c38d6577c481d13a195d7d80a159b960f4f3119ac7b534cf4f8`。

### 从全新环境运行 GPU 冒烟

脚本会创建新的虚拟环境、安装 CUDA 版 PyTorch 和其余依赖，下载固定 RVM 代码、权重与数据，校验哈希，并在 CUDA 上评估固定的 30 张图片。将下列变量设为仓库外的新路径；脚本不会覆盖已有环境或输出。命令从本实验目录执行：

```bash
export XL_RUN_ID="rvm-$(date +%Y%m%d-%H%M%S)"
export XL_DATA_DIR="/path/outside/repository/portrait-matting-data"
export XL_OUT_DIR="/path/outside/repository/portrait-matting-output"
export XL_REQUIRE_EXTERNAL_ENV=1
bash scripts/clean_env_gpu_smoke.sh
```

默认安装目标为 NVIDIA GB10/Linux aarch64；其他平台应先确认 CUDA PyTorch wheel 可用，并设置相应的 `PYTHON_BIN`、`TORCH_INDEX_URL`。脚本固定直接依赖与 PyTorch，传递依赖仍由 pip 解析，并非跨平台锁文件。

准备 Python、适用于设备的 PyTorch 和 torchvision，另安装 `numpy`、`Pillow`。历史记录使用 PyTorch 2.14.0 和 CUDA 13.0；其他依赖的完整版本没有锁定。RVM 上游代码采用 [GPL-3.0](https://github.com/PeterL1n/RobustVideoMatting/blob/master/LICENSE)，使用其代码与模型前应核对相应许可。按 [PyTorch 官方安装指南](https://pytorch.org/get-started/locally/)选择对应平台与 CUDA，CPU 也可运行但更慢。先将 `WORK_DIR` 设为仓库外的已有工作目录。取固定版本 RVM 代码，并从其 [官方仓库的 PyTorch 下载入口](https://github.com/PeterL1n/RobustVideoMatting#download)下载 `rvm_mobilenetv3.pth` 至 `$WORK_DIR/weights/`。代码校验的权重 SHA-256 为 `3c7c1d92033f7c38d6577c481d13a195d7d80a159b960f4f3119ac7b534cf4f8`。

```bash
: "${WORK_DIR:?请先设置仓库外的工作目录}"
mkdir -p "$WORK_DIR/weights"
python3 -m venv "$WORK_DIR/.venv"
source "$WORK_DIR/.venv/bin/activate"
# 按 PyTorch 官方安装指南安装 torch 和 torchvision
git clone https://github.com/PeterL1n/RobustVideoMatting.git "$WORK_DIR/rvm-src"
git -C "$WORK_DIR/rvm-src" checkout 53d74c6826735f01f4406b5ca9075eee27bec094
python -m pip install -r requirements.txt
curl -fL https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3.pth -o "$WORK_DIR/weights/rvm_mobilenetv3.pth"
echo '3c7c1d92033f7c38d6577c481d13a195d7d80a159b960f4f3119ac7b534cf4f8  '"$WORK_DIR"'/weights/rvm_mobilenetv3.pth' | sha256sum -c -
curl -fL https://paddleseg.bj.bcebos.com/matting/datasets/PPM-100.zip -o "$WORK_DIR/PPM-100.zip"
python scripts/prepare_ppm.py --archive "$WORK_DIR/PPM-100.zip" --output "$WORK_DIR/local-ppm-30"
```

采样脚本先检查压缩包 SHA-256 `5a3d9029437954594c26ea773e927c4b7c4caaee7e94f6bbd8e96196c868254a`，输出图像、标注和包含逐文件散列的 manifest。`$WORK_DIR/local-ppm-30` 是本地数据目录，不应提交到公开仓库。文件名或哈希变更会导致检查失败，应重新核对来源，而不是禁用检查。

安装和准备后，可执行 `python -m pip check` 与 `python -c "import torch, torchvision, numpy, PIL; print('CUDA:', torch.cuda.is_available())"`。这只检查依赖导入和 CUDA 可见性，不代表模型加载或实测成功。

2026-11-15 在 NVIDIA GB10（Linux aarch64、Python 3.12.3）上运行上述脚本，从新建虚拟环境开始安装，`pip check` 通过；固定 RVM 权重和 PPM-100 压缩包均通过 SHA-256 校验，30 张图全部完成 CUDA 推理。环境版本为 PyTorch 2.15.0+cu132、torchvision 0.29.1+cu132、NumPy 2.5.4、Pillow 13.0.0。宏平均 IoU 为 0.9730、Dice 为 0.9862、Boundary F1 为 0.8916、MAE 为 0.00857；同步后的模型前向平均约 6.98 ms。详见[干净环境 GPU 评测摘要](evidence/clean-env-gpu-smoke-2026-11-15.json)。这是固定 30 张探索性静态图片评测，不代表更大数据集效果、视频帧间稳定性或其他设备的复现；数据标注许可和人物肖像使用条件仍需分别核查。

2026-11-13 的历史 CUDA 评测使用了分层依赖环境，其中 torchvision 与 PyTorch CUDA 构建不匹配，基础环境 `pip check` 也报告平台不支持的依赖；该记录不应作为环境安装参考。2026-11-14 在复制调整过的隔离环境中复跑，`pip check` 通过，但并非从零安装。11 月 15 日的记录补上了干净安装到真实数据 CUDA 评测的验证。

## 运行单帧评测

```bash
python scripts/run_rvm_ppm.py \
  --manifest "$WORK_DIR/local-ppm-30/manifest.jsonl" --output "$WORK_DIR/local-rvm-results" \
  --rvm-repo "$WORK_DIR/rvm-src" --weights "$WORK_DIR/weights/rvm_mobilenetv3.pth" \
  --device cuda --max-side 1024 --warmup 1 --repeats 3
```

没有 CUDA 时可改为 `--device cpu`。输出目录必须不存在，防止覆盖。RVM 的实际依赖以其固定版本的依赖文件为准；首次执行前应确认本机 PyTorch 能正常加载这些依赖。

输入经过 RGB 解码、最长边不超过 1024 的等比例缩放，转为 0–1 的 FP32 `1×3×H×W`；标签使用双线性缩放到同样尺寸。模型内部低分辨率比例设为 `min(1,512/max(H,W))`。每次前向传入四个空状态，输出为同尺寸 alpha；预测舍入为 8 位 PNG，指标与该保存结果一致。源图片和标签不独立做 EXIF 旋转，避免坐标错位。

每张图预热一次，计时三次；CUDA 计时前后同步。计时涵盖模型前向及同步，不含解码、输入上传、输出下载和写盘。不同图片的尺寸可能不同，所得耗时不能视作固定分辨率吞吐，也不能与端到端耗时直接比较。输出的 `results.json` 包含模型与数据哈希、尺寸、参数、逐次时间、逐图指标与宏平均；`alpha/` 和 `target/` 分别保存预测与缩放标注，二者包含受限素材信息，应留在本地。

## 检查

```bash
python scripts/test_score_mattes.py
python scripts/test_run_rvm_ppm.py
```

这些测试使用合成数据与模拟预测器，检查指标、预处理衔接、结果写出和校验失败；它们不运行 RVM，不能代替真实模型与设备测试。
