# 用 RVM 评估人物抠像：区域、发丝边界与推理时间

[English](README.en.md)

本实验用 RVM MobileNetV3 处理 30 张人物照片，检查主体范围、边缘、透明度和模型前向耗时。每张照片独立推理，不评估视频的帧间稳定性，也不包含其他模型的同条件对照。实验动机、指标解读和实际结果见[人物抠像讲解](../../docs/zh/multimodal/人物抠像质量与推理速度.md)。

[scripts](scripts/) · [evidence](evidence/) · [assets](assets/)

复现时先准备固定版本的模型与数据，再运行评测；输出中的逐图指标有助于定位平均值掩盖的失败案例。原始照片和输出只在本地使用，发布案例图前需单独核查对应素材的使用条件。

以下命令均从本实验目录运行。

## 安装与准备

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

2026-11-13 使用 Linux ARM64、NVIDIA GB10、Python 3.12.3、PyTorch 2.15.0+cu132、torchvision 0.29.1+cu130、NumPy 2.5.4 和 Pillow 13.0.0，按下述命令在 CUDA 上处理了固定采样的 30 张 PPM 人物静态照片及其 alpha 标注，并检查了 30 份非空 alpha 图和结果记录。输入来自数据集，并非脚本生成的合成图；本次没有从视频提帧或评估帧间稳定性。此次依赖来自分层环境，基础环境的 `pip check` 报告 `nvidia-cusparselt-cu13` 平台不受支持，尚未验证从全新环境完整安装；其他组合仍需自行检查。该次检查确认脚本和固定权重可运行，不代表所有设备均可复现，也不代替对数据许可和人物肖像使用条件的核查。

2026-11-14 在同类设备上使用上述依赖版本，以复制并调整的独立环境重跑了这 30 张图片，`pip check` 通过，评测正常结束。该环境并非从零安装；这次仍只覆盖静态图片，未评估视频帧间稳定性。

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
