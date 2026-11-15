# Jina CLIP 在 Intel CPU 与 NPU 上的部署验证

[English](README.en.md)

本目录保存实验代码、配置与结果证据。实验目的、原理、数据、结果和边界见[完整实验文档](../../docs/zh/inference/OpenVINO与NPU多模态推理.md)。

[scripts](scripts/) · [evidence](evidence/) · [assets](assets/)

以下命令均从本实验目录运行。

## 复现

### 1. 环境与依赖

复现本次设备结果需要 Windows 11 x86-64、Intel NPU 和本次使用的 **32.0.100.5540 驱动**；其他版本需要重新验证。CPU 检查可独立运行；NPU 检查还要求 OpenVINO 能识别 NPU。安装清单固定了 OpenVINO、PyTorch 与 Transformers，其他包未全部锁定，无法视为完整环境锁文件。旧驱动可用于复现历史失败，但不能与当前结果混合报告。Intel 官方驱动入口：[Intel NPU Driver - Windows](https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html)。

以下 PowerShell 命令从本实验目录运行。将 `$Work` 设置为仓库外的工作目录，`$Image` 指向有权使用的本地图片；`HF_HOME` 一并重定向模型和自定义代码缓存。按需要预留模型、IR 和结果的空间：

```powershell
$Work = 'C:\path\to\jina-work'
$Image = 'C:\path\to\your-image.jpg'
New-Item -ItemType Directory -Force -Path $Work | Out-Null
$env:HF_HOME = "$Work\hf"
python -m venv "$Work\.venv"
$Py = "$Work\.venv\Scripts\python.exe"
& $Py -m pip install -r requirements-openvino.txt
& $Py -c "import openvino as ov; print(ov.Core().available_devices)"
```

[jinaai/jina-clip-v2](https://huggingface.co/jinaai/jina-clip-v2) 模型权重与自定义实现按脚本中的固定 revision 下载，不随仓库分发。准备仓外模型缓存和运行产物目录，并预留下载、转换和运行空间。首次运行需要网络；后续可用 `--offline`。下载使用 Hugging Face Hub 的默认端点；如已通过 `HF_ENDPOINT` 指定其他端点，脚本会保留该设置。模型卡标注 [CC BY-NC 4.0](https://huggingface.co/jinaai/jina-clip-v2)，商用前需另行确认授权。转换使用 `trust_remote_code=True`，应先审阅固定版本的远程代码。

### 2. CPU 转换与双分支验证

```powershell
& $Py scripts\run_openvino_smoke.py --device CPU --single-task --image "$Image" --work-dir "$Work\artifacts" --cache-dir "$Work\hf\hub"
```

该命令生成静态 IR、native 参照和预处理器，并完成 CPU 对照。

### 3. NPU 双分支验证

模型和自定义实现已缓存后，使用同一组静态输入：

```powershell
& $Py scripts\run_openvino_smoke.py --device NPU --single-task --offline --image "$Image" --work-dir "$Work\artifacts" --cache-dir "$Work\hf\hub"
```

`$Image` 使用绝对路径。示例不再检查特定标签；图像与历史样本不同，因此结果不能视为原始视觉结果的复现。需要重建 IR 时增加 `--force-convert`。更换图片或文本时，应使用新的 `--work-dir`，避免复用旧 native 参照。

### 4. 复用现有 IR 做文本对照

```powershell
& $Py scripts\validate_existing_text_ir.py --device CPU --ir "$Work\artifacts\fp16\jina-clip-text_v2_fp16.xml" --reference "$Work\artifacts\fp16\native_reference.npz" --preprocessor "$Work\artifacts\fp16\preprocessor" --output "$Work\artifacts\text-CPU.json"
& $Py scripts\validate_existing_text_ir.py --device NPU --ir "$Work\artifacts\fp16\jina-clip-text_v2_fp16.xml" --reference "$Work\artifacts\fp16\native_reference.npz" --preprocessor "$Work\artifacts\fp16\preprocessor" --output "$Work\artifacts\text-NPU.json"
```

### 5. 重复文本基准

先 CPU 后 NPU 顺序执行，避免设备任务互相干扰：

```powershell
& $Py scripts\benchmark_text_ir.py --device CPU --ir "$Work\artifacts\fp16\jina-clip-text_v2_fp16.xml" --reference "$Work\artifacts\fp16\native_reference.npz" --preprocessor "$Work\artifacts\fp16\preprocessor" --output "$Work\artifacts\benchmark-CPU.json" --warmup 2 --iterations 10
& $Py scripts\benchmark_text_ir.py --device NPU --ir "$Work\artifacts\fp16\jina-clip-text_v2_fp16.xml" --reference "$Work\artifacts\fp16\native_reference.npz" --preprocessor "$Work\artifacts\fp16\preprocessor" --output "$Work\artifacts\benchmark-NPU.json" --warmup 2 --iterations 10
```

### 6. 重复视觉基准

```powershell
& $Py scripts\benchmark_vision_ir.py --device CPU --ir "$Work\artifacts\fp16\jina-clip-vision_v2_fp16.xml" --reference "$Work\artifacts\fp16\native_reference.npz" --preprocessor "$Work\artifacts\fp16\preprocessor" --image "$Image" --output "$Work\artifacts\benchmark-vision-CPU.json" --warmup 2 --iterations 10
& $Py scripts\benchmark_vision_ir.py --device NPU --ir "$Work\artifacts\fp16\jina-clip-vision_v2_fp16.xml" --reference "$Work\artifacts\fp16\native_reference.npz" --preprocessor "$Work\artifacts\fp16\preprocessor" --image "$Image" --output "$Work\artifacts\benchmark-vision-NPU.json" --warmup 2 --iterations 10
```

### 7. 真实图文检索

先在能访问 Dataset Server 和图片源的机器上准备 200 对样本；需要 Python 和 `requests`（Windows 的 `requirements-openvino.txt` 已包含它）：

```powershell
& $Py scripts\prepare_coco_api_subset.py --output-dir "$Work\data\coco2017-val200" --limit 200
```

OpenVINO 检索需要文档侧无任务 LoRA 的文本 IR。先完成上面的 CPU 转换，以准备本地缓存和预处理器，再在 Windows 的实验目录运行：

```powershell
& $Py scripts\convert_document_text_ir.py --preprocessor "$Work\artifacts\fp16\preprocessor" --cache-dir "$Work\hf\hub" --out "$Work\artifacts\no-lora-fp16" --device CPU --offline
```

转换脚本将默认任务 LoRA 设为 `None`，按固定输入生成 IR，并在指定设备上与同一输入的原生输出比较；只有 `validation.json` 的 `ok` 为 `true` 才继续使用。`--out` 须为不存在的新目录。该入口尚未在真实模型上重新执行，转换或校验失败时不能使用生成的 IR，也不能声称已完成端到端复现。可运行 `& $Py -m unittest discover -s tests -v`，通过模拟模型和 OpenVINO 检查任务 LoRA 禁用、输出目录保护和校验失败退出；测试不加载权重，也不能验证转换或 NPU 推理。可将 `--device NPU` 换成目标设备再次使用新的输出目录验证。

CUDA native 在有 NVIDIA GPU 的 Linux 环境运行，按下方步骤创建独立环境，再按 [PyTorch 官方说明](https://pytorch.org/get-started/locally/)安装与 GPU 匹配的 CUDA 构建。历史 CUDA 基线使用 PyTorch 2.14.0+cu130 与 Transformers 4.57.6；新安装组合尚未从零验证。

随后分别运行 OpenVINO NPU 与 CUDA native。NPU 命令使用上述 Windows 工作目录；Linux 设备需另设仓外目录，并将 Windows 工作目录中的 `data/coco2017-val200/` 和 `artifacts/fp16/preprocessor/` 原样复制到 Linux 工作目录对应位置。首次 CUDA 运行需要网络下载固定版本的模型与自定义代码；若已在 Linux 的 `HF_HOME` 缓存完整准备好这些文件，之后可加 `--offline`。跨设备复制缓存时还需保留缓存链接及自定义代码，不能只复制模型权重。

```powershell
& $Py scripts\evaluate_retrieval.py --backend openvino --device NPU --subset "$Work\data\coco2017-val200\subset.json" --query-ir "$Work\artifacts\fp16\jina-clip-text_v2_fp16.xml" --document-ir "$Work\artifacts\no-lora-fp16\jina-clip-text_v2_fp16.xml" --vision-ir "$Work\artifacts\fp16\jina-clip-vision_v2_fp16.xml" --preprocessor "$Work\artifacts\fp16\preprocessor" --output "$Work\artifacts\retrieval-coco200-NPU.json"
```

在 Linux 的实验目录运行；替换 `NATIVE_WORK` 为刚才复制文件的仓外目录：

```bash
export NATIVE_WORK=/path/to/jina-work
export HF_HOME="$NATIVE_WORK/hf"
mkdir -p "$NATIVE_WORK/artifacts"
python3 -m venv "$NATIVE_WORK/.venv"
source "$NATIVE_WORK/.venv/bin/activate"
# 按官方说明安装适配本设备的 CUDA PyTorch 后继续
python -m pip install -r requirements-native.txt
python -c "import torch; assert torch.cuda.is_available(), '需要可用的 CUDA GPU'"
python scripts/evaluate_retrieval.py --backend native --device cuda --subset "$NATIVE_WORK/data/coco2017-val200/subset.json" --preprocessor "$NATIVE_WORK/artifacts/fp16/preprocessor" --cache-dir "$NATIVE_WORK/hf/hub" --output "$NATIVE_WORK/artifacts/retrieval-coco200-native-cuda.json" --batch-size 8
```

### 8. 辅助检查

```powershell
& $Py scripts\compare_lora_modes.py --cache-dir "$Work\hf\hub" --preprocessor "$Work\artifacts\fp16\preprocessor" --output "$Work\artifacts\lora-disable-probe.json"
& $Py scripts\inspect_ir_dynamic_ops.py "$Work\artifacts\fp16\jina-clip-text_v2_fp16.xml"
& $Py scripts\inspect_ir_dynamic_ops.py "$Work\artifacts\fp16\jina-clip-vision_v2_fp16.xml"
```
