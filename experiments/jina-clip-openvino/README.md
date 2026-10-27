# Jina CLIP 在 Intel CPU 与 NPU 上的部署验证

本实验验证 `jinaai/jina-clip-v2` 在 Intel CPU 与 NPU 上能否产生**正确且可重复加速**的 OpenVINO 推理结果。这里不把“模型能转换”“设备能编译”当作部署成功，验收必须同时通过向量正确性、双分支冒烟和重复性能测试。

> **核心发现：同一份 FP16 模型在旧 NPU 驱动上输出全 NaN，升级驱动后文本与视觉均通过。NPU 的价值不仅是能运行，在本次固定文本负载中，平均吞吐达到 CPU 的 12.57 倍。**

## 为什么值得做这个实验

多模态向量模型最终服务于检索和排序。转换工具接受模型结构，只能说明格式问题被处理，并不能证明设备计算仍然正确。驱动、编译器和模型适配层任何一环异常，都可能让输出变成 NaN 或失去语义。

这个实验因此回答三个实际问题：

1. 静态转换是否保留原模型的文本与视觉行为，包括 `retrieval.query` 对应的 LoRA。
2. CPU 与 NPU 的输出是否和 PyTorch native 一致，而不是仅仅完成一次推理。
3. 在正确性成立之后，NPU 是否能带来可重复的延迟和吞吐收益。

这条验证链也用于定位设备软件栈问题：如果 CPU 正确而 NPU 异常，不能急着改模型；如果同一 OpenVINO 和 IR 只更换驱动就恢复正常，就应优先检查驱动，而不是把失败归因于模型本身。

## 实验对象与验收口径

| 项目 | 配置 |
| --- | --- |
| 模型 | `jinaai/jina-clip-v2` |
| revision | `e10d47f5691d0454a0fb5d13f46f2199b74cb436` |
| 机器 | Core Ultra 7 155H + Intel AI Boost |
| OpenVINO | 2026.4.1 |
| 当前 NPU 驱动 | 32.0.100.5540（旧对照为 32.0.100.4023） |
| PyTorch / Transformers | 2.14.1 / 4.57.6 |
| 文本输入 | `3 × 512`，三条文本 |
| 视觉输入 | `1 × 3 × 512 × 512` |
| 文本 IR | FP16，保留 `retrieval.query` LoRA |
| 视觉 IR | FP16 |

正确性以 native 向量为参照，逐行计算余弦相似度，并检查 NaN/Inf。验收要求为：输出全部为有限值，最大绝对差不超过 0.02，且最小行余弦不低于 0.999。文本和视觉分别判断，不能用某一分支代替完整模型结论。

## 我们做了什么

- 将文本和视觉转换为固定形状的静态 IR，动态节点均为 0。
- 保留 `retrieval.query` LoRA，并另做关闭 LoRA 的 A/B，检查这种“简化”是否改变语义。
- 在 CPU 与 NPU 上分别运行，以 PyTorch native 输出为基准检查正确性。
- 保持 OpenVINO、IR、输入和 native 参照不变，对 32.0.100.4023 与 32.0.100.5540 做驱动 A/B。
- 额外检查 NNCF INT8，判断“输出不再是 NaN”是否真的等于“向量仍可用”。
- 在正确性通过后，对同一份文本 IR 做 CPU/NPU 顺序基准：预热 2 次、测量 10 次、编译时间不计入稳态延迟。

## 发现了什么

### 1. 关闭 LoRA 不是无代价的转换捷径

启用与关闭 LoRA 的 A/B 余弦仅为 **0.9166–0.9384**。两条路径不能视为等价输出，因此关闭 LoRA 只能用于排障对照，不能冒充保留原模型行为的部署方案。

### 2. CPU 双分支通过，建立了可靠参照

CPU 文本最小行余弦为 **0.99999982**，视觉为 **1.0**。这证明静态转换链路可以产生与 native 高度一致的向量，也使后续 NPU 异常能够被单独隔离。

### 3. 驱动更新直接改变了 NPU 文本的可用性

旧对照使用 OpenVINO 2026.4.1 + 驱动 32.0.100.4023：视觉通过，但文本 FP16 输出全部 NaN。升级到 32.0.100.5540 后，继续使用同一个 OpenVINO、同一份 IR 和相同输入：

| 验证项 | 结果 |
| --- | --- |
| NPU 文本 | 通过，最小行余弦 **0.99999201** |
| NPU 视觉 | 通过，最小行余弦 **0.99997693** |
| 冒烟图片标签 | 正确，最佳标签为 `a photo of a beach` |
| 实际文本编译器 | `CompilerType.DRIVER` |

这组 A/B 说明旧失败与设备软件栈有关，不能据此宣称 Jina 模型或静态转换整体不可用。它也再次证明：**能编译、能执行，不等于结果正确。**

### 4. INT8 没有成为可用的替代路径

NNCF INT8 模型虽已生成，旧驱动下 NPU 最小行余弦只有 **0.06670082**；升级到 5540 后提升到 **0.99768680**，但仍低于本实验 0.999 的验收阈值，CPU 为 **0.99777812**。即使输出不再包含 NaN，也不能视为通过验收。

因此，当前公开主线是 **驱动 5540 + FP16 双分支**，不是通过关闭 LoRA、切换 FP32 或 INT8 绕过正确性检查。

## 同机重复性能基准

两个设备使用同一份 FP16 文本 IR，均先通过 native 检查，再预热 2 次、测量 10 次。延迟是一整个 batch-3 的同步文本编码时间，不是单条短句；吞吐按 `3 / 平均批耗时` 计算。

| 设备 | 平均延迟 | P50 | P95 | 平均吞吐 |
| --- | ---: | ---: | ---: | ---: |
| CPU | 10.84758 s | 10.83500 s | 10.92145 s | 0.27656 sequences/s |
| NPU | 0.86285 s | 0.86276 s | 0.86343 s | 3.47686 sequences/s |

在这组固定文本负载下，NPU 平均延迟约为 CPU 的 **1/12.57**，平均吞吐约为 **12.57 倍**。全部原始耗时保存在 `docs/evidence/` 的基准 JSON 中，加速比由两个平均值直接相除得到。

该结果只适用于本次模型 revision、FP16 IR、batch-3 输入、OpenVINO 2026.4.1 和驱动 5540，不外推到其他 batch、长序列服务或能效。首次驱动切换后的编译时间也不计入稳态性能。

### 视觉分支

视觉使用同一张 beach1.jpg 和 `1 × 3 × 512 × 512` 输入，采用相同的预热与重复测量口径：

| 设备 | 平均延迟 | P50 | P95 | 平均吞吐 |
| --- | ---: | ---: | ---: | ---: |
| CPU | 10.25726 s | 10.25549 s | 10.32516 s | 0.09749 images/s |
| NPU | 0.67428 s | 0.67471 s | 0.67514 s | 1.48306 images/s |

视觉 NPU 平均延迟约为 CPU 的 **1/15.21**，平均吞吐约为 **15.21 倍**。视觉 native 对照同样通过，说明加速没有以牺牲输出正确性为代价。

### DGX Spark CUDA native 补充基线

为区分“Intel CPU 与 NPU 的差异”和“不同推理后端的差异”，另在 DGX Spark（NVIDIA GB10）上以 PyTorch native 跑同模型、同 native 参照和相同预热次数：

| 分支与精度 | 平均延迟 | 平均吞吐 |
| --- | ---: | ---: |
| 文本 CUDA FP32 | 0.08150 s | 36.80894 sequences/s |
| 文本 CUDA FP16 | 0.08230 s | 36.45318 sequences/s |
| 视觉 CUDA FP32 | 0.14874 s | 6.72324 images/s |

这些结果与 OpenVINO 后端不同，只用于补充跨后端基线，不与 NPU 结果拼成同栈加速比。FP32 与 FP16 文本在该负载下差异很小，不能仅凭精度名称推断必然翻倍。

## 真实 COCO 图文检索

为了验证“向量接近 native”是否也能转化为检索质量，使用 [phiyodr/coco2017](https://huggingface.co/datasets/phiyodr/coco2017) 的 validation 前 200 对图片与英文 caption。每张图片取第一条 caption，查询文本使用 `retrieval.query`，文档文本不加 LoRA，与模型卡的用法一致。图片不随仓库发布，可用下方脚本重建子集。

| 检索方向 | 后端 | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 文本查图片 | CUDA native | 0.810 | 0.995 | 0.995 | 0.88609 | 0.91360 |
| 文本查图片 | OpenVINO NPU | 0.810 | 0.995 | 0.995 | 0.88526 | 0.91295 |
| 图片查文本 | CUDA native | 0.825 | 0.990 | 0.995 | 0.89678 | 0.92151 |
| 图片查文本 | OpenVINO NPU | 0.825 | 0.990 | 0.995 | 0.89702 | 0.92173 |

两个方向均有 **198/200 条排名完全一致**，平均绝对排名差分别只有 0.01 和 0.015。这说明 OpenVINO NPU 与 CUDA native 的检索结果不仅汇总指标接近，逐条排序也稳定一致。

编码总耗时方面，CUDA native 的查询文本、文档文本和图片分别为 4.94、3.10、34.38 秒，OpenVINO NPU 分别为 58.16、58.11、145.33 秒。**NPU 在这批负载上没有超过 CUDA native**；它的价值是提供可用的 Intel 本地推理路径，并在与 Intel CPU 对比时表现出明显加速，而不是宣称优于所有 GPU。

该结果只覆盖 200/5000 条验证样本和英文 caption，不代表完整 COCO 或其他语言、其他数据域的表现。

## 复现

### 1. 环境与依赖

需要 Windows 11、Intel NPU 和 **32.0.100.5540 或兼容的新驱动**。旧驱动可用于复现历史失败，但不能与当前结果混合报告。Intel 官方驱动入口：[Intel NPU Driver - Windows](https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html)。

```powershell
cd experiments/jina-clip-openvino
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-openvino.txt
```

模型权重与自定义实现按脚本中的固定 revision 下载，不随仓库分发。首次运行需要网络；后续可用 `--offline`。

### 2. CPU 转换与双分支验证

```powershell
.venv\Scripts\python.exe scripts\run_openvino_smoke.py --device CPU --single-task --expected-label "a photo of a beach" --image images\beach1.jpg --work-dir artifacts --cache-dir models\huggingface
```

该命令生成静态 IR、native 参照和预处理器，并完成 CPU 对照。

### 3. NPU 双分支验证

模型和自定义实现已缓存后，使用同一组静态输入：

```powershell
.venv\Scripts\python.exe scripts\run_openvino_smoke.py --device NPU --single-task --offline --expected-label "a photo of a beach" --image images\beach1.jpg --work-dir artifacts --cache-dir models\huggingface
```

需要重建 IR 时增加 `--force-convert`。更换图片或文本时，应使用新的 `--work-dir`，避免复用旧 native 参照。

### 4. 复用现有 IR 做文本对照

```powershell
.venv\Scripts\python.exe scripts\validate_existing_text_ir.py --device CPU --ir artifacts\fp16\jina-clip-text_v2_fp16.xml --reference artifacts\fp16\native_reference.npz --preprocessor artifacts\fp16\preprocessor --output artifacts\text-CPU.json
.venv\Scripts\python.exe scripts\validate_existing_text_ir.py --device NPU --ir artifacts\fp16\jina-clip-text_v2_fp16.xml --reference artifacts\fp16\native_reference.npz --preprocessor artifacts\fp16\preprocessor --output artifacts\text-NPU.json
```

### 5. 重复文本基准

先 CPU 后 NPU 顺序执行，避免设备任务互相干扰：

```powershell
.venv\Scripts\python.exe scripts\benchmark_text_ir.py --device CPU --ir artifacts\fp16\jina-clip-text_v2_fp16.xml --reference artifacts\fp16\native_reference.npz --preprocessor artifacts\fp16\preprocessor --output artifacts\benchmark-CPU.json --warmup 2 --iterations 10
.venv\Scripts\python.exe scripts\benchmark_text_ir.py --device NPU --ir artifacts\fp16\jina-clip-text_v2_fp16.xml --reference artifacts\fp16\native_reference.npz --preprocessor artifacts\fp16\preprocessor --output artifacts\benchmark-NPU.json --warmup 2 --iterations 10
```

### 6. 重复视觉基准

```powershell
.venv\Scripts\python.exe scripts\benchmark_vision_ir.py --device CPU --ir artifacts\fp16\jina-clip-vision_v2_fp16.xml --reference artifacts\fp16\native_reference.npz --preprocessor artifacts\fp16\preprocessor --image images\beach1.jpg --output artifacts\benchmark-vision-CPU.json --warmup 2 --iterations 10
.venv\Scripts\python.exe scripts\benchmark_vision_ir.py --device NPU --ir artifacts\fp16\jina-clip-vision_v2_fp16.xml --reference artifacts\fp16\native_reference.npz --preprocessor artifacts\fp16\preprocessor --image images\beach1.jpg --output artifacts\benchmark-vision-NPU.json --warmup 2 --iterations 10
```

### 7. 真实图文检索

先在能访问 Dataset Server 和图片源的机器上准备 200 对样本：

```bash
python scripts/prepare_coco_api_subset.py --output-dir data/coco2017-val200 --limit 200
```

OpenVINO 检索前，先生成并验证文档侧无 LoRA 的 IR：

```powershell
.venv\Scripts\python.exe scripts\validate_text_without_lora.py --device CPU --work-dir artifacts --cache-dir models\huggingface
```

随后分别运行 OpenVINO NPU 与 CUDA native：

```powershell
.venv\Scripts\python.exe scripts\evaluate_retrieval.py --backend openvino --device NPU --subset data\coco2017-val200\subset.json --query-ir artifacts\fp16\jina-clip-text_v2_fp16.xml --document-ir artifacts\no-lora-fp16\jina-clip-text_v2_fp16.xml --vision-ir artifacts\fp16\jina-clip-vision_v2_fp16.xml --preprocessor artifacts\fp16\preprocessor --output artifacts\retrieval-coco200-NPU.json
```

```bash
python scripts/evaluate_retrieval.py --backend native --device cuda --subset data/coco2017-val200/subset.json --preprocessor artifacts/fp16/preprocessor --cache-dir models/huggingface --output artifacts/retrieval-coco200-native-cuda.json --batch-size 8 --offline
```

### 8. 辅助检查

```powershell
.venv\Scripts\python.exe scripts\compare_lora_modes.py
.venv\Scripts\python.exe scripts\inspect_ir_dynamic_ops.py artifacts\fp16\jina-clip-text_v2_fp16.xml
.venv\Scripts\python.exe scripts\inspect_ir_dynamic_ops.py artifacts\fp16\jina-clip-vision_v2_fp16.xml
```

## 证据与边界

- `docs/evidence/`：当前驱动环境下的五条核心结果，公开时仅移除主机名并把路径改为仓库相对路径，数值未改。
- `scripts/`：转换、正确性、LoRA、动态节点和重复基准主入口。
- `images/beach1.jpg`：冒烟样例，图片 SHA256 同时记录在结果中。

本实验不证明所有模型都能在 Intel NPU 上运行，也不提供 INT8 部署结论。其他模型、batch 和视觉性能需要分别做正确性与重复基准。生产使用前应重新核对设备、驱动和 OpenVINO 版本。
