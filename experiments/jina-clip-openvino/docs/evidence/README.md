# 公开证据文件说明

本目录保存多组实际运行结果，包括 Core Ultra 7 155H 的 OpenVINO CPU/NPU 对照与 DGX Spark 的 CUDA native 基线。

| 文件 | 证明内容 |
| --- | --- |
| `driver5540-full-NPU.json` | NPU 文本与视觉双分支通过 native 对照，冒烟标签排序正确 |
| `driver5540-ov2026-4-text-CPU-final.json` | CPU 文本 FP16 通过，包含实际编译器与驱动字段 |
| `driver5540-ov2026-4-text-NPU-final.json` | NPU 文本 FP16 通过，实际为 `CompilerType.DRIVER` |
| `driver5540-benchmark-text-CPU.json` | CPU 预热后 10 次 batch-3 推理的统计值和全部原始耗时 |
| `driver5540-benchmark-text-NPU.json` | NPU 预热后 10 次 batch-3 推理的统计值和全部原始耗时 |
| `driver5540-benchmark-vision-CPU.json` / `driver5540-benchmark-vision-NPU.json` | 视觉重复基准与原始耗时 |
| `text-no-lora-fp16-CPU.json` / `text-no-lora-fp16-NPU.json` | 文档侧无 LoRA 的 CPU/NPU 正确性验证 |
| `native-text-cuda-fp32.json` / `native-text-cuda-fp16.json` / `native-vision-cuda-fp32.json` | DGX Spark CUDA native 基线 |
| `driver5540-ov2026-4-text-CPU-int8.json` / `driver5540-ov2026-4-text-NPU-int8.json` | 新驱动下 INT8 仍未通过验收的对照 |
| `retrieval-coco200-native-cuda-fp32.json` | 200 对 COCO 图文的 CUDA native 双向检索指标与原始排名 |
| `retrieval-coco200-NPU.json` | 同一批图文的 OpenVINO NPU 检索指标与原始排名 |
| `retrieval-coco200-comparison.json` | 两种后端的指标、编码耗时和 198/200 条排名一致性对照 |

公开时只做了两项脱敏处理：移除主机名字段，以及把图片和 IR 路径改为仓库相对路径。所有数值、时间戳、模型 revision、输入形状、延迟与余弦结果均未修改。

结果中的 `npu_driver_version: "1005540"` 是 OpenVINO 返回的原始属性值，对应 Windows 设备信息中的驱动版本 **32.0.100.5540**。

旧驱动 32.0.100.4023 的失败结果以主 README 的 A/B 表概括。早期 NaN 结果和中间探索文件不作为当前复现入口发布。
