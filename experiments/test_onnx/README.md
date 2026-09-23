# ONNX 全流程：训练、导出、推理到部署优化

一套完整的 ONNX 教程，六个 notebook 按依赖顺序覆盖模型从 PyTorch 到 ONNX 再到部署的全链路。与模型框架耦合的推理换成 ONNX 是为了跨框架和跨硬件部署，这条链路上的每一步都有坑，这里逐个走通。

## Notebook 顺序

| Notebook | 内容 |
|---|---|
| `01_intro_and_setup.ipynb` | ONNX 是什么、解决什么问题、环境搭建 |
| `02_pytorch_model_training.ipynb` | 先用 PyTorch 训练一个基准模型（后续导出用它） |
| `03_export_to_onnx.ipynb` | 导出 ONNX：`torch.onnx.export` 的参数、动态轴、算子支持 |
| `04_onnx_model_operations.ipynb` | 模型操作：读写、剪裁、改图、元数据 |
| `05_onnx_runtime_inference.ipynb` | ONNX Runtime 推理：session 配置、执行提供者、性能对比 |
| `06_model_optimization_deployment.ipynb` | 部署前的优化手段 |

## 与 jina_deployment 的关系

`jina_deployment` 系列里的 ONNX 部分是针对 Jina CLIP 这一个模型的操作实例（转 IR、量化），本目录是通用全流程教材。要做 ONNX 部署先在这里建立全链路概念，再看具体模型的实例。

## 常见坑提示

导出时的动态 batch/序列长度要显式声明，否则部署时形状被写死；算子不支持是导出失败主因，降版本或拆分模型通常能绕；ONNX Runtime 换执行提供者（CUDA、TensorRT）才算拿到硬件加速。
