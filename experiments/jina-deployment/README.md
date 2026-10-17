# Jina CLIP 模型部署：从原生推理到 ONNX/OpenVINO 加速

同一个 Jina CLIP 图文模型的四种部署形态，按优化程度从浅到深排列。Jina CLIP 是图文双塔模型，文本和图片编码到同一向量空间，可以做文本搜图、图文匹配、跨语言检索。这几个子目录回答的是同一个问题：模型能跑之后，怎么在 CPU 上把它跑快、把体积压小。

## 子目录与阅读顺序

| 目录 | 形态 | 内容 |
|---|---|---|
| `jina_clip_inference/` | 原生推理加 OpenVINO 封装 | `test_jina_clip_v2.ipynb`（SentenceTransformers 原生推理实测：中文同义句相似度、图文匹配，含两张测试图）、`jina_openvino.py`（`JinaClipOpenVINO` 类，文本视觉双模型分开编译，兼容 SentenceTransformers 调用）、`test_jina_openvino.ipynb`（封装类的使用示例）、`jina_model_path/`（tokenizer 与预处理器配置） |
| `onnx_openvino_scripts/` | ONNX 转 OpenVINO 脚本 | 多个 `infer_*.py` 脚本：直接加载 ONNX、走 OpenVINO runtime、预处理与推理流水线验证 |
| `openvino_nncf_quant/` | NNCF INT8 量化 | `test_openvino-jina-clip.ipynb`：转 OpenVINO IR、文本和图像分支分别量化、量化前后体积与延迟对比、Gradio 演示 |

建议按上表顺序读：先在 jina_clip_inference 确认模型效果和用法，理解朴素推理与工程封装的对应关系，需要极致 CPU 性能时读 openvino_nncf_quant 的量化流程。onnx_openvino_scripts 是中间探索过程，脚本里保留了 ONNX 到 OpenVINO 转换的多种尝试路径，正式用法以封装类为准。

## 三条技术路线的区别

部署优化在这里实际有三条路，不要混为一谈：ONNX 是格式转换，把模型从框架格式变成跨框架的交换格式，本身不保证加速；OpenVINO 是换推理引擎加硬件适配，在 Intel CPU 上靠算子优化提速；NNCF 量化是改精度，把 FP16 进一步压到 INT8，需要准备文本和图像校准数据集，是三者里收益最大但步骤最多的。

## 模型与数据

模型权重不随仓，按各脚本和 notebook 里的说明下载。`jina_clip_inference/jina_model_path/` 里只有 tokenizer 与预处理器配置，不含权重。OpenVINO 量化 notebook 引用的模型结构按 Jina CLIP 的文本和图像双分支组织，量化也要分两次做，不能当成单塔模型一次量化。
