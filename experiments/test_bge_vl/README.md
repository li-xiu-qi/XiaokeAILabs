# BGE-VL 多模态模型实战

BGE-VL（视觉-语言嵌入模型）的上手实测：图文混合输入怎么编码、相似度怎么算。用 FlagEmbedding 的接口跑 torch.no_grad() 推理，`model_mapping.txt` 记录模型标识映射，cat/dog 两张测试图做简单文搜图验证。

要系统学图文模型部署（原生推理、ONNX、OpenVINO 量化）看 `jina_deployment` 系列；这里是单模型快速验证。
