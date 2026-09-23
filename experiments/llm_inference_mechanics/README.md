# LLM 推理机制：从 Logits 到 KV Cache 的代码实操

围绕 LLM 推理时到底发生了什么的两段机制，全部用 Qwen2.5 本地模型加 PyTorch 手工跑通，不依赖推理框架。这是机制层，回答「模型每一步在算什么、为什么快、为什么慢」；性能的定量公式在 `test_inference_theory`，两边不重复。

## 子目录与阅读顺序

| 目录 | 主题 | 内容 |
|---|---|---|
| `logit_and_sampling/` | 模型怎么选出下一个词 | logits 到概率分布、argmax 贪心解码、temperature 与 top_k 采样 |
| `kv_cache/` | 自回归推理为什么会重复计算、怎么省 | 手动实现 KV Cache、有无缓存的实测对比、预测机制拆解 |

建议先读 logit_and_sampling 建立「每一步的输出是什么」的直觉，再读 kv_cache 理解这些输出在时间维度上的复用。两个目录共用同一批前置知识：Qwen2.5 本地权重、`outputs.logits` 的结构、自回归逐 token 生成。

## logit_and_sampling 的两份 notebook

`test_llm_logit.ipynb` 是最小演示：取一段输入的 logits，转 softmax 概率，取 top-k 看概率分布，演示如何选一个 token 继续生成。重点是看懂 logits 和概率的关系。

`Qwen2.5模型预测机制分析.ipynb` 是完整版，把同一条链路走深：从输入输出结构、每个位置的预测，到贪心解码（`argmax`）、temperature 缩放、top_k 截断、多步生成。采样策略的对比看这一份。

## kv_cache 的两份 notebook

`test_kv_cache.ipynb` 是核心：先写一个不用缓存的生成函数，每步重算全部序列；再手写 `generate_with_manual_kv_cache`，每步只算新 token 的 key/value 并拼到历史后面，最后实测两者的耗时对比。也澄清了缓存真正的适用场景（长序列多步生成才显出收益）。

`model_predict.ipynb` 是预测机制拆解：加载本地 Qwen2.5，逐位置看模型输出，理解一次前向为什么能给出所有位置的预测，是 KV Cache 能生效的前提。

## 运行环境

本地 Qwen2.5 权重（notebook 里写明下载方式），PyTorch。按 notebook 内说明准备模型目录后从上到下执行。
