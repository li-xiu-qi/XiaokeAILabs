# LLM 机制：从分词到解码的代码实操

[English](README.en.md)

大语言模型从接收文本到输出 token 的完整链路，各环节通过代码示例拆解；不同示例的依赖和模型权重要求各不相同。这是机制层，回答「每一步在算什么」；性能的定量公式在 `inference-cost-models`（成本模型系列），两边不重复。

## 子目录与阅读顺序

| 目录 | 环节 | 内容 |
|---|---|---|
| [`quantization/`](quantization/README.md) | 模型权重：体积与精度如何取舍 | Qwen3-0.6B 的 BF16、Q4_K_M 与 Q8_0 CPU 实测 |
| [`tokenizer-mechanics/`](../tokenizer-mechanics/README.md) | 输入侧：文本怎么变成 token id | 字节级 BPE 教学实现、真实编码器比较、token 与计费 |
| `logit_and_sampling/` | 输出侧：模型怎么选出下一个词 | logits 到概率分布、argmax 贪心解码、temperature 与 top_k 采样 |
| [`kv_cache/`](kv_cache/README.md) | 推理机制：为什么重复计算、怎么复用 | 缓存传递与完整前缀重算的数值对照 |

按上表顺序读就是一条完整链路：模型先被加载（量化与否决定显存起点），文本进来先分词，模型逐步产出 logits，采样决定下一个 token，KV Cache 决定每步要不要把历史重算一遍。

## 各子目录要点

[`tokenizer-mechanics/`](../tokenizer-mechanics/README.md)：从字节级 BPE 玩具实现出发，解释分词、token 计数与计费，并比较真实编码器。原 `tokenizer/` 含历史示例，依赖与范围各异；其中 `bbpe.py` 用十进制字符串拼接表示字节，合并后存在歧义、无法可靠解码，仅供查看，不应作为分词器使用。

`logit_and_sampling/`：`test_llm_logit.ipynb` 是最小演示（logits 转 softmax、取 top-k 看分布）；`Qwen2.5模型预测机制分析.ipynb` 是完整链路（argmax、temperature、top_k、多步生成）。

[`kv_cache/`](kv_cache/README.md)：使用随机初始化的小模型，对比缓存和完整前缀重算的 logits。示例手动传递框架提供的缓存，不是从零实现注意力内核，也不是速度基准。

`quantization/`：以相同 Qwen3-0.6B 权重比较 GGUF 文件体积、WikiText-2 小样本困惑度与 CPU 推理速度。BF16 路径速度异常，不能把本次速度差推广为通用量化收益。

## 运行环境

KV Cache 示例使用随机初始化模型，无需下载权重；tokenizer 教程的字节级 BPE 只需 Python，真实编码器比较需安装 tiktoken。量化示例的依赖和模型来源见其说明。旧 notebook 和其他脚本各有独立依赖与权重要求。
