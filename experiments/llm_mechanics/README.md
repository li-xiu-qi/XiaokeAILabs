# LLM 机制：从分词到解码的代码实操

大语言模型从接收文本到输出 token 的完整链路，每个环节都拆开用代码跑通：不依赖推理框架，用 Qwen2.5 本地模型、HuggingFace tokenizers 库和自己的从零实现演示。这是机制层，回答「每一步在算什么」；性能的定量公式在 `test_inference_theory`（成本模型系列），两边不重复。

## 子目录与阅读顺序

| 目录 | 环节 | 内容 |
|---|---|---|
| `quantization/` | 加载侧：模型怎么进显存 | bitsandbytes 8bit 量化配置、量化与不量化加载对照 |
| `tokenizer/` | 输入侧：文本怎么变成 token id | BPE 从零实现与库训练、SentencePiece、tiktoken、Qwen tokenizer 实测、jieba 与 ngram 对照 |
| `logit_and_sampling/` | 输出侧：模型怎么选出下一个词 | logits 到概率分布、argmax 贪心解码、temperature 与 top_k 采样 |
| `kv_cache/` | 推理性能：自回归为什么重复计算、怎么省 | 手动实现 KV Cache、有无缓存的实测对比、预测机制拆解 |

按上表顺序读就是一条完整链路：模型先被加载（量化与否决定显存起点），文本进来先分词，模型逐步产出 logits，采样决定下一个 token，KV Cache 决定每步要不要把历史重算一遍。

## 各子目录要点

`tokenizer/`：`my_bpe_tokenizer.py` 用 HuggingFace tokenizers 训练一个 BPE；`bbpe.py` 是字节级 BPE 的从零实现（先转 UTF-8 字节再合并，能看到未登录词怎么被兜住）；`test_qwen_tokenizer.py` 看真实模型的分词行为，`test_jieba.py` 和 `ngram_explanation.ipynb` 是与中文方案的对照。训练语料在 `corpus.txt`、`chinese_corpus.txt`。

`logit_and_sampling/`：`test_llm_logit.ipynb` 是最小演示（logits 转 softmax、取 top-k 看分布）；`Qwen2.5模型预测机制分析.ipynb` 是完整链路（argmax、temperature、top_k、多步生成）。

`kv_cache/`：`test_kv_cache.ipynb` 是核心，先写无缓存生成（每步重算全序列），再手写 KV Cache 版本逐 token 拼接，最后实测耗时差异并澄清缓存的真实适用场景；`model_predict.ipynb` 解释一次前向为什么能给出所有位置的预测，是缓存生效的前提。

`quantization/`：`quantize_qwen.py` 单文件。用 `BitsAndBytesConfig` 配 8bit（`llm_int8_threshold=6.0`），同一脚本封装量化开关与 4bit/8bit/bf16/fp16/fp32 多条加载路径，跑之前把权重放到 `MODEL_PATH` 指向的本地目录。与 `test_inference_theory` 的量化成本模型互补：那边算「量化省多少显存」的公式，这边跑「怎么加载」。

## 运行环境

本地 Qwen2.5 权重（kv_cache 与 logit 两个子目录用），Python 加 PyTorch；quantization 子目录额外需要 bitsandbytes；tokenizer 子目录额外需要 tokenizers、sentencepiece、tiktoken，依赖见其 `requirements.txt`。按各脚本和 notebook 内说明准备模型目录后从上到下执行。
