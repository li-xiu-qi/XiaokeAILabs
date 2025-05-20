# 筱可AI研习社实战项目仓库

✨【你好，我是筱可，欢迎来到"筱可AI研习社"】✨

🌈 期待与你：
成为"AI+成长"的双向奔赴伙伴！在这里你将获得：

这里是**筱可AI研习社**的实战项目仓库！

**这个仓库主要用于存储和展示为公众号撰写的各类实战项目。我们会不断优化和迭代这些项目，以探索AI的无限可能。**

## 仓库概览

### 实战项目

| 项目名称                       | 文件名                          | 状态     | 地址                                                                 |
| ------------------------------ | ------------------------------- | -------- | -------------------------------------------------------------------- |
| 智能文档助手                   | `xiaoke_doc_assist.py`          | ✅ 已完成 | [项目地址](projects/xiaoke_doc_assist/README.md)                     |
| 基于BM25检索算法的智能文档助手 | `xiaoke_doc_assist_by_bm25.py` | ✅ 已完成 | [项目地址](projects/xiaoke_doc_assist_by_bm25/README.md)            |

### 学习资料与实验代码

本仓库还包含了大量AI技术相关的学习实验和代码示例，这些资料可以帮助你更好地理解和应用各种AI技术。

#### 向量检索与压缩

| 实验名称 | 描述 | 路径 |
| ------- | ---- | ---- |
| ColBERT实验 | 实现了ColBERT延迟交互检索模型与残差压缩技术 | [datas/colbert](datas/colbert) |
| FAISS向量检索 | FAISS库的基础用法与各种索引类型比较 | [datas/test_faiss](datas/test_faiss) |
| 最大边际相关性搜索 | MMR搜索算法实现，平衡相关性与多样性 | [datas/test_mmr_search](datas/test_mmr_search) |
| K-means聚类 | 基于K-means的文档聚类与多样性检索策略 | [datas/test_k_means](datas/test_k_means) |

#### 文本分块与处理

| 实验名称 | 描述 | 路径 |
| ------- | ---- | ---- |
| 混合分块策略 | 结合语义和结构的文档分块方法 | [datas/test_hybrid_chunking](datas/test_hybrid_chunking) |
| 延迟分块技术 | 先编码后分块的策略，保持上下文连贯性 | [datas/test_late_chunking](datas/test_late_chunking) |
| 布局排序算法 | 针对不同布局的文本排序方法 | [datas/layout_sorter](datas/layout_sorter) |

#### 分词技术

| 实验名称 | 描述 | 路径 |
| ------- | ---- | ---- |
| BPE分词算法 | 字节对编码分词原理与实现 | [datas/test_tokenizer](datas/test_tokenizer) |
| tiktoken测试 | OpenAI tiktoken分词器特性展示 | [datas/test_tokenizer/test_tiktoken.py](datas/test_tokenizer/test_tiktoken.py) |
| SentencePiece | Google SentencePiece分词器用法示例 | [datas/test_tokenizer/test_sentencepiece.py](datas/test_tokenizer/test_sentencepiece.py) |
| 千问分词器 | Qwen/通义千问分词器特性测试 | [datas/test_tokenizer/test_qwen_tokenizer.py](datas/test_qwen_tokenizer/test_qwen_tokenizer.py) |

#### 多模态与图算法

| 实验名称 | 描述 | 路径 |
| ------- | ---- | ---- |
| BGE-VL测试 | 测试BGE视觉-语言多模态模型 | [datas/test_bge_vl](datas/test_bge_vl) |
| Jina CLIP | 测试Jina CLIP图文匹配模型 | [datas/test_jina_clip_v2](datas/test_jina_clip_v2) |
| 图算法学习 | 图算法基础与应用示例 | [datas/test_graph](datas/test_graph) |

#### 其他实验

| 实验名称 | 描述 | 路径 |
| ------- | ---- | ---- |
| 句子长度相似性 | 研究句子长度对相似度计算的影响 | [datas/test_sentence_length](datas/test_sentence_length) |
| 代码/表格相似度 | 特殊格式文本的相似度计算实验 | [datas/test_sentence_similarity_with_code_or_table](datas/test_sentence_similarity_with_code_or_table) |
| ONNX运行时测试 | ONNX加速推理示例 | [datas/test_onnx](datas/test_onnx) |
| LlamaIndex实验 | LlamaIndex框架使用示例 | [datas/test_llama_index](datas/test_llama_index) |
| Jina AI测试 | Jina AI框架功能测试 | [datas/test_jina](datas/test_jina) |

## 下一步工作

- 使用RAG技术对智能文档助手进行升级改造。

#### 使用方法

请参考 [xiaoke_doc_assist项目说明文档](projects/xiaoke_doc_assist/README.md)

## 未来计划

我们计划在未来的项目中改进项目中的RAG（Retrieval-Augmented Generation）技术，以进一步提升文档助手的性能和准确性。具体项目名称和细节将在后续更新中公布，敬请期待！

## 贡献

欢迎大家为我们的项目贡献代码和想法！如果你有任何建议或发现了问题，请提交issue或pull request。

## 许可证

本项目基于 [Apache 2.0 许可证](http://www.apache.org/licenses/LICENSE-2.0) 开源。

---
![公众号](images/筱可AI研习社_860.jpg)
感谢您的关注和支持！让我们一起探索AI的无限可能！
