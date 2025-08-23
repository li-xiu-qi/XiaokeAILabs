# 筱可AI研习社实战项目仓库

✨【你好，我是筱可，欢迎来到"筱可AI研习社"】✨

🌈 期待与你成为"AI+成长"的双向奔赴伙伴！

这里是**筱可AI研习社**的实验项目仓库！

## 📋 免责声明

⚠️ **重要提醒**：

- 实验代码中部分项目可能处于开发阶段或实验性质
- 部分实验可能存在失败案例，这些都是学习过程的一部分
- 请以公众号文章说明为准，代码仅供学习参考
- 在生产环境中使用前，请进行充分的测试和验证

## 📚 技术学习与实验代码

### 实验项目

| 实验名称 | 描述 | 技术栈 | 路径 |
| ------- | ---- | ------ | ---- |
| **深度搜索实验** | 深度搜索算法 | - | [datas/test_deepsearch](datas/test_deepsearch) |
| **混合图文处理** | 图文混合内容分析 | PIL, OpenCV | [datas/mixd_image_text](datas/mixd_image_text) |
| **Agent测试** | AI Agent测试| - | [datas/test_agent](datas/test_agent) |

### 🔍 向量检索与相似度计算

| 实验名称 | 描述 | 技术栈 | 路径 |
| ------- | ---- | ------ | ---- |
| **MMR多样性检索** | 最大边际相关性搜索算法，平衡相关性与多样性 | Scikit-learn | [datas/test_mmr_search](datas/test_mmr_search) |
| **K-means聚类检索** | 基于聚类的文档分组与多样性检索策略 | Scikit-learn, Matplotlib | [datas/test_k_means](datas/test_k_means) |
| **SimHash相似度** | 局部敏感哈希算法实现，用于近似相似度计算 | - | [datas/test_simhash](datas/test_simhash) |
| **BGE稀疏检索** | 稀疏向量检索与稠密向量检索对比分析 | FlagEmbedding | [datas/test_bge_sparse](datas/test_bge_sparse) |
| **语义文档分块** | 基于语义的智能文档分块策略 | Transformers | [datas/test_semantic_splitter](datas/test_semantic_splitter) |

### 🗃️ 数据库与存储技术

| 实验名称 | 描述 | 技术栈 | 路径 |
| ------- | ---- | ------ | ---- |
| **SQLite向量扩展在RAG中的应用** | sqlite-vec扩展的使用教程，实现向量存储与检索 | SQLite, sqlite-vec | [datas/test_sqlite](datas/test_sqlite) |
| **DuckDB数据库在RAG中的应用** | 现代分析型数据库DuckDB的向量搜索与全文检索 | DuckDB, VSS扩展 | [datas/test_duckdb](datas/test_duckdb) |

### 🔤 分词与文本处理

| 实验名称 | 描述 | 技术栈 | 路径 |
| ------- | ---- | ------ | ---- |
| **分词算法** | BPE、WordPiece、SentencePiece等分词算法原理与实现 | Transformers, tiktoken | [datas/test_tokenizer](datas/test_tokenizer) |
| **文档分块** | 结合语义和结构的混合分块策略 | LangChain, Transformers | [datas/test_hybrid_chunking](datas/test_hybrid_chunking) |
| **延迟分块** | 先编码后分块的策略，保持上下文连贯性 | Sentence-Transformers | [datas/test_late_chunking](datas/test_late_chunking) |
| **布局排序算法** | 针对复杂文档布局的智能排序方法 | - | [datas/layout_sorter](datas/layout_sorter) |
| **SpaCy模型微调** | 自然语言处理模型的微调和训练 | SpaCy | [datas/spacy_finetune](datas/spacy_finetune) |
| **句子长度影响分析** | 句子长度对相似度计算的影响研究 | - | [datas/test_sentence_length](datas/test_sentence_length) |
| **代码表格相似度** | 特殊格式文本的相似度计算方法 | - | [datas/test_sentence_similarity_with_code_or_table](datas/test_sentence_similarity_with_code_or_table) |

### 🎯 检索增强与重排序

| 实验名称 | 描述 | 技术栈 | 路径 |
| ------- | ---- | ------ | ---- |
| **BGE重排序模型** | 使用BGE Reranker提升检索精度 | FlagEmbedding | [datas/test_rerank](datas/test_rerank) |
| **BM25增强检索** | 传统BM25与现代向量检索的结合 | - | [datas/test_bm25_augmentation](datas/test_bm25_augmentation) |
| **深度搜索系统** | 构建企业级智能搜索系统的完整方案 | - | [datas/test_deepsearch](datas/test_deepsearch) |

### 🎨 多模态AI技术

| 实验名称 | 描述 | 技术栈 | 路径 |
| ------- | ---- | ------ | ---- |
| **BGE-VL多模态** | 视觉-语言多模态模型的实战应用 | BGE-VL, Transformers | [datas/test_bge_vl](datas/test_bge_vl) |
| **Jina CLIP** | 图文匹配与多模态检索实现 | Jina AI | [datas/test_jina_clip_v2](datas/test_jina_clip_v2) |

### 🤖 模型优化与部署

| 实验名称 | 描述 | 技术栈 | 路径 |
| ------- | ---- | ------ | ---- |
| **OpenVINO加速** | Intel OpenVINO模型优化与推理加速 | OpenVINO, ONNX | [datas/test_openvino](datas/test_openvino) |
| **ONNX模型转换** | 模型格式转换与优化部署 | ONNX Runtime | [datas/test_onnx](datas/test_onnx) |
| **Jina OpenVINO** | Jina模型的OpenVINO优化方案 | Jina AI, OpenVINO | [datas/test_jina_openvino](datas/test_jina_openvino) |
| **Sentence-Transformers优化** | 句子编码模型的性能优化 | Sentence-Transformers | [datas/test_openvino_sentence_transformer](datas/test_openvino_sentence_transformer) |

### 🧮 算法基础与数学原理

| 实验名称 | 描述 | 技术栈 | 路径 |
| ------- | ---- | ------ | ---- |
| **图算法实战** | 图论算法在AI中的应用 | NetworkX | [datas/test_graph](datas/test_graph) |
| **KV缓存优化** | 大模型推理中的KV缓存机制优化 | - | [datas/test_kv_cache](datas/test_kv_cache) |
| **大模型Logit分析** | 解析大模型输出概率分布 | - | [datas/test_llm_logit](datas/test_llm_logit) |
| **知识图谱构建** | 知识图谱的构建与查询技术 | Neo4j, NetworkX | [datas/test_kg](datas/test_kg) |
| **红楼梦知识图谱** | 基于红楼梦的知识图谱构建案例 | - | [datas/test_hong_lou_meng_kg](datas/test_hong_lou_meng_kg) |
| **函数调用** |Agent函数调用 | openai | [datas/test_fc](datas/test_fc) |

### 📊 数据处理与评估

| 实验名称 | 描述 | 技术栈 | 路径 |
| ------- | ---- | ------ | ---- |
| **Embedding微调** | 向量模型的数据准备与微调训练 | Transformers | [datas/test_embedding](datas/test_embedding) |
| **训练：Embedding 模型** | 示例脚本与数据，用于对嵌入模型进行微调（train_embedding.py） | Transformers, PyTorch | [datas/test_train_embedding](datas/test_train_embedding) |
| **训练：Reranker（BERT）** | 基于 BERT 的重排序模型训练示例（train_bert_rerank.py），含默认训练数据与损失曲线 | Transformers, PyTorch | [datas/test_train_reranker](datas/test_train_reranker) |
| **模型下载管理** | ModelScope模型下载与管理工具 | ModelScope | [datas/test_download_modelscope_model](datas/test_download_modelscope_model) |

### 🔧 系统集成与工程化

| 实验名称 | 描述 | 技术栈 | 路径 |
| ------- | ---- | ------ | ---- |
| **FastAPI集成** | 构建高性能AI API服务 | FastAPI | [datas/test_fastapi](datas/test_fastapi) |
| **Rust Python集成** | 使用Rust优化Python性能瓶颈 | Rust, PyO3 | [datas/test_rust_in_python](datas/test_rust_in_python) |
| **LlamaIndex框架** | 企业级RAG应用开发框架 | LlamaIndex | [datas/test_llama_index](datas/test_llama_index) |

### 🧠 模型工程

| 实验名称|描述|技术栈|路径|
| ------- | ---- | ------ | ---- |
|模型量化|基于bitsbytes的模型量化|bitsbytes|[datas/test_quantize_model/quantize_qwen.py](datas/test_quantize_model/quantize_qwen.py) |

## 🤝 贡献指南

我们欢迎所有形式的贡献，包括但不限于：

- 🐛 问题反馈
- 💡 新功能建议
- 🔧 代码优化
- 📝 文档完善

请通过Issue或Pull Request的方式参与贡献。

## 📄 许可证

本项目基于 [Apache 2.0 许可证](http://www.apache.org/licenses/LICENSE-2.0) 开源。

## 📞 联系我们

- � 作者：**li-xiu-qi**
- �📧 邮箱：<lixiuqixiaoke@qq.com>
- 📢 公众号：**筱可AI**
- 🌐 仓库地址：[GitHub](https://github.com/li-xiu-qi/XiaokeAILabs)

---

公众号：
![公众号](images/筱可AI研习社_860.jpg)

---

感谢您的关注和支持！让我们一起探索AI的无限可能！🚀
