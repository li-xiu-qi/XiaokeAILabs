# 筱可AI工程实验仓库

✨【你好，我是筱可，欢迎来到"筱可AI的工程实验仓库"】✨

🌈 期待与你成为"AI+成长"的双向奔赴伙伴！

这里是**筱可AI**的工程实验仓库！

## 📋 免责声明

⚠️ **重要提醒**：

- 实验代码中部分项目可能处于开发阶段或实验性质
- 部分实验可能存在失败案例，这些都是学习过程的一部分
- 请以公众号文章说明为准，代码仅供学习参考
- 在生产环境中使用前，请进行充分的测试和验证

## 📚 技术学习与实验代码

### 🔍 向量检索与相似度计算

1. **MMR多样性检索**：最大边际相关性搜索算法，平衡相关性与多样性，地址： [experiments/test_mmr_search](experiments/test_mmr_search)
2. **K-means聚类检索**：基于聚类的文档分组与多样性检索策略，地址： [experiments/test_k_means](experiments/test_k_means)
3. **SimHash相似度**：局部敏感哈希算法实现，用于近似相似度计算，地址： [experiments/test_simhash](experiments/test_simhash)
4. **BGE稀疏检索**：稀疏向量检索与稠密向量检索对比分析，地址： [experiments/test_bge_sparse](experiments/test_bge_sparse)
5. **语义文档分块**：基于语义的文档分块策略，地址： [experiments/test_semantic_splitter](experiments/test_semantic_splitter)
6. **Colbert检索**：Colbert晚期交互检索模型，地址： [experiments/test_colbert](experiments/test_colbert)
7. **FAISS向量检索**：FAISS向量索引构建与相似度检索，地址： [experiments/test_faiss](experiments/test_faiss)
8. **NLI相似度迁移**：基于NLI数据的嵌入相似度迁移学习，地址： [experiments/test_nli_merge_sim_transfer](experiments/test_nli_merge_sim_transfer)

### 🗃️ 向量数据库数据库

1. **SQLite向量扩展在RAG中的应用**：sqlite-vec扩展的使用教程，实现向量存储与检索，地址： [experiments/test_sqlite](experiments/test_sqlite)
2. **DuckDB数据库在RAG中的应用**：现代分析型数据库DuckDB的向量搜索与全文检索，地址： [experiments/test_duckdb](experiments/test_duckdb)
3. **五库向量检索对比**：SurrealDB、Milvus、Qdrant、LanceDB、sqlite-vec 五个向量库的写入吞吐、查询延迟、召回率实测对比，揭示 HNSW、IVF_PQ、暴力扫描三种索引形态的差异，并验证随机向量作为 ANN 测试数据的缺陷，地址： [experiments/test_vector_db_bench](experiments/test_vector_db_bench)

### 🔤 分词与文本处理

1. **分词算法**：BPE、WordPiece、SentencePiece等分词算法原理与实现，地址： [experiments/test_tokenizer](experiments/test_tokenizer)
2. **文档分块**：结合语义和结构的混合分块策略，地址： [experiments/test_hybrid_chunking](experiments/test_hybrid_chunking)
3. **延迟分块**：先编码后分块的策略，保持上下文连贯性，地址： [experiments/test_late_chunking](experiments/test_late_chunking)
4. **布局排序算法**：针对复杂文档布局的智能排序方法，地址： [experiments/layout_sorter](experiments/layout_sorter)
5. **SpaCy模型微调**：自然语言处理模型的微调和训练，地址： [experiments/spacy_finetune](experiments/spacy_finetune)
6. **句子长度影响分析**：句子长度对相似度计算的影响研究（⚠️ 含未完成的 multilingual 子实验），地址： [experiments/test_sentence_length](experiments/test_sentence_length)
7. **代码表格相似度**：特殊格式文本的相似度计算方法，地址： [experiments/test_sentence_similarity_with_code_or_table](experiments/test_sentence_similarity_with_code_or_table)

### 🎯 检索增强与重排序

1. **BGE重排序模型**：使用BGE Reranker提升检索精度，地址： [experiments/test_rerank](experiments/test_rerank)
2. **BM25增强检索**：传统BM25与现代向量检索的结合，地址： [experiments/test_bm25_augmentation](experiments/test_bm25_augmentation)
3. **深度搜索**：构建一个deepsearch，地址： [experiments/test_deepsearch](experiments/test_deepsearch)
4. **深度搜索(Google)**：基于Google的深度搜索实现，地址： [experiments/deepsearch_google](experiments/deepsearch_google)

### 🎨 多模态AI技术

1. **BGE-VL多模态**：视觉-语言多模态模型的实战应用，地址： [experiments/test_bge_vl](experiments/test_bge_vl)
2. **Jina CLIP**：图文匹配与多模态检索实现，地址： [experiments/test_jina_clip_v2](experiments/test_jina_clip_v2)
3. **CLIP模型微调**：CLIP模型的轻量级微调实现，支持冻结部分层以提高训练效率，地址： [experiments/test_finetune_clip](experiments/test_finetune_clip)
4. **图文混合处理**：Markdown图文混合内容的解析与增强，地址： [experiments/mixd_image_text](experiments/mixd_image_text)

### 🤖 模型优化与部署

1. **OpenVINO加速**：Intel OpenVINO模型优化与推理加速，地址： [experiments/test_openvino](experiments/test_openvino)
2. **ONNX模型转换**：模型格式转换与优化部署，地址： [experiments/test_onnx](experiments/test_onnx)
3. **Jina OpenVINO**：Jina模型的OpenVINO优化方案，地址： [experiments/test_jina_openvino](experiments/test_jina_openvino)
4. **openvino_sentence_transformer**：openvino_sentence_transformer intel NPU部署（⚠️ 含未完成的 preprocessing 子实验），地址： [experiments/test_openvino_sentence_transformer](experiments/test_openvino_sentence_transformer)

### 🧮 算法基础与数学原理

1. **KV缓存实现**：大模型推理中的KV缓机制实现，地址： [experiments/test_kv_cache](experiments/test_kv_cache)
2. **大模型Logit分析**：解析大模型输出概率分布，地址： [experiments/test_llm_logit](experiments/test_llm_logit)
3. **知识图谱构建**：知识图谱的构建与查询技术，地址： [experiments/test_kg](experiments/test_kg)
4. **红楼梦知识图谱**：基于红楼梦的知识图谱构建案例，地址： [experiments/test_hong_lou_meng_kg](experiments/test_hong_lou_meng_kg)
5. **function call**：Agent函数调用，地址： [experiments/test_fc](experiments/test_fc)
6. **UMAP降维**：使用UMAP进行高维数据降维与可视化，包含PCA对比和相似度分析，地址： [experiments/test_umap](experiments/test_umap)
7. **BFPRT算法**：BFPRT（TopK选择）算法实现，地址： [experiments/test_bfprt](experiments/test_bfprt)
8. **图算法学习**：图论基本概念与算法，地址： [experiments/test_graph](experiments/test_graph)
9. **激活函数**：常见激活函数的原理与实现，地址： [experiments/test_popular_activate_func](experiments/test_popular_activate_func)
10. **Agent实现**：MCP协议与记忆机制的Agent实现，地址： [experiments/test_agent](experiments/test_agent)

### 📊 模型微调训练与评估

1. **Embedding微调**：向量模型的数据准备与微调训练，地址： [experiments/test_embedding](experiments/test_embedding)
2. **训练：Embedding 模型**：示例脚本与数据，用于对嵌入模型进行微调（train_embedding.py），地址： [experiments/test_train_embedding](experiments/test_train_embedding)
3. **训练：Reranker（BERT）**：基于 BERT 的重排序模型训练示例（train_bert_rerank.py），含默认训练数据与损失曲线，地址： [experiments/test_train_reranker](experiments/test_train_reranker)
4. **模型下载管理**：ModelScope模型下载与管理工具，地址： [experiments/test_download_modelscope_model](experiments/test_download_modelscope_model)
5. **DPO 训练脚本**： DPO（Direct Preference Optimization）训练脚本，支持按样本量与 epoch 控制，地址： [experiments/test_dpo/dpo.py](experiments/test_dpo/dpo.py)
6. **音频模型微调** 🚧未完成：音频模型微调（当前仅有加载数据集的开头，尚未继续），地址： [experiments/test_finetune_audio](experiments/test_finetune_audio)

### 💾 数据存储与格式选型

1. **xlsx与sqlite的体积换算与速度实测**：xlsx转sqlite后的体积膨胀规律、xlsx压缩率上限（500字节约110倍）、索引对查询速度的真实影响（点查快217倍，但范围过滤和分组聚合会变慢4倍以上），地址： [experiments/test_xlsx_sqlite_size](experiments/test_xlsx_sqlite_size)

### 🔧 系统集成与工程化

1. **Rust Python集成**：使用Rust优化Python性能瓶颈，地址： [experiments/test_rust_in_python](experiments/test_rust_in_python)
2. **FastAPI接口服务**：FastAPI接口服务示例，地址： [experiments/test_fastapi](experiments/test_fastapi)
3. **BFPRT的Rust实现**：BFPRT算法的Rust语言实现，地址： [experiments/test_rs_bfprt](experiments/test_rs_bfprt)

### 🧠 模型工程

1. 模型量化：基于bitsbytes的模型量化，地址： [experiments/test_quantize_model/quantize_qwen.py](experiments/test_quantize_model/quantize_qwen.py)

### 🚀 完整应用

以下是可以直接运行的应用，不是单点实验：

1. **筱可文档助手**：文档解析与问答助手，支持 MinerU PDF 转换、Markdown 结构切分，地址： [experiments/xiaoke_doc_assist](experiments/xiaoke_doc_assist)
2. **文档助手（BM25 检索版）**：同一助手的稀疏检索实现，含 KMP 字符串匹配、路径切分等模块，地址： [experiments/xiaoke_doc_assist_by_bm25](experiments/xiaoke_doc_assist_by_bm25)
3. **文档助手（FAISS 检索版）**：同一助手的向量检索实现，地址： [experiments/xiaoke_doc_assist_by_faiss](experiments/xiaoke_doc_assist_by_faiss)
4. **RAG 评估工具**：QA 数据集生成与 RAGAs 评估，覆盖上下文召回、答案相关性、实体召回等指标，地址： [experiments/evaluate_rag](experiments/evaluate_rag)

## 🤝 贡献指南

我们欢迎所有形式的贡献，包括但不限于：

- 🐛 问题反馈
- 💡 新功能建议
- 🔧 代码优化
- 📝 文档完善

请通过Issue或Pull Request的方式参与贡献。

## 📄 许可证

本项目基于 [Apache 2.0 许可证](http://www.apache.org/licenses/LICENSE-2.0) 开源。

## 📞 联系我

- 🧑‍💻 作者：**li-xiu-qi**
- 📧 邮箱：<lixiuqixiaoke@qq.com>
- 📢 公众号：**筱可AI**
- 🌐 仓库地址：[GitHub](https://github.com/li-xiu-qi/XiaokeAILabs)

---

公众号：
![公众号](https://oss-liuchengtu.hudunsoft.com/userimg/cd/cd7e1ea8a192f17bbdf8efe8418449e8.jpg)

---

感谢您的关注和支持！让我们一起探索AI的无限可能！🚀
