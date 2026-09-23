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

1. **K-means聚类检索**：基于聚类的文档分组、查询路由与多样性检索策略，地址： [experiments/clustering_and_dimreduction/test_k_means](experiments/clustering_and_dimreduction/test_k_means)
2. **SimHash相似度**：局部敏感哈希算法实现，用于近似相似度计算，地址： [experiments/test_simhash](experiments/test_simhash)
3. **检索与重排实践**：检索链路四环节上手教程（FAISS 稠密库、BGE-M3 稀疏向量、ColBERT 晚期交互、BGE Reranker 重排），地址： [experiments/retrieval_and_rerank](experiments/retrieval_and_rerank)
4. **文档分块**：结合语义和结构的混合分块、迟分等策略的综合对比（含三个早期分块实验的归档），地址： [experiments/test_text_segmentation](experiments/test_text_segmentation)
5. **NLI相似度迁移**：基于NLI数据的嵌入相似度迁移学习，地址： [experiments/embedding_research/behavior_tests/test_nli_merge_sim_transfer](experiments/embedding_research/behavior_tests/test_nli_merge_sim_transfer)

### 🗃️ 向量数据库

1. **嵌入式数据库学习**：SQLite 与 DuckDB 入门及其向量检索扩展对照（sqlite-vec 虚拟表路线 vs DuckDB ARRAY 列加 HNSW 索引路线），地址： [experiments/embedded_db_tutorials](experiments/embedded_db_tutorials)
2. **五库向量检索对比**：SurrealDB、Milvus、Qdrant、LanceDB、sqlite-vec 五个向量库的写入吞吐、查询延迟、召回率实测对比，揭示 HNSW、IVF_PQ、暴力扫描三种索引形态的差异，并验证随机向量作为 ANN 测试数据的缺陷，地址： [experiments/test_vector_db_bench](experiments/test_vector_db_bench)

### 📝 文本处理

1. **布局排序算法**：针对复杂文档布局的智能排序方法，地址： [experiments/layout_sorter](experiments/layout_sorter)
2. **SpaCy模型微调**：自然语言处理模型的微调和训练，地址： [experiments/spacy_finetune](experiments/spacy_finetune)
3. **句子长度影响分析**：句子长度对相似度计算的影响研究（⚠️ 含未完成的 multilingual 子实验），地址： [experiments/embedding_research/behavior_tests/test_sentence_length](experiments/embedding_research/behavior_tests/test_sentence_length)
4. **代码表格相似度**：特殊格式文本的相似度计算方法，地址： [experiments/embedding_research/behavior_tests/test_sentence_similarity_with_code_or_table](experiments/embedding_research/behavior_tests/test_sentence_similarity_with_code_or_table)

### 🎯 检索增强与重排序

1. **深度搜索（设计与实现）**：深度搜索系统的架构设计与 Google 实现，地址： [experiments/deepsearch_google](experiments/deepsearch_google)

### 🎨 多模态AI技术

1. **多模态 Embedding**：BGE-VL 视觉-语言嵌入快速验证与 CLIP 冻结层轻量微调，地址： [experiments/embedding_research/multimodal_embedding](experiments/embedding_research/multimodal_embedding)
2. **Jina CLIP 部署**：同一图文模型的三种部署形态（原生推理与封装、ONNX/OpenVINO 脚本、NNCF INT8 量化），地址： [experiments/jina_deployment](experiments/jina_deployment)
3. **图文混合处理**：Markdown 图片描述增强工具与演示应用，地址： [experiments/mixd_image_text](experiments/mixd_image_text)

### ⚙️ 模型优化与部署

1. **ONNX模型转换**：模型格式转换与优化部署，地址： [experiments/test_onnx](experiments/test_onnx)

### 🧮 算法基础与数学原理

1. **LLM机制系列**：从分词到解码的全链路机制实操（BPE 从零实现与多种 tokenizer、logits 到采样解码、手动 KV Cache 与耗时实测），地址： [experiments/llm_mechanics](experiments/llm_mechanics)
2. **推理成本模型系列**：12 个可计算性能模型（Roofline、显存、KV Cache、吞吐规模化、Scaling Laws、量化等），输入模型配置与硬件规格即可外推显存、速度与服务吞吐，GB10 实测锚点自测，地址： [experiments/test_inference_theory](experiments/test_inference_theory)
3. **索引成本模型系列**：45 个数据结构与算法的成本模型与实测（公式推导已迁知识库，本仓保留复现脚本与 GB10/faiss 版本绑定实测），地址： [experiments/test_index_theory](experiments/test_index_theory)
4. **知识图谱构建**：知识图谱的构建与查询技术，地址： [experiments/test_kg](experiments/test_kg)
5. **红楼梦知识图谱**：基于红楼梦的知识图谱构建案例，地址： [experiments/test_hong_lou_meng_kg](experiments/test_hong_lou_meng_kg)
6. **function call**：Agent函数调用，地址： [experiments/test_agent/function_calling](experiments/test_agent/function_calling)
7. **UMAP降维**：使用UMAP进行高维数据降维与三维可视化，包含降维前后相似度与距离分析，地址： [experiments/clustering_and_dimreduction/test_umap](experiments/clustering_and_dimreduction/test_umap)
8. **经典算法系列**：BFPRT 选择算法的 Python/Rust 双实现与实测对比、图算法的从零实现（DFS/BFS、Dijkstra、Kruskal），地址： [experiments/classic_algorithms](experiments/classic_algorithms)
9. **激活函数**：常见激活函数的原理与实现，地址： [experiments/test_popular_activate_func](experiments/test_popular_activate_func)
10. **Agent实现**：MCP协议与记忆机制的Agent实现，地址： [experiments/test_agent](experiments/test_agent)

### 📊 模型微调训练与评估

1. **Embedding 研究总组**：评测基准与微调数据、相似度行为专项（句子长度、代码与表格、NLI 迁移）、嵌入与 Reranker 微调、多模态图文（BGE-VL、CLIP），地址： [experiments/embedding_research](experiments/embedding_research)
2. **DPO 训练脚本**： DPO（Direct Preference Optimization）训练脚本，支持按样本量与 epoch 控制，地址： [experiments/test_dpo/dpo.py](experiments/test_dpo/dpo.py)

### ⚡ 判别式决策模型

1. **开源版 Jev 横向测评**：Jev（TypeSafe System One，不生成文本只输出概率的判别式模型）的三个开源复刻路线 Laya 1.1B、Nimble 9B、metask-jev-4B 在同一台 aarch64 GB10 上的延迟与准确率实测，Qwen3-VL-4B 作生成式基线对照，含中文 Noul 题型失效、Nimble CUDA 逐字段无并行等结构性结论，地址： [experiments/test_jev_open_source](experiments/test_jev_open_source)

### 💾 数据存储与格式选型

1. **xlsx、sqlite、json、jsonl 的体积与速度实测**：同一份数据在四种格式下的体积换算与读写耗时，含 xlsx 压缩率上限（500字节约110倍）、索引对查询速度的真实影响（点查快217倍，范围过滤与分组聚合反而变慢4倍以上）、jsonl 的增量能力（追加一行快三个数量级，只取前1000行快66~239倍），地址： [experiments/test_index_theory](experiments/test_index_theory)（数据格式实测已并入该实验）

### 🔧 系统集成与工程化

1. **Rust Python集成**：使用Rust优化Python性能瓶颈，地址： [experiments/test_rust_in_python](experiments/test_rust_in_python)
2. **FastAPI接口服务**：FastAPI 依赖注入与 sqlite 会话管理示例（单文件，内容已归档，不再随仓维护）

### 🧠 模型工程

1. 模型量化：基于 bitsandbytes 的 8-bit 量化（Qwen2.5-0.5B-Instruct，单文件，内容已归档，不再随仓维护）

### 🚀 完整应用

以下是可以直接运行的应用，不是单点实验：

1. **文档智能助手**：Streamlit 文档问答应用的三个演进版本（无检索教学版、BM25 稀疏检索完整版、FAISS 向量检索开发中），支持 MinerU PDF 转换、Markdown 结构切分，地址： [experiments/doc_assistant](experiments/doc_assistant)
2. **RAG 评估工具**：QA 数据集生成与 RAGAs 评估，覆盖上下文召回、答案相关性、实体召回等指标，地址： [experiments/evaluate_rag](experiments/evaluate_rag)

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
