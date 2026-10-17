# 筱可 AI 工程实验仓库（XiaokeAILabs）

> 李秀奇（筱可）的 AI 工程实验集合。以「能跑、能复现、讲得清」为标准，沉淀 embedding / 检索 / 推理成本 / 多模态 / 判别式模型 / Agent 等方向的实验代码。

## 这是什么

- **定位**：AI 工程实验仓库，不是产品仓库。每个 `experiments/<name>` 是一个自包含实验。
- **标准**：能跑、能复现、结论经得起复核；配套文章发布在公众号「筱可AI」。
- **实验分场**：教学、单点验证、探索中的实验放在**私有**仓 `pkm-hub-lab`；本仓只留**成型、可展示**的实验。

## 实验索引

### 检索与向量
- **embedding_research** — 评测基准 + 微调数据 + 相似度行为专项（句长 / 跨语言 / 代码表格 / NLI 迁移）+ 多模态（BGE-VL、CLIP） → `experiments/embedding_research`
- **retrieval_and_rerank** — 检索链路四环节：FAISS 稠密、BGE-M3 稀疏、ColBERT 晚期交互、BGE Reranker → `experiments/retrieval_and_rerank`
- **test_vector_db_bench** — Milvus / Qdrant / SurrealDB / LanceDB / ChromaDB / sqlite-vec 六库吞吐、延迟、召回实测 → `experiments/test_vector_db_bench`
- **test_text_segmentation** — 语义 + 结构混合分块、迟分等策略对比 → `experiments/test_text_segmentation`

### 推理成本与模型理论
- **test_inference_theory** — 12 个可计算推理性能模型（Roofline、显存、KV Cache、吞吐、Scaling Laws、量化），GB10 实测锚点 → `experiments/test_inference_theory`
- **test_index_theory** — 45 个数据结构 / 算法成本模型与实测（公式迁知识库，本仓留复现脚本） → `experiments/test_index_theory`
- **llm_inference_perf** — 推理成本分阶段报告（MTP、DeepSeek 等） → `experiments/llm_inference_perf`
- **llm_mechanics** — 从模型加载到解码的全链路机制实操 → `experiments/llm_mechanics`

### 多模态与微调
- **image_finetune / video_finetune** — LoRA 微调（图文 / 视频） → `experiments/image_finetune`、`experiments/video_finetune`
- **jina_deployment** — 同一 CLIP 的三种部署形态（原生 / ONNX / NNCF INT8） → `experiments/jina_deployment`

### 判别式模型与 Agent
- **test_jev_open_source** — Jev 判别式模型三个开源复刻（Laya / Nimble / metask-jev）在 GB10 上的延迟与准确率实测 → `experiments/test_jev_open_source`
- **test_agent** — MCP 协议 + 记忆机制的 Agent 实现 → `experiments/test_agent`

### 知识图谱
- **knowledge_graph** — 知识图谱构建与查询（含红楼梦案例） → `experiments/knowledge_graph`

### 应用
- **evaluate_rag** — QA 生成 + RAGAs 评估（上下文召回、答案相关性、实体召回） → `experiments/evaluate_rag`
- **deepsearch_google** — 深度搜索系统设计 + Google 实现 → `experiments/deepsearch_google`

## 免责声明
- 部分实验为开发 / 教学性质，可能存在失败案例，均为学习过程。
- 以公众号文章说明为准，代码仅供学习参考；生产环境使用前请充分测试。

## 许可证
[Apache 2.0](http://www.apache.org/licenses/LICENSE-2.0)

## 联系
- 作者：li-xiu-qi（筱可）｜邮箱：lixiuqixiaoke@qq.com｜公众号：**筱可AI**
- 仓库：https://github.com/li-xiu-qi/XiaokeAILabs
- 实验室介绍页：https://li-xiu-qi.github.io/XiaokeAILabs-website/
