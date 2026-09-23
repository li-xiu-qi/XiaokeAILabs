# RAG 评估：QA 数据集生成与 RAGAs 指标实测

RAG 系统好不好，不能靠「感觉答得不错」判断，要用可量化的指标。这个目录做的是评估这一环：从文档自动生成问答对（QA 数据集），再用 RAGAs 框架对检索和生成质量打分，每个指标单独拆开学。

## 子目录

| 目录 | 内容 |
|---|---|
| `qa_system/` | 从文档自动生成 QA 数据集，含前后端说明与增量模式 |
| `test_ragas/` | RAGAs 评估框架的逐个指标实测与指标含义笔记 |

## qa_system：评估的输入从哪来

评估 RAG 需要问答对，人工标注慢，这个模块从文档自动生成。`qa_generator.py` 负责生成逻辑，`docs/` 下四篇说明分别讲后端架构、配置项、前端交互和增量模式（已生成过的文档不重复生成，语料更新时只补增量）。

## test_ragas：指标逐个数

RAGAs 的指标分两族：检索侧（context precision、context recall、context entities recall）衡量「召回的上下文对不对、全不全」；生成侧（faithfulness 衡量回答是否忠于上下文、response relevancy 衡量是否切题）。`information/` 下六篇笔记逐篇解释每个指标的定义和适用场景，`evaluate_rag.ipynb` 是完整评估流水线，另有 `test_*.py` 单指标脚本可单独跑。

`test_bert.py` 用 BERT 类模型做相关性判断的辅助测试。

## 阅读顺序

先读 `test_ragas/information/` 下的指标笔记建立概念，再跑 `evaluate_rag.ipynb` 看流水线，最后按需用单指标脚本复测。没有现成 QA 数据集的，先用 `qa_system/` 生成一份。
