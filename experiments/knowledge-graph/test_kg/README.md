# 知识图谱构建：从原始文档到图谱数据

用 LLM 从非结构化文档抽取实体和关系构建知识图谱的数据准备实验。成品案例（红楼梦图谱）见同组的 [test_hong_lou_meng_kg](../test_hong_lou_meng_kg/)：通用管线在这里落地成一个能跑能看的完整应用。这里是通用的构建管线。

## test_data_prepare 的处理链

| 文件 | 环节 |
|---|---|
| `chapter_cleaner.py` | 章节清洗：去噪声、统一格式 |
| `split_by_chapter.py` / `markdown_split.py` | 按章节与 Markdown 结构切分 |
| `async_translator.py` | 异步翻译（按需） |
| `test_translation.py` | 翻译质量检查 |
| `fallback_openai_client.py` | OpenAI 兼容客户端兜底封装 |
| `data_process.md` | 数据格式与流程说明 |

## 流程

原文按章切分、清洗、按需翻译，再用 LLM 抽取实体关系入库，产出三元组数据喂给图谱应用。`data_process.md` 记录中间数据格式，换数据源时先读它。
