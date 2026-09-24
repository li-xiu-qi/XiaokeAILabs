# 知识图谱研究

从原始文档到图谱数据的两个阶段：先有一条通用构建管线（清洗、切分、翻译、LLM 抽取实体关系），再有一个完整成品案例（红楼梦图谱：基础 NLP 与 LLM 增强两条抽取路线、NetworkX 构图、可视化与分析）。

| 子目录 | 阶段 | 内容 |
|---|---|---|
| [test_kg/](test_kg/) | 通用构建管线 | `test_data_prepare`：章节清洗、按章与 Markdown 切分、异步翻译、LLM 实体关系抽取，产出三元组数据喂图谱应用 |
| [test_hong_lou_meng_kg/](test_hong_lou_meng_kg/) | 成品案例 | 红楼梦图谱构建器：预定义人物/地点实体、规则+LLM 混合抽取、NetworkX、GEXF 导出与可视化（Matplotlib / Bokeh / Dash / Streamlit）、中心性与分布分析 |

两者的关系：`test_kg` 处理「任意文档怎么变成三元组」，`test_hong_lou_meng_kg` 回答「三元组怎么变成能看能查的图谱应用」。读顺序建议先管线后案例。

注意：案例目录需要自备 `红楼梦.txt` 原文放在运行目录（不随仓），抽取质量与 LLM API 密钥相关，基础路线（纯规则）不依赖任何 API。
