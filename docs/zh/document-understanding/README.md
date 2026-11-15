# 文档理解与信息提取：规则解析、OCR 与多模态模型

> **阅读导航**：[项目首页](../../../README.md) · [实验地图](../README.md)

PDF、扫描件、研报和合同可能同时包含文字、图片、表格与公式。文档理解要解决的不只是提取字符，还包括保留阅读顺序和结构。这里介绍几类方法的工作方式和局限；同一份文档也可能包含不同类型的内容。

## 主要方法及其局限

| 方法 | 如何处理文档 | 主要局限 |
|---|---|---|
| **PDF 规则解析** | 读取 PDF 中已有的文字、坐标和图形等信息。 | 不能仅靠文字提取识别图片里的文字；阅读顺序、表格和公式也可能无法按预期还原。PDF 能提取出一些文字，并不代表其中的图片内容已经被识别。 |
| **组件化 OCR 流水线** | 将版面分析、文字检测、识别、表格结构识别等步骤组合起来，再整理结果。 | 前面步骤的遗漏或错误可能传递到后续步骤；复杂表格、跨栏阅读顺序和公式仍需要处理，不同组件也需要分别维护。 |
| **多模态模型解析** | 让模型根据页面图像识别内容，并输出文本或结构化结果。 | 生成结果可能遗漏、误认或加入原文没有的内容，格式也不一定正确；算力开销和效果随模型、设备与文档变化，仍需要核对。 |

这些方法关注的问题有交集，但处理机制不同。提取文字为零并不能单独证明页面是扫描件，也不能仅凭文字数量判断整页是否提取完整。

## 专题讲解与实验索引

| 内容 | 主要讲解 | 代码与实验 |
|---|---|---|
| PDF 原生文字与图形提取 | [PDF 规则解析与矢量提取](PDF规则解析与矢量提取.md) | [pdf-rule-parsers](../../../experiments/pdf-rule-parsers/) |
| 文字检测与识别 | [PP-OCRv5 文字检测与识别](PaddleOCR文字检测与识别.md) | [paddle-ocr-detection-recognition](../../../experiments/paddle-ocr-detection-recognition/) |
| 表格结构识别 | [SLANet+ 表格结构识别](SLANet表格结构识别.md) | [table-structure-recognition](../../../experiments/table-structure-recognition/) |
| 表格还原方法及误差 | [表格还原：模块化拼接与端到端多模态](表格还原：模块化拼接与端到端多模态.md) | [assemble_table.py](../../../experiments/table-structure-recognition/scripts/assemble_table.py) · [实验结果](../../../experiments/table-structure-recognition/results/) |
| 文档模型解析与质检 | [MiniCPM 文档解析与质检](MiniCPM文档解析与质检.md) | [minicpm-ocr](../../../experiments/minicpm-ocr/) |
