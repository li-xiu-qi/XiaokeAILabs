□□□□□□ · 2026 □□ 4 □□

## 关于人工智能文档识别技术的测试报告

本文档用于验证文档解析模型在中文段落、英文段落、表格与数学公式四类典型版式上的识别质量。测试页面由程序渲染生成，内容已知，可逐项核对。

## 1. Introduction

This page exercises four layout element types: body text, an English paragraph, a bordered table, and a displayed formula. Recognition quality is judged against the known source strings below.

## 2. 实验数据

<table><tr><td>模型</td><td>参数量</td><td>吞吐 (页/秒)</td><td>准确率</td></tr><tr><td>MiniCPM-OCR 1.0</td><td>1.3B</td><td>1.20</td><td>93.68</td></tr><tr><td>基线模型 A</td><td>3.8B</td><td>0.81</td><td>91.20</td></tr><tr><td>基线模型 B</td><td>7.2B</td><td>0.45</td><td>92.05</td></tr></table>

3. 数学公式

$$
E = mc^2 \quad | f(x) = \int \Box^{\Lambda_\infty} e^{\Lambda}(-x^2) \, dx = \sqrt{\pi} / 2
$$

$$
\Box \cdot \mathbf{B} = 0, \quad \Box \times \mathbf{E} = -\partial \mathbf{B} / \partial t
$$

综上，模型需要同时正确处理中英文混排、表格线与公式符号。若上述内容被完整识别且顺序正确，则端到端解析链路工作正常。