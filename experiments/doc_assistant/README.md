# 文档智能助手：一个 Streamlit RAG 应用的三个演进版本

同一个智能文档助手应用在不同阶段的形态：上传 PDF 或 TXT，解析成文本块，按检索结果把相关内容喂给大模型回答问题。三个子目录不是三个独立项目，是同一应用按检索能力从弱到强的三次实现，共用 Streamlit 界面和同一批模型接口。

## 版本与阅读顺序

| 目录 | 阶段 | 检索方式 | 状态 |
|---|---|---|---|
| `simple_version/` | 最初版 | 无检索，把文档前 15k 字符直接塞进系统提示 | 可运行，教学版 |
| `bm25_version/` | 成熟版 | BM25 稀疏检索，jieba 分词，支持中英文 | 功能完整 |
| `faiss_wip/` | 计划版 | 计划改 FAISS 向量检索 | 开发中，实际仍走 BM25 |

建议先读 `simple_version/xiaoke_doc_assist.py`，单文件就能看到 Streamlit 应用的完整骨架：文件上传、会话管理、流式输出。再看 `bm25_version/`，理解加了真实检索之后的模块化拆分：文档解析（PyMuPDF、pymupdf4llm、MinerU 三条路线）、Markdown 分块、BM25 索引、聊天处理各自独立成文件。

`faiss_wip/` 当前与 BM25 版代码几乎相同，FAISS 依赖和调用都还没写，只在需要继续做向量检索版时参考它的目录骨架，不要当成已完成的实现。

## bm25 版的文档处理链路

这是三个版本里最值得复现的部分，处理的是「PDF 里的图表和公式怎么不丢」：

1. PyMuPDF 提取基础文本
2. pymupdf4llm 提取带结构的 Markdown
3. 复杂 PDF 交给 MinerU 解析图像、表格、公式，模型由 `download_mineru_models.py` 下载
4. 按 Markdown 结构分块，BM25 建索引
5. 查询时自动检测语言、分词、过滤停用词、检索相关块，按上下文长度窗口截取后拼进提示

## 运行

各版本目录都有 `.env.example` 与 `requirements.txt`，进入对应目录按其 README 操作。主入口：

```bash
streamlit run simple_version/xiaoke_doc_assist.py
streamlit run bm25_version/main.py
```
