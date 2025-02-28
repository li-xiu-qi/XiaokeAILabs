# BM25文档智能助手项目说明文档

## 项目概述

这是一个基于Streamlit和大语言模型的智能文档助手项目，使用BM25算法进行文档检索，支持PDF/TXT文件解析、智能问答和复杂文档内容（图片、公式、表格等）的提取与分析。

### 核心特性

- ✅ 多模型选择（支持DeepSeek和Qwen系列模型）
- ✅ 文档智能解析（支持PDF、TXT文件复杂内容提取）
- ✅ BM25检索增强（基于相关度的文档块智能检索）
- ✅ 可调节对话参数（温度值、上下文长度等）
- ✅ 实时流式响应
- ✅ 对话历史管理

## 项目结构

```
xiaoke_doc_assist_by_bm25/
├── main.py               # 主程序入口
├── chat_handler.py       # 聊天处理逻辑
├── file_processor.py     # 文档处理模块
├── bm25.py               # BM25检索算法实现
├── config.py             # 配置与常量
├── ui_components.py      # UI组件定义
├── split_by_markdown.py  # Markdown文档分块
├── mineru_convert.py     # MinerU文档转换
├── detect_language.py    # 语言检测工具
├── download_mineru_models.py  # MinerU模型下载
├── stopwords.py          # 停用词库
└── requirements.txt      # 项目依赖
```



## 模型支持

| 模型名称          | 模型ID                                   | 适用场景                     |
|-----------------|------------------------------------------|------------------------------|
| DeepSeek-R1-1.5B | deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | 免费快速响应基础问答           |
| DeepSeek-V3     | deepseek-ai/DeepSeek-V3                  | 通用场景对话                   |
| DeepSeek-R1     | deepseek-ai/DeepSeek-R1                  | 复杂推理和长文本理解             |
| Qwen-72B-128k   | Qwen/Qwen2.5-72B-Instruct-128K         | 超大上下文复杂文档分析           |
| Qwen-72B        | Qwen/Qwen2.5-72B-Instruct                | 通用场景对话                   |
| Qwen-14B        | Qwen/Qwen2.5-14B-Instruct                | 基本通用场景对话                 |
| Qwen-7B         | Qwen/Qwen2.5-7B-Instruct                 | 免费快速响应基础问答           |

## 技术架构

### 文档处理流程

1. **文档上传**：支持PDF/TXT格式文件
2. **文档解析**：
   - PyMuPDF提取基础文本
   - pymupdf4llm提取结构化文本
   - MinerU解析复杂PDF（图像、表格、公式等）
3. **文本分块**：使用markdown格式进行分块
4. **索引构建**：将分块内容保存并构建BM25检索索引
5. **检索匹配**：用户查询时进行相关性检索，提取相关文档片段

### BM25检索

使用改进的BM25算法进行文档检索，具有以下特点：

- 支持中英文混合文档
- 自动语言检测与分词处理
- 停用词过滤
- 高效检索排序

## 安装与配置

### 环境要求

- Python 3.10+
- 依赖库详见requirements.txt

### 安装步骤

1. **克隆项目**（或下载项目文件）

2. **安装依赖**

```bash
pip install -r requirements.txt
```

3. **配置API密钥**

创建`.env`文件并写入：

```
API_KEY=your_api_key
```

4. **下载MinerU模型**（用于高质量PDF解析）

```bash
python download_mineru_models.py
```

## 使用说明

### 启动应用

```bash
streamlit run main.py
```

### 界面操作指南

1. **侧边栏设置**
   - 模型选择下拉菜单
   - 温度参数滑动条（0.001-1.2）
   - 上下文长度设置（100-32000 tokens）
   - PDF处理方法选择

2. **文档上传区**
   - 支持拖拽或点击上传
   - 显示已处理文件列表和状态
   - 自动解析文件内容

3. **对话界面**
   - 输入问题并获取智能回复
   - 实时流式显示回答
   - 保持对话历史记录

## 关键组件说明

### file_processor.py

负责文档处理，包含三种PDF解析方式：

- `process_pdf_with_pymupdf`: 基础文本提取
- `process_pdf_with_pymupdf4llm`: 结构化文本提取
- `process_pdf_with_mineru`: 高质量复杂PDF内容提取

### bm25.py

实现了BM25检索算法，支持：

- 英文、中文和混合语言文档检索
- 自定义参数调优
- 索引保存和加载

### chat_handler.py

负责聊天交互处理：

- 管理对话上下文
- 检索相关文档内容
- 调用大语言模型API生成回答
- 处理流式响应

## 注意事项

- MinerU解析需先运行下载模型脚本
- API密钥必须通过`.env`文件配置
- 对话上下文长度受模型限制，建议不超过16000字符
- 大文件解析可能耗时较长，请耐心等待
- 复杂图表和公式解析推荐使用MinerU模式
- 测试可以使用tests/test_datas内预先准备的文档，当然你也可以使用自己的文件测试

## 性能优化建议

- 适当调整上下文长度以平衡记忆与性能
- 对大型PDF文件，考虑使用分页处理
- 根据文档类型选择合适的解析方法
- 复杂问题建议拆分为多个简单问题

## 未来改进计划

- 添加向量数据库支持
- 优化检索算法，结合语义检索
- 增加文档批处理功能
- 支持更多文档格式（Word、Excel等）
- 提供自定义提示词模板

---

**开发者**: 筱可
**版本**: 1.0
