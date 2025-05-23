### DeepSeek 智能文档助手应用说明文档

✨【你好，我是筱可，欢迎来到“筱可AI研习社”】✨

本次项目主题是：

**基于 Streamlit 和 DeepSeek 的智能文档助手开发实战**

🚀 标签关键词： AI实战派开发者 | 技术成长陪伴者 | RAG前沿探索者

🌈 期待与你：
成为"AI+成长"的双向奔赴伙伴！在这里你将获得：

- ✅ 每周更新的AI开发笔记
- ✅ 可复用的代码案例库
- ✅ 前沿技术应用场景拆解
- ✅ 学生党专属成长心法
- ✅ 定期精读好书

#### 应用概述

本应用是基于DeepSeek大模型的智能对话助手，支持以下核心功能：

- ✅ 多模型选择（提供3个不同规模的模型）
- ✅ 文档分析模式（支持TXT/PDF文件解析）
- ✅ 可调节的对话参数（温度值、上下文长度）
- ✅ 实时流式响应
- ✅ 对话历史管理

#### 功能特点

##### 1. 多模型支持

| 模型名称                          | 适用场景       |
| ----------------------------- | ---------- |
| DeepSeek-R1-1.5B | 免费快速响应基础问答 |
| DeepSeek-V3                   | 通用场景对话     |
| DeepSeek-R1        | 复杂推理和长文本理解 |

##### 2. 文档分析模式

- 支持格式：TXT/PDF
- 处理流程：
  1. 上传文件自动解析
  2. 提取前15k字符作为系统提示（可配置）
  3. 保留最近32k字符上下文（可配置）
  4. 自动切换文档/普通模式

##### 3. 参数配置

- **Temperature** (0.001-1.2)：控制生成随机性
- **上下文长度** (100-30k tokens)：管理对话历史长度
- **最大输出长度** (固定8192 tokens)

#### 使用指南

##### 环境要求

```bash
Python 3.10+
依赖库：streamlit, pymupdf, python-dotenv, openai
```

##### 安装依赖

```
pip install streamlit PyMuPDF python-dotenv openai
```

##### 运行步骤

4. 配置环境变量
创建`.env`文件并写入api_key
注意：API_KEY来自硅基流动。

```env
API_KEY=your_deepseek_api_key
```

5. 启动应用

```bash
streamlit run xiaoke_doc_assist.py
```

##### 界面操作

6. 模型选择区（左侧）
   - 下拉选择模型版本
   - 滑动调节温度值
   - 设置上下文长度

7. 文档上传区
   - 支持拖拽上传
   - 自动识别文件类型
   - 成功加载后显示绿色提示

8. 对话界面
   - 用户输入框位于底部
   - 实时显示对话历史
   - 助理响应带打字机效果

#### 关键模块

- **文件处理** (`process_uploaded_file`)
  - TXT文件：直接读取UTF-8编码
  - PDF文件：使用PyMuPDF解析文本
  - 内容截取：保留前N个字符（N=context_length）

- **消息管理**
  - 保留最近10条消息（5轮对话）
  - 系统消息处理规则：
    - 文件更新时添加文档摘要

- **API调用**

  ```python
  response = client.chat.completions.create(
      model=model_list[selected_model],
      messages=messages_for_api,  # 包含系统提示的完整上下文
      stream=True,                  # 启用流式传输
      temperature=temperature,
      max_tokens=8192              # 固定最大输出长度
  )
  ```

#### 注意事项

10. API密钥需通过`.env`文件配置
11. PDF解析依赖字体嵌入，可能影响格式还原
12. 上下文长度设置需考虑模型实际支持的最大长度
13. 温度值设置低于0.3时可能产生确定性响应
14. 文件上传大小受Streamlit默认限制（200MB）

#### 错误处理机制

- 文件解析异常：显示红色错误提示
- API调用失败：回滚最近消息记录
- 流式中断：保留已接收内容
- 类型错误：自动过滤无效消息类型

#### 交互优化

- 采用流式响应
- 上下文窗口滑动管理
- 文件内容预处理器
- 对话缓存策略（保留最近5轮）

#### 性能优化建议

- 单次对话上下文建议 ≤16000字符
- PDF文件页数建议 ≤50页
- 复杂问题建议拆分为多步骤提问

___
PS: 如果是你，你会如何实现本次的实战项目代码逻辑，你觉得有更好的实现方案吗？
