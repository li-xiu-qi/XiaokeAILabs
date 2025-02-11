# 筱可AI研习社实战项目仓库

✨【你好，我是筱可，欢迎来到“筱可AI研习社”】✨

🚀 标签关键词： AI实战派开发者 | 技术成长陪伴者 | RAG前沿探索者

🌈 期待与你：
成为"AI+成长"的双向奔赴伙伴！在这里你将获得：

- ✅ 每周更新的AI开发笔记
- ✅ 可复用的代码案例库
- ✅ 前沿技术应用场景拆解
- ✅ 学生党专属成长心法
- ✅ 定期精读好书

这里是**筱可AI研习社**的实战项目仓库！这个仓库主要用于存储和展示为公众号撰写的各类实战项目。我们会不断优化和迭代这些项目，以探索AI的无限可能。

## 整体项目概览

| 项目名称       | 文件名                | 状态     |
| -------------- | --------------------- | -------- |
| 智能文档助手   | `xiaoke_doc_assist.py` | ✅ 已完成 |

## 下一步工作

- 使用RAG技术对智能文档助手进行升级改造。

## 项目介绍

### 智能文档助手

**智能文档助手**是一个基于DeepSeek模型的文档分析工具，旨在帮助用户从技术文档中提取关键信息。该项目使用了以下技术栈：

- **Python**
- **Streamlit**
- **OpenAI API**
- **PyMuPDF**

#### 功能

- 支持PDF和TXT文件的上传和解析
- 提供多种DeepSeek模型选择
- 支持上下文长度和文件内容读取长度的自定义配置
- 实时生成文档分析结果

#### 使用方法

1. 克隆仓库到本地：

    ```bash
    git clone https://github.com/yourusername/XiaokeAILabs.git
    ```

2. 进入项目目录并安装依赖：

    ```bash
    cd projects/xiaoke_doc_assist
    pip install -r requirements.txt
    ```

3. 配置API_KEY

```
将`.env.example`文件改成`.env`, 并配置你的API_KEY。
注意：API_KEY来自硅基流动。
```

4. 运行项目：

```bash

streamlit run xiaoke_doc_assist.py

```

## 未来计划

我们计划在未来的项目中引入RAG（Retrieval-Augmented Generation）技术，以进一步提升文档助手的性能和准确性。具体项目名称和细节将在后续更新中公布，敬请期待！

## 贡献

欢迎大家为我们的项目贡献代码和想法！如果你有任何建议或发现了问题，请提交issue或pull request。

## 许可证

本项目基于 [Apache 2.0 许可证](http://www.apache.org/licenses/LICENSE-2.0) 开源。

---

感谢您的关注和支持！让我们一起探索AI的无限可能！
