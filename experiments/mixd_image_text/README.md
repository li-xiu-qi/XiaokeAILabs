# Markdown 图文混合内容处理：图片描述增强工具

把 Markdown 里的图片变成文字，让只读文本的 LLM 也能理解图文混排文档。扫描一份 Markdown 中的远程图片 URL，用多模态模型为每张图生成标题和描述，写回原文对应位置，输出仍是纯文本 Markdown，但语义完整了。

## 解决什么问题

图文混排内容（网页文章、技术博客、报告）直接喂给纯文本 LLM，图片部分会整段丢失，回答里要么忽略图里的信息，要么在引用时把空占位也一起带上。让 LLM 直接看图是一条路，但很多场景用不了多模态输入：做检索索引、灌进只支持文本的模型、或者为了省 token 只保留摘要。这个工具的定位就是那条路——把图片在入库或送模型之前预处理成语义描述。

与 PDF 版面解析（MinerU、pymupdf4llm 一类）不是同一机制：那种靠版面结构还原，这个靠视觉模型生成自然语言描述，输入是已经抓取好的 Markdown 而不是原始 PDF。

## 文件结构

```
mixd_image_text/
├── markdown_image_enhancer.py   # 核心：MarkdownImageEnhancer 类与便捷函数
├── image_utils/
│   ├── async_image_analysis.py  # 多模态调用的异步封装（并发、重试、base64 转换）
│   └── prompts.py               # 视觉模型提示词模板
├── data_process.py              # 命令行测试入口
├── main.py                      # Streamlit 演示应用
└── web_datas/                   # 增强前后对照样例
```

## 核心模块怎么用

`markdown_image_enhancer.py` 是自包含的，文档字符串里写了「独立模块，可用于其他项目」，可以整个文件拷走去用。典型调用：

```python
from markdown_image_enhancer import enhance_markdown_images

enhanced = enhance_markdown_images(
    markdown_text,
    provider="zhipu",            # 视觉模型服务方
    api_key="...",
    vision_model="...",
    max_concurrent=10,           # 并发图片分析数
)
```

三步流程：`extract_img_urls_with_alt` 按 Markdown 图片语法抓出全部图片 URL（空 alt 也能抓）；`analyze_images_batch` 并发送视觉模型，每张图从模型输出里解析出标题和描述；`replace_img_with_analysis` 把结果写回原位。

写回格式是把空 alt 补成模型生成的标题，图片后面加一行引用块放描述：

```markdown
![黑白牧羊犬](http://.../xxx.jpeg)
> 这是一张展示一只黑色和白色相间的边境牧羊犬的照片，它站在户外的小路上。
```

`web_datas/` 下两篇同名文章就是同一内容的处理前后对照（原版 34 行，增强版 40 行，差别全在 alt 和描述行）。

并发和容错都在 `image_utils/async_image_analysis.py`：base64 转换用 aiofiles 做真异步，支持多图并发和失败重试，输出按固定分隔标记 `【图片分析开始】` 到 `【图片分析结束】` 解析，提示词模板在 `prompts.py`，改描述风格只动那一处。

## 演示应用

`main.py` 是个 Streamlit 应用，标题写着「联网搜索对话系统」，实际没有搜索逻辑（代码里的 User-Agent 都没启用），就是读 `web_datas/` 里增强后的文章作为固定上下文，演示增强内容喂给 LLM 做问答。侧边栏的图片分析开关控制是否走增强链路。当演示看，不要当搜索系统用。

## 环境

视觉模型走 OpenAI 兼容接口，通过环境变量配置：`ZHIPU_API_KEY`、`ZHIPU_BASE_URL`、`ZHIPU_MODEL`（data_process 用的键名）；演示应用另用 `GUIJI_API_KEY`、`GUIJI_BASE_URL`、`GUIJI_TEXT_MODEL` 指向文本模型。注意测试样例里的图片 URL 是外部 http 地址，需要能访问外网才能跑通。
