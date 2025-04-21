# 文本分割器库设计文档

## 1. 项目概述

### 1.1 背景与目的

文本分割器库是一个专门用于将大型文本文档分割成较小、可管理块的工具集。这种分割在以下场景中特别有用：

- 大语言模型(LLM)处理：满足上下文窗口限制
- 文档检索和搜索系统：构建细粒度的文档索引
- 语义分析：在保持上下文的同时处理合适大小的文本单元

### 1.2 核心功能

- 基于多种策略分割文本（字符、正则表达式、递归）
- 支持中文文本的特殊分割需求
- 针对Markdown文档的结构感知分割
- 保留文本片段的元数据信息
- 可配置的分块大小和重叠控制

## 2. 架构设计

### 2.1 整体架构

文本分割器库采用了面向对象的设计模式，通过抽象基类和具体实现类的组合提供灵活的文本分割功能。下面是系统的类层次结构：

```
TextSplitter (抽象基类)
├── CharacterTextSplitter
├── RecursiveCharacterTextSplitter
│   └── ChineseRecursiveTextSplitter
│   └── MarkdownTextRefSplitter
└── MarkdownHeaderTextSplitter (独立类)
```

### 2.2 数据流

文本分割过程的典型数据流如下：

1. 输入原始文本
2. 应用分割策略创建初始分割
3. 根据大小限制合并或进一步分割
4. 为每个文本块附加元数据
5. 输出最终的文本块列表(Chunk对象)

## 3. 数据结构

### 3.1 核心数据类型

#### 3.1.1 Chunk

`Chunk`类是库的基本输出单元，包含文本内容及其相关元数据：

```python
@dataclass
class Chunk:
    content: str = ''           # 分割后的文本内容
    metadata: dict = field(default_factory=dict)  # 相关元数据

    def to_markdown(self, return_all: bool = False) -> str:
        """将块转换为 Markdown 格式。

        Args:
            return_all: 如果为 True，则在内容前包含 YAML 格式的元数据。

        Returns:
            Markdown 格式的字符串。
        """
        # ... 实现细节 ...
```

#### 3.1.2 辅助类型定义

```python
class LineType(TypedDict):
    metadata: Dict[str, str]    # 行元数据
    content: str                # 行内容

class HeaderType(TypedDict):
    level: int                  # 标题级别(1-6)
    name: str                   # 标题类型标识符(如'h1','h2')
    data: str                   # 标题文本内容
```

## 4. 核心组件

### 4.1 TextSplitter (抽象基类)

`TextSplitter`是所有文本分割器的基类，定义了通用接口和实用方法：

- **主要职责**：定义文本分割接口，提供通用功能
- **关键参数**：
  - `chunk_size`：输出块的目标大小
  - `chunk_overlap`：相邻块间的重叠字符数
  - `length_function`：计算文本长度的函数
  - `keep_separator`：分割后是否保留分隔符
  - `add_start_index`：是否记录块在原文中的起始位置
  - `strip_whitespace`：是否移除首尾空白

- **核心方法**：
  - `split_text()`（抽象方法）：实际执行文本分割
  - `create_chunks()`：创建带有元数据的Chunk对象
  - `_merge_splits()`：按大小限制合并小块
  - `_join_chunks()`：使用指定分隔符连接文本块

### 4.2 CharacterTextSplitter

- **主要职责**：使用指定分隔符(字符或正则表达式)分割文本
- **特点**：简单直接，适用于结构清晰的文本
- **独特参数**：
  - `separator`：用于分割的字符串
  - `is_separator_regex`：分隔符是否为正则表达式

### 4.3 RecursiveCharacterTextSplitter

- **主要职责**：通过尝试多种分隔符递归分割文本
- **特点**：更灵活，能处理复杂结构文档
- **独特参数**：
  - `separators`：按优先级排序的分隔符列表

- **工作流程**：
  1. 从最优先的分隔符开始尝试分割
  2. 对于大于目标大小的块，使用下一个分隔符继续分割
  3. 当没有更多分隔符可用时，直接返回过大的块

### 4.4 ChineseRecursiveTextSplitter

- **主要职责**：针对中文文本优化的递归分割器
- **特点**：使用适合中文语法和标点的分隔符
- **默认分隔符**：
  - 段落分隔(`\n\n`)
  - 行分隔(`\n`)
  - 中文句末标点(`。|！|？`)
  - 英文句末标点加空格(`\.\s|\!\s|\?\s`)
  - 分号(`；|;\s`)
  - 逗号(`，|,\s`)

### 4.5 MarkdownTextRefSplitter

- **主要职责**：专门针对Markdown文档的分割器
- **特点**：识别Markdown语法元素进行分割
- **默认分隔符**：包括Markdown标题、代码块、水平线等

### 4.6 MarkdownHeaderTextSplitter

- **主要职责**：基于Markdown标题结构分割文本
- **特点**：
  - 保持文档层次结构
  - 为每个块附加其所属层次的标题元数据
  - 可处理嵌套标题
  - 忽略代码块中的标记

- **独特参数**：
  - `headers_to_split_on`：指定要识别的标题级别
  - `strip_headers`：是否从内容中移除标题

## 5. 实现细节

### 5.1 文本分割策略

库实现了多种文本分割策略：

1. **简单分隔符**：使用固定字符串分割
2. **正则表达式**：使用模式匹配分割
3. **递归分割**：按优先级尝试多种分隔符
4. **结构感知分割**：利用文档结构(如Markdown标题)

### 5.2 块大小控制

为确保分割后的文本块符合指定大小限制，实现了以下机制：

1. **合并小块**：当分割产生过小块时，合并相邻块
2. **递归分割大块**：当块超过大小限制时，尝试其他分隔符
3. **块重叠**：控制相邻块间的重叠程度，提高上下文连贯性

### 5.3 特殊处理

#### 5.3.1 Markdown处理

- **代码块保护**：防止代码块内容被错误分割
- **标题层次追踪**：维护当前活动的标题堆栈
- **元数据提取**：从标题中提取文档结构信息

#### 5.3.2 中文文本处理

- **适应中文标点**：识别中文特有的标点符号
- **句末识别**：准确识别中文句子边界

### 5.4 辅助功能

#### clean_md

提供了`clean_md`函数用于清理Markdown文本：

- 移除代码块
- 简化链接
- 移除HTML标签
- 清理多余空白

## 6. 使用示例

### 6.1 基本用法

```python
# 使用CharacterTextSplitter分割文本
splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separator="\n\n"
)
chunks = splitter.create_chunks(["这是一个很长的文本..."])

# 使用RecursiveCharacterTextSplitter分割文本
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
chunks = recursive_splitter.split_text("这是一个很长的文本...")

# 使用MarkdownHeaderTextSplitter处理Markdown文档
md_splitter = MarkdownHeaderTextSplitter()
with open("article.md", "r", encoding="utf-8") as f:
    text = f.read()
chunks = md_splitter.split_text(text)
```

### 6.2 中文文本处理

```python
chinese_splitter = ChineseRecursiveTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = chinese_splitter.split_text("这是一段中文文本，包含标点符号和句子结构。")
```

### 6.3 自定义配置

```python
# 自定义分隔符和其他参数
custom_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n\n", "\n\n", "\n", "。", "，", ""],
    chunk_size=800,
    chunk_overlap=40,
    length_function=len,
    keep_separator="start",
    add_start_index=True
)
```

## 7. 扩展考虑

### 7.1 潜在扩展方向

- **多语言支持**：增加更多语言特定的分割器
- **语义感知分割**：使用NLP模型识别语义边界
- **更多文档格式**：支持HTML、PDF等格式的结构感知分割
- **并行处理**：优化大文档的分割性能
- **自适应分割**：根据内容复杂度动态调整分割策略

### 7.2 性能优化

- **缓存机制**：对重复文档的分割结果进行缓存
- **流处理**：支持流式处理大型文档
- **批量处理API**：优化多文档批量分割

## 8. 总结

文本分割器库提供了一套全面、灵活的工具，用于将长文本分割成适合处理的小块。通过结合多种分割策略和针对特定文档类型的优化，可满足不同应用场景的需求。核心设计原则包括灵活性、可配置性和文档结构感知，使其能够作为各种文本处理任务的强大基础组件。
