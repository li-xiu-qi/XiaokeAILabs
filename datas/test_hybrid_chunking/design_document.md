# HybridChunker 设计文档

## 1. 引言

`HybridChunker` 是一个用于将大型文本文档（尤其是 Markdown 格式）分割为更小块的 Python 类，适用于检索增强生成（RAG）等场景。其核心思想是结合结构化（基于 Markdown 标题）和基于长度的递归分割，兼顾语义连贯性和块大小约束，并支持块的合并。  
**多语言支持：** 用户可通过参数指定语言（如 `language="zh"` 或 `language="en"`），分句与分隔符将自动适配对应语言的符号。

---

## 2. 核心类与方法

### 2.1. 初始化 (`__init__`)

- **目的**：初始化分块器，设置分块参数。
- **参数**：
  - `chunk_size` (int)：单块最大字符数，默认 1000。
  - `markdown_headers` (List[str])：用于结构化分割的 Markdown 标题前缀，默认 `["#", "##", "###"]`。
  - `default_separators` (List[str])：递归字符分割的分隔符，默认根据 `language` 参数自动选择（如英文为 `["\n\n", "\n", ".", " "]`，中文为 `["\n\n", "\n", "。", "！", "？", " "]`）。
  - `language` (str)：语言类型，支持 `"zh"`（中文）、`"en"`（英文），默认 `"zh"`。
- **内部状态**：
  - 存储上述参数。
  - `chunk_counter`：用于分配唯一块 ID，每次分块操作时重置。

---

### 2.2. Markdown 标题分割 (`split_by_markdown_headers`)

- **目的**：按 Markdown 标题结构对文本进行初步分块。
- **逻辑**：
  1. 用正则表达式识别标题行。
  2. 遍历文本行，收集每个标题下的内容。
  3. 第一个标题前的内容归为""，也就是空字符串（level 0）。
  4. 每个块分配唯一 `chunk_id` 和 `level`（标题 `#` 数）。
- **返回**：块字典列表。

---

### 2.3. 递归字符分割 (`recursive_character_split`)

- **目的**：将超出 `chunk_size` 的文本递归分割为更小块，优先按自然边界。
- **逻辑**：
  1. 若文本未超长，直接返回。
  2. 按分隔符优先级依次尝试分割，**分隔符根据 `language` 参数自动适配中英文符号**。
  3. 将分割片段组合为不超过 `chunk_size` 的块。
  4. 若片段仍超长，则递归分割。
  5. 若无合适分隔符，则硬性按长度切分。
- **返回**：文本字符串列表。

---

### 2.4. 句子分割 (`sentence_split`)

- **目的**：按句子边界分割文本。
- **逻辑**：
  1. 用标点符号分句，**标点符号根据 `language` 参数自动适配（如英文为 ".", "!", "?"，中文为 "。", "！", "？"）**。
  2. 合并句子为不超过 `chunk_size` 的块。
- **返回**：文本字符串列表。

---

### 2.5. 混合分块 (`hybrid_chunk`)

- **目的**：主分块流程，结合标题分割和递归分割。
- **逻辑**：
  1. 重置 `chunk_counter`。
  2. 用 `split_by_markdown_headers` 得到初步块。
  3. 对每个块，若未超长直接加入结果，否则递归分割。
  4. 对递归分割出的子块，生成复合 ID 和修改标题（如 "Part x/y"）。
- **返回**：最终块字典列表。

---

### 2.6. 按大小合并块 (`merge_chunks_by_size`)

- **目的**：将相邻块合并，目标为 `target_chunk_size`，同时尊重结构层级。
- **逻辑**：
  1. 顺序遍历块列表，维护当前合并块。
  2. 检查合并后是否超长，不超长则允许合并。
  3. 若不能合并，则将当前合并块加入结果，开始新合并块。
  4. 处理最后一个合并块。
  5. **注意：level=0（无标题）块也允许合并，无需特殊排除。**
  6. **合并时需带上标题，通常采用第一个块的标题作为合并块的标题。**
  7. **如果即将合并的块的 level 比当前合并块中所有块的 level 都小（即级别更高），则不能合并，需另起新块。**
  8. **合并后块的 `chunk_id` 需要重新排序为连续整数（如 1, 2, 3...），不再拼接原有 id。**
  9. **特别注意：如果当前合并缓冲区（buf）为 level=0（无标题）块，遇到下一个有标题（level>0）块时，必须立即断开，不能把无标题块和有标题块合并在一起。**
- **返回**：合并后的块字典列表，`chunk_id` 为新分配的顺序号。

---

### 2.7. 按 ID 合并块 (`_merge_chunks_by_id`)

- **目的**：按指定块列表合并块，私有方法，主要给 merge_chunks_by_size 使用。
- **逻辑**：
  1. 用第一个块的标题和级别，合并内容。
  2. 合并 `original_chunk_ids` 字段，追踪来源。
  3. 不负责分配最终 `chunk_id`，由外部统一排序赋值。
- **返回**：单个合并块字典（不含最终 `chunk_id`，由外部分配）。

---

## 3. 数据结构

**块类（Chunk Class）**：

```python
class Chunk:
    def __init__(
        self,
        chunk_id: Union[int, str],
        title: str,
        content: str,
        level: int,
        original_chunk_ids: Optional[List[Union[int, str]]] = None
    ):
        self.chunk_id = chunk_id                # 唯一标识符
        self.title = title                      # 标题或子块标题
        self.content = content                  # 块内容
        self.level = level                      # Markdown 标题级别（0=无标题，1=#，2=##，…）
        self.original_chunk_ids = original_chunk_ids  # 合并进此块的 chunk_id 列表（可选）
```

合并时 `original_chunk_ids` 字段用于追踪合并来源。

---

## 4. 用法示例

```python
markdown_text = "..." 
chunker = HybridChunker(max_chunk_size=500, language="en")
initial_chunks = chunker.hybrid_chunk(markdown_text)
merged_chunks = chunker.merge_chunks_by_size(initial_chunks, target_chunk_size=1000)
```
