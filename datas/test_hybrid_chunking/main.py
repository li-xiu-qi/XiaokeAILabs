from hybrid_chunking import HybridChunker

if __name__ == "__main__":
    markdown_text = """
测试无标题的情况。
# 第一章 简介
本章介绍了项目的背景和目标。项目旨在通过混合分块算法提升文本处理效率。混合分块算法结合了结构化分块和递归字符分割，能够适应不同长度和结构的文档需求。

## 1.1 背景
随着大模型的发展，文本分块成为RAG等场景的基础需求。传统的分块方法往往只考虑长度，忽略了文档的结构信息，导致块之间的语义连贯性较差。

## 1.2 目标
- 结构化分块
- 递归分割
- 块合并
本项目的目标是实现一种既能保持语义结构，又能灵活控制块大小的分块方法。

# 第二章 方法
本章详细描述了HybridChunker的实现细节，包括分块和合并策略。方法部分分为结构化分块、递归分割和块合并三大部分。

## 2.1 分块流程
首先按Markdown标题分块，然后对超长块递归分割。递归分割时优先按自然语义边界（如段落、句子）进行，无法再分时则按字符长度硬切。

### 2.1.1 结构化分块
通过正则表达式识别Markdown标题，将文档划分为不同层级的块。每个块记录其标题、内容和层级信息。

### 2.1.2 递归字符分割
对于超出最大长度的块，采用递归方式按分隔符分割，优先级为段落、换行、句号、空格等。

## 2.2 合并策略
合并相邻同级块，保证每块不超过指定长度。合并时保留原始块的ID信息，便于追踪来源。

# 第三章 实验
本章通过多个示例验证HybridChunker的有效性，包括中英文混合文本、大段落文本和结构复杂文档。

## 3.1 示例一：中英文混合
This is a test paragraph. 它包含了中英文混合的句子。分块算法需要正确处理不同语言的标点和分隔符。

## 3.2 示例二：大段落文本
本段为超长段落，旨在测试递归分割功能。文本内容较长，需要被分割为多个块。分割时应尽量保持语义完整，避免在单词或句子中间切分。递归分割的优先级应从段落到句子再到字符。

# 第四章 总结
HybridChunker 能够灵活地对结构化文档进行分块，兼顾语义和长度约束，适用于多种文本处理场景。
"""

    chunker = HybridChunker(chunk_size=80)
    print("=== Hybrid Chunking ===")
    chunks = chunker.hybrid_chunk(markdown_text)
    for c in chunks:
        print(f"ID: {c['chunk_id']}, Title: {c['title']}, Level: {c['level']}")
        print(c['content'])
        print("-" * 40)

    print("\n=== Merged Chunks ===")
    merged = chunker.merge_chunks_by_size(chunks, target_chunk_size=150)
    for c in merged:
        print(f"ID: {c['chunk_id']}, Title: {c['title']}, Level: {c['level']}")
        print(f"长度: {len(c['content'])}")
        print(c['content'])
        print("=" * 40)
