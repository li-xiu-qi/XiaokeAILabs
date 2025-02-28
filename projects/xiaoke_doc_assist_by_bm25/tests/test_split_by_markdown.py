from typing import List, Dict

from split_by_markdown import split_markdown_by_headers, \
    merge_markdown_chunks

# 使用示例
if __name__ == "__main__":
    # 中英文混合示例
    markdown_text = """
# 主标题
这是一个混合内容的示例。It has both Chinese and English。这里是中文部分。

## 二级标题
更多内容在这里。这是中文句子！This is an English sentence with more words to test splitting.

没有标题的段落。Another paragraph without header。这里继续中文。
    """

    initial_chunks = split_markdown_by_headers(markdown_text)
    # 测试自动检测语言
    merged_chunks_auto = merge_markdown_chunks(initial_chunks, chunk_size=50, chunk_overlap=10)


    # 测试指定中文
    merged_chunks_zh = merge_markdown_chunks(initial_chunks, chunk_size=50, chunk_overlap=10, language='zh')
    # 测试指定英文
    merged_chunks_en = merge_markdown_chunks(initial_chunks, chunk_size=50, chunk_overlap=10, language='en')

    print("自动检测语言结果:")
    for i, chunk in enumerate(merged_chunks_auto):
        print(f"Chunk {i}:")
        print(f"Header: '{chunk['header']}'")
        print(f"Level: {chunk['level']}")
        print(f"Content: '{chunk['content']}'")
        print("-" * 50)

    print("\n指定中文分割结果:")
    for i, chunk in enumerate(merged_chunks_zh):
        print(f"Chunk {i}:")
        print(f"Header: '{chunk['header']}'")
        print(f"Level: {chunk['level']}")
        print(f"Content: '{chunk['content']}'")
        print("-" * 50)