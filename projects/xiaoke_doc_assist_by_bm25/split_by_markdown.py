#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：筱可
# 2025-02-22
"""
#### 使用说明：
该模块提供了处理Markdown文本的功能，包括语言检测、按标题分块和按长度合并分块。

#### 主要功能：
1. `detect_language`: 检测文本中的主语言，判断是否以中文或英文为主。
2. `split_markdown_by_headers`: 根据标题级别将Markdown文本分块。
3. `merge_markdown_chunks`: 将已分块的Markdown文本合并，支持按固定长度分割，同时考虑中文和英文标点。
4. `_split_text_by_size`: 按照给定长度对文本进行分割，支持中英文智能分割。

#### 参数说明：
1. detect_language 函数：
    - text (str): 输入的文本。
    - 返回值：'zh' 表示中文占主导，'en' 表示英文占主导。

2. split_markdown_by_headers 函数：
    - text (str): 输入的Markdown文本。
    - min_level (int): 处理的最小标题级别，默认为1。
    - max_level (int): 处理的最大标题级别，默认为6。
    - 返回值：一个字典列表，包含标题、内容和标题级别。

3. merge_markdown_chunks 函数：
    - chunks (List[Dict[str, any]]): 已分块的Markdown文本。
    - chunk_size (int): 每个块的最大长度，默认为1000。
    - chunk_overlap (int): 块之间的重叠长度，默认为200。
    - separator (str): 块之间的分隔符，默认为"\n\n"。
    - language (str): 语言选项，默认为'auto'，自动检测语言，'zh' 表示中文，'en' 表示英文。
    - 返回值：合并后的Markdown分块。

4. _split_text_by_size 函数：
    - text (str): 要分割的文本。
    - chunk_size (int): 每个分块的最大长度。
    - chunk_overlap (int): 分块之间的重叠部分。
    - header (str): 当前分块的标题。
    - level (int): 当前分块的标题级别。
    - language (str): 语言选项，默认为'auto'，可以手动指定'zh'或'en'。
    - 返回值：一个字典列表，每个字典包含一个分块的标题、内容和级别。

#### 注意事项：
1. 确保传入的文本符合Markdown格式。
2. 对于合并后的文本，可能需要手动调整块之间的分隔符和内容。
3. `chunk_size` 和 `chunk_overlap` 需要合理设置，以避免产生过小或过大的文本块。

#### 更多信息：
无
"""

from typing import List, Dict

from detect_language import split_text_detect_language




def split_markdown_by_headers(
        text: str,
        min_level: int = 1,
        max_level: int = 6
) -> List[Dict[str, any]]:
    """
    功能描述: 将Markdown文本按标题级别分块。

    参数:
        text (str): 输入的Markdown文本。
        min_level (int): 处理的最小标题级别，默认为1。
        max_level (int): 处理的最大标题级别，默认为6。

    返回值:
        List[Dict[str, any]]: 一个字典列表，包含标题、内容和标题级别。
    """
    if not isinstance(text, str):
        raise TypeError("text参数必须是字符串")
    if not isinstance(min_level, int) or not isinstance(max_level, int):
        raise TypeError("min_level和max_level必须是整数")
    if not text.strip():
        return ""    
    # 确保标题级别在1到6之间
    min_header_level = max(1, min(6, min_level))
    max_header_level = max(min_header_level, min(6, max_level))

    # 将文本按行分割
    lines = text.split('\n')
    chunks = []
    current_chunk = {'header': '', 'content': [], 'level': 0}

    # 遍历每一行
    for line in lines:
        is_header = False
        # 检查当前行是否为标题
        for level in range(min_header_level, max_header_level + 1):
            header_prefix = '#' * level + ' '
            if line.strip().startswith(header_prefix):
                # 如果当前块有内容或标题，将其添加到块列表中
                if current_chunk['content'] or current_chunk['header']:
                    chunks.append({
                        'header': current_chunk['header'],
                        'content': '\n'.join(current_chunk['content']).strip(),
                        'level': current_chunk['level']
                    })
                # 初始化新的块
                current_chunk = {
                    'header': line.strip()[level + 1:].strip(),
                    'content': [],
                    'level': level
                }
                is_header = True
                break
        # 如果当前行不是标题，将其添加到当前块的内容中
        if not is_header:
            current_chunk['content'].append(line)

    # 将最后一个块添加到块列表中
    if current_chunk['content'] or current_chunk['header']:
        chunks.append({
            'header': current_chunk['header'],
            'content': '\n'.join(current_chunk['content']).strip(),
            'level': current_chunk['level']
        })
    return chunks


def merge_markdown_chunks(
        chunks: List[Dict[str, any]],
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separator: str = "\n\n",
        language: str = 'auto'  # 新增参数控制语言分割类型
) -> List[Dict[str, any]]:
    """
    功能描述: 合并Markdown分块，按固定长度分割，支持中英文标点。

    参数:
        chunks (List[Dict[str, any]]): 已分块的Markdown文本。
        chunk_size (int): 每个块的最大长度，默认为1000。
        chunk_overlap (int): 块之间的重叠长度，默认为200。
        separator (str): 块之间的分隔符，默认为"\n\n"。
        language (str): 语言选项，默认为'auto'，自动检测语言，'zh' 表示中文，'en' 表示英文。

    返回值:
        List[Dict[str, any]]: 合并后的Markdown分块。
    """
    if not isinstance(chunks, list):
        raise TypeError("chunks参数必须是列表")
    if not isinstance(chunk_size, int) or not isinstance(chunk_overlap, int):
        raise TypeError("chunk_size和chunk_overlap必须是整数")
    if chunk_size <= chunk_overlap:
        raise ValueError("chunk_size必须大于chunk_overlap")
    if chunk_size <= 0 or chunk_overlap < 0:
        raise ValueError("chunk_size必须大于0且chunk_overlap不能为负数")

    # 初始化合并后的块列表
    merged_chunks = []
    # 当前文本块内容
    current_text = ""
    # 当前文本块的标题级别
    current_level = 0
    # 当前文本块的标题
    current_header = ""

    # 遍历所有的分块
    for chunk in chunks:
        # 获取当前分块的文本内容，如果有标题则包含标题
        chunk_text = f"{chunk['header']}\n{chunk['content']}" if chunk['header'] else chunk['content']

        # 如果当前分块有标题且当前文本块不为空
        if chunk['level'] > 0 and current_text:
            # 如果当前文本块不为空白
            if current_text.strip():

                # 将当前文本块按指定大小分割并添加到合并后的块列表中
                merged_chunks.extend(_split_text_by_size(
                    current_text.strip(),
                    chunk_size,
                    chunk_overlap,
                    current_header,
                    current_level,
                    language
                ))

            # 重置当前文本块内容
            current_text = ""
            # 更新当前文本块的标题和标题级别
            current_header = chunk['header']
            current_level = chunk['level']
        # 将当前分块的文本内容添加到当前文本块中，使用分隔符连接
        if current_text.strip():
            current_text += separator + chunk_text
        else:
            current_text = chunk_text

    # 如果最后一个文本块不为空白
    if current_text.strip():
        # 将最后一个文本块按指定大小分割并添加到合并后的块列表中
        merged_chunks.extend(_split_text_by_size(
            current_text.strip(),
            chunk_size,
            chunk_overlap,
            current_header,
            current_level,
            language
        ))

    # 返回合并后的块列表
    return merged_chunks


def _split_text_by_size(
        text: str,
        chunk_size: int,
        chunk_overlap: int,
        header: str,
        level: int,
        language: str = 'auto'
) -> List[Dict[str, any]]:
    """
    功能描述: 内部函数：按固定长度分割文本，支持中英文智能分割

    参数:
        text (str): 要分割的文本。
        chunk_size (int): 每个分块的最大长度。
        chunk_overlap (int): 分块之间的重叠部分。
        header (str): 当前分块的标题。
        level (int): 当前分块的标题级别。
        language (str): 语言选项，默认为'auto'，可以手动指定'zh'或'en'。

    返回值:
        List[Dict[str, any]]: 按要求分割后的文本块。
    """
    # 定义中英文分割标点
    EN_PUNCTUATION = ' \n\t.,!?'
    ZH_PUNCTUATION = ' \n\t。！？，、；'

    # 确定使用哪种语言的分割规则
    if language == 'auto':
        # 自动检测语言
        lang = split_text_detect_language(text)
    else:
        # 使用指定的语言
        lang = language.lower()
    
    # 根据语言选择分割标点符号
    if lang == 'zh':
        split_chars = ZH_PUNCTUATION 
    elif lang == 'en':
        split_chars = EN_PUNCTUATION
    else:
        raise ValueError("Unsupported language type: {}".format(language))

    chunks = []
    text_length = len(text)
    start_pos = 0

    while start_pos < text_length:
        # 计算当前块的结束位置
        end_pos = min(start_pos + chunk_size, text_length)

        if end_pos < text_length:
            # 从end_pos向前找最近的标点符号
            while end_pos > start_pos and text[end_pos - 1] not in split_chars:
                end_pos -= 1
            if end_pos == start_pos:
                # 如果没找到合适的标点，就硬切
                end_pos = start_pos + chunk_size

        # 获取当前块的内容并去除首尾空白字符
        chunk_content = text[start_pos:end_pos].strip()
        if chunk_content:
            # 添加当前块到结果列表
            chunks.append({
                'header': header if not chunks else '',
                'content': chunk_content,
                'level': level if not chunks else 0
            })

        # 更新start_pos，考虑重叠部分
        # 如果end_pos小于文本长度，则start_pos更新为end_pos减去chunk_overlap
        # 这样可以确保下一个块与当前块有重叠部分
        # 如果end_pos等于或大于文本长度，则start_pos更新为end_pos
        start_pos = end_pos - chunk_overlap if end_pos < text_length else end_pos

    return chunks
