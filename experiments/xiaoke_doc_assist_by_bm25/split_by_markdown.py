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
    - metadata (Dict[str, any]): 文档元数据，默认为None。
    - 返回值：一个字典列表，包含标题、内容、标题级别和元数据。

3. merge_markdown_chunks 函数：
    - chunks (List[Dict[str, any]]): 已分块的Markdown文本。
    - chunk_size (int): 每个块的最大长度，默认为1000。
    - chunk_overlap (int): 块之间的重叠长度，默认为200。
    - separator (str): 块之间的分隔符，默认为"\n\n"。
    - language (str): 语言选项，默认为'auto'，自动检测语言，'zh' 表示中文，'en' 表示英文。
    - metadata (Dict[str, any]): 文档元数据，默认为None。
    - 返回值：合并后的Markdown分块，每个分块包含索引信息。

#### 注意事项：
1. 确保传入的文本符合Markdown格式。
2. 对于合并后的文本，可能需要手动调整块之间的分隔符和内容。
3. `chunk_size` 和 `chunk_overlap` 需要合理设置，以避免产生过小或过大的文本块。
4. 每个分块现在包含元数据和索引信息，可用于追踪和引用。

#### 更多信息：
无
"""

from typing import List, Dict, Optional, Any, Tuple

from detect_language import split_text_detect_language


def split_markdown_by_headers(
        text: str,
        min_level: int = 1,
        max_level: int = 6,
        metadata: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    功能描述: 将Markdown文本按标题级别分块。

    参数:
        text (str): 输入的Markdown文本。
        min_level (int): 处理的最小标题级别，默认为1。
        max_level (int): 处理的最大标题级别，默认为6。
        metadata (Dict[str, any]): 文档元数据，默认为None。

    返回值:
        List[Dict[str, any]]: 一个字典列表，包含标题、内容、标题级别和元数据。
    """
    # 参数验证
    if not isinstance(text, str):
        raise TypeError("text参数必须是字符串")
    if not isinstance(min_level, int) or not isinstance(max_level, int):
        raise TypeError("min_level和max_level必须是整数")
    if not text.strip():
        return []
        
    # 确保元数据是字典类型
    metadata = metadata or {}
    
    # 确保标题级别在1到6之间
    min_header_level = max(1, min(6, min_level))
    max_header_level = max(min_header_level, min(6, max_level))

    # 将文本按行分割
    lines = text.split('\n')
    chunks = []
    current_chunk = {'header': '', 'content': [], 'level': 0}

    # 遍历每一行
    for line in lines:
        # 检查当前行是否为标题
        header_match = False
        for level in range(min_header_level, max_header_level + 1):
            header_prefix = '#' * level + ' '
            if line.strip().startswith(header_prefix):
                # 如果当前块有内容或标题，将其添加到块列表中
                if current_chunk['content'] or current_chunk['header']:
                    chunks.append({
                        'header': current_chunk['header'],
                        'content': '\n'.join(current_chunk['content']).strip(),
                        'level': current_chunk['level'],
                        'metadata': metadata
                    })
                # 初始化新的块
                current_chunk = {
                    'header': line.strip()[level + 1:].strip(),
                    'content': [],
                    'level': level
                }
                header_match = True
                break
        
        # 如果当前行不是标题，将其添加到当前块的内容中
        if not header_match:
            current_chunk['content'].append(line)

    # 将最后一个块添加到块列表中
    if current_chunk['content'] or current_chunk['header']:
        chunks.append({
            'header': current_chunk['header'],
            'content': '\n'.join(current_chunk['content']).strip(),
            'level': current_chunk['level'],
            'metadata': metadata
        })
    return chunks


def merge_markdown_chunks(
        chunks: List[Dict[str, Any]],
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separator: str = "\n\n",
        language: str = 'auto',
        metadata: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    功能描述: 合并Markdown分块，按固定长度分割，支持中英文标点。

    参数:
        chunks (List[Dict[str, any]]): 已分块的Markdown文本。
        chunk_size (int): 每个块的最大长度，默认为1000。
        chunk_overlap (int): 块之间的重叠长度，默认为200。
        separator (str): 块之间的分隔符，默认为"\n\n"。
        language (str): 语言选项，默认为'auto'，自动检测语言，'zh' 表示中文，'en' 表示英文。
        metadata (Dict[str, any]): 文档元数据，默认为None。

    返回值:
        List[Dict[str, any]]: 合并后的Markdown分块，每个分块包含索引信息。
    """
    # 参数验证
    if not isinstance(chunks, list):
        raise TypeError("chunks参数必须是列表")
    if not isinstance(chunk_size, int) or not isinstance(chunk_overlap, int):
        raise TypeError("chunk_size和chunk_overlap必须是整数")
    if chunk_size <= chunk_overlap:
        raise ValueError("chunk_size必须大于chunk_overlap")
    if chunk_size <= 0 or chunk_overlap < 0:
        raise ValueError("chunk_size必须大于0且chunk_overlap不能为负数")
    
    # 确保元数据是字典类型
    metadata = metadata or {}
    # 使用chunks中的元数据，如果不存在则使用传入的元数据
    if chunks and isinstance(chunks[0], dict) and 'metadata' in chunks[0]:
        metadata.update(chunks[0].get('metadata', {}))

    # 初始化合并后的块列表和当前文本块信息
    merged_chunks = []
    current_text = ""
    current_level = 0
    current_header = ""
    next_chunk_index = 0

    # 处理所有分块
    for chunk in chunks:
        # 获取当前分块的文本内容，如果有标题则包含标题
        chunk_text = f"{chunk['header']}\n{chunk['content']}" if chunk['header'] else chunk['content']

        # 如果当前分块有标题且当前文本块不为空，处理当前已累积的文本
        if chunk['level'] > 0 and current_text.strip():
            # 将当前文本块分割并添加到结果中
            split_chunks = _split_text_by_size(
                current_text.strip(), 
                chunk_size, 
                chunk_overlap,
                current_header, 
                current_level, 
                language, 
                metadata, 
                next_chunk_index
            )
            merged_chunks.extend(split_chunks)
            next_chunk_index += len(split_chunks)
            
            # 重置当前文本块
            current_text = ""
            current_header = chunk['header']
            current_level = chunk['level']
        elif not current_text:
            # 初始块或前一块已处理完成
            current_header = chunk['header']
            current_level = chunk['level']
        
        # 添加当前分块的文本内容
        if current_text.strip():
            current_text += separator + chunk_text
        else:
            current_text = chunk_text

    # 处理最后一个累积的文本块
    if current_text.strip():
        merged_chunks.extend(_split_text_by_size(
            current_text.strip(),
            chunk_size,
            chunk_overlap,
            current_header,
            current_level,
            language,
            metadata,
            next_chunk_index
        ))

    return merged_chunks


def _split_text_by_size(
        text: str,
        chunk_size: int,
        chunk_overlap: int,
        header: str,
        level: int,
        language: str = 'auto',
        metadata: Optional[Dict[str, Any]] = None,
        start_index: int = 0
) -> List[Dict[str, Any]]:
    """
    功能描述: 内部函数：按固定长度分割文本，支持中英文智能分割

    参数:
        text (str): 要分割的文本。
        chunk_size (int): 每个分块的最大长度。
        chunk_overlap (int): 分块之间的重叠部分。
        header (str): 当前分块的标题。
        level (int): 当前分块的标题级别。
        language (str): 语言选项，默认为'auto'，可以手动指定'zh'或'en'。
        metadata (Dict[str, any]): 文档元数据，默认为None。
        start_index (int): 块索引的起始值，默认为0。

    返回值:
        List[Dict[str, any]]: 按要求分割后的文本块，带有索引和元数据。
    """
    # 确保元数据是字典类型
    metadata = metadata or {}
    
    # 定义中英文分割标点
    PUNCTUATION = {
        'en': ' \n\t.,!?;:',
        'zh': ' \n\t。！？，、；：'
    }

    # 确定使用哪种语言的分割规则
    lang = language.lower()
    if lang == 'auto':
        lang = split_text_detect_language(text)
    
    # 使用对应语言的标点符号，默认使用英文
    split_chars = PUNCTUATION.get(lang, PUNCTUATION['en'])

    # 分块处理
    chunks = []
    text_length = len(text)
    start_pos = 0
    chunk_index = start_index

    while start_pos < text_length:
        # 计算当前块的结束位置
        end_pos = min(start_pos + chunk_size, text_length)

        # 智能查找分割点
        if end_pos < text_length:
            # 寻找合适的分割点
            best_end_pos = _find_best_split_point(text, start_pos, end_pos, split_chars)
            if best_end_pos > start_pos:  # 找到合适的分割点
                end_pos = best_end_pos
            # 如果没找到合适的标点，就使用原始的end_pos

        # 获取当前块的内容并去除首尾空白字符
        chunk_content = text[start_pos:end_pos].strip()
        if chunk_content:
            # 只有第一个块保留标题和级别信息
            current_header = header if chunk_index == start_index else ''
            current_level = level if chunk_index == start_index else 0
            
            # 添加当前块到结果列表，包含索引和元数据
            chunks.append({
                'header': current_header,
                'content': chunk_content,
                'level': current_level,
                'metadata': metadata,
                'chunk_index': chunk_index
            })
            chunk_index += 1

        # 更新start_pos，考虑重叠部分
        start_pos = end_pos - chunk_overlap if end_pos < text_length else end_pos

    return chunks


def _find_best_split_point(text: str, start_pos: int, end_pos: int, split_chars: str) -> int:
    """查找最佳分割点，尽量在标点符号处分割"""
    # 从end_pos向前查找最近的分割字符
    pos = end_pos
    while pos > start_pos and text[pos - 1] not in split_chars:
        pos -= 1
        
    # 如果找不到合适的分割点，就返回原始end_pos
    return pos if pos > start_pos else end_pos
