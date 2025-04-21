# This code is based on or derived from code obtained from:
# https://github.com/InternLM/HuixiangDou/blob/main/LICENSE
#
# Original License:
# BSD 3-Clause License
#
# Copyright (c) 2024, tpoisonooo
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import copy
import os
import pdb
import json
import re
import yaml # 导入 yaml 库
from abc import ABC, abstractmethod
from typing import (AbstractSet, Any, Callable, Collection, Dict, Iterable,
                    List, Literal, Optional, Sequence, Tuple, Type, TypedDict,
                    TypeVar, Union)
from dataclasses import dataclass, field
from loguru import logger


# --- 数据结构和类型定义 ---

@dataclass
class Chunk:
    """用于存储文本片段及相关元数据的类。"""
    content: str = ''
    metadata: dict = field(default_factory=dict)

    def __str__(self) -> str:
        """重写 __str__ 方法，使其仅包含 content 和 metadata。"""
        if self.metadata:
            return f"content='{self.content}' metadata={self.metadata}"
        else:
            return f"content='{self.content}'"

    def __repr__(self) -> str:
        return self.__str__()

    def to_markdown(self, return_all: bool = False) -> str:
        """将块转换为 Markdown 格式。

        Args:
            return_all: 如果为 True，则在内容前包含 YAML 格式的元数据。

        Returns:
            Markdown 格式的字符串。
        """
        md_string = ""
        if return_all and self.metadata:
            # 使用 yaml.dump 将元数据格式化为 YAML 字符串
            # allow_unicode=True 确保中文字符正确显示
            # sort_keys=False 保持原始顺序
            metadata_yaml = yaml.dump(self.metadata, allow_unicode=True, sort_keys=False)
            md_string += f"---\n{metadata_yaml}---\n\n"
        md_string += self.content
        return md_string

class LineType(TypedDict):
    """行类型，使用类型字典定义。"""
    metadata: Dict[str, str] # 元数据字典
    content: str # 行内容

class HeaderType(TypedDict):
    """标题类型，使用类型字典定义。"""
    level: int # 标题级别
    name: str # 标题名称 (例如, 'Header 1')
    data: str # 标题文本内容

# --- 文本分割逻辑 ---

def _split_text_with_regex(
        text: str, separator: str,
        keep_separator: Union[bool, Literal['start', 'end']]) -> List[str]:
    """使用正则表达式分隔符分割文本。"""
    if (separator):
        if keep_separator:
            _splits = re.split(f'({separator})', text)
            if not _splits:
                return []
            if keep_separator == 'end':
                splits = [''.join(pair) for pair in zip(_splits[::2], _splits[1::2])]
                if len(_splits) % 2 == 1:
                    splits.append(_splits[-1])
            elif keep_separator == 'start' or keep_separator is True:
                splits = [''.join(pair) for pair in zip(_splits[1::2], _splits[2::2])]
                if _splits[0]:
                    splits.insert(0, _splits[0])
            else:
                splits = re.split(separator, text)
        else:
            splits = re.split(separator, text)
    else:
        splits = list(text)
    return [s for s in splits if s]

class TextSplitter(ABC):
    """将文本分割成块的接口。"""

    def __init__(
        self,
        chunk_size: int = 832,
        chunk_overlap: int = 32,
        length_function: Callable[[str], int] = len,
        keep_separator: Union[bool, Literal['start', 'end']] = False,
        add_start_index: bool = False,
        strip_whitespace: bool = True,
    ) -> None:
        """创建一个新的 TextSplitter。"""
        if chunk_overlap < 0:
            raise ValueError("块重叠必须为非负数。")
        if chunk_overlap > chunk_size:
            raise ValueError(
                f'块重叠 ({chunk_overlap}) 大于块大小 ({chunk_size})，应更小。')
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._length_function = length_function
        self._keep_separator = keep_separator
        self._add_start_index = add_start_index
        self._strip_whitespace = strip_whitespace

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """将文本分割成多个组件。"""
        pass

    def create_chunks(self,
                      texts: List[str],
                      metadatas: Optional[List[dict]] = None) -> List[Chunk]:
        """从文本列表创建块。"""
        _metadatas = metadatas or [{}] * len(texts)
        if len(_metadatas) != len(texts):
            raise ValueError("文本和元数据的数量必须相同。")

        chunks = []
        for i, text in enumerate(texts):
            start_index = 0
            previous_chunk_len = 0
            for chunk_text in self.split_text(text):
                metadata = copy.deepcopy(_metadatas[i])
                if self._add_start_index:
                    search_start = max(0, start_index + previous_chunk_len - self._chunk_overlap)
                    try:
                        index = text.index(chunk_text, search_start)
                        metadata['start_index'] = index
                        start_index = index
                        previous_chunk_len = len(chunk_text)
                    except ValueError:
                        logger.warning(f"无法在原始文本中找到确切的块文本。起始索引元数据可能不准确。块: '{chunk_text[:100]}...'")
                new_chunk = Chunk(content=chunk_text, metadata=metadata)
                chunks.append(new_chunk)
        return chunks

    def _join_chunks(self, chunks: List[str], separator: str) -> Optional[str]:
        """使用分隔符连接文本块列表。"""
        text = separator.join(chunks)
        if self._strip_whitespace:
            text = text.strip()
        return text if text else None

    def _merge_splits(self, splits: Iterable[str], separator: str) -> List[str]:
        """将小的分割合并成小于块大小的块。"""
        separator_len = self._length_function(separator)
        chunks = []
        current_chunk_list: List[str] = []
        current_chunk_len = 0

        for split in splits:
            split_len = self._length_function(split)
            potential_len = current_chunk_len + split_len + (separator_len if current_chunk_list else 0)
            if potential_len > self._chunk_size and current_chunk_list:
                if current_chunk_len > self._chunk_size:
                    logger.warning(
                        f'创建了一个大小为 {current_chunk_len} 的块, '
                        f'它比指定的 {self._chunk_size} 更长'
                    )
                chunk = self._join_chunks(current_chunk_list, separator)
                if chunk is not None:
                    chunks.append(chunk)
                while current_chunk_list and (current_chunk_len > self._chunk_overlap or
                       (current_chunk_len + split_len + (separator_len if current_chunk_list else 0) > self._chunk_size and current_chunk_len > 0)):
                    len_to_remove = self._length_function(current_chunk_list[0]) + (separator_len if len(current_chunk_list) > 1 else 0)
                    current_chunk_len -= len_to_remove
                    current_chunk_list.pop(0)
            current_chunk_list.append(split)
            current_chunk_len += split_len + (separator_len if len(current_chunk_list) > 1 else 0)

        if current_chunk_list:
            chunk = self._join_chunks(current_chunk_list, separator)
            if chunk is not None:
                chunks.append(chunk)
        return chunks

class CharacterTextSplitter(TextSplitter):
    """按字符分割文本。"""

    def __init__(self,
                 separator: str = '\n\n', # 分隔符，默认为两个换行符
                 is_separator_regex: bool = False, # 分隔符是否为正则表达式
                 **kwargs: Any) -> None:
        """创建一个新的 CharacterTextSplitter。"""
        super().__init__(**kwargs)
        self._separator = separator
        self._is_separator_regex = is_separator_regex
        if self._is_separator_regex and self._keep_separator not in [False, 'start', 'end']:
             logger.warning(f"keep_separator='{self._keep_separator}' 在正则表达式分隔符下未明确支持。行为可能不符合预期。")


    def split_text(self, text: str) -> List[str]:
        """分割传入的文本并返回块。"""
        separator_to_use = (self._separator if self._is_separator_regex else
                            re.escape(self._separator))
        splits = _split_text_with_regex(text, separator_to_use, self._keep_separator)
        _separator_join = '' if self._keep_separator else self._separator
        return self._merge_splits(splits, _separator_join)

class RecursiveCharacterTextSplitter(TextSplitter):
    """通过递归地查看字符来分割文本。

    递归地尝试按不同的字符进行分割，以找到有效的分隔符。
    """

    def __init__(
        self,
        separators: Optional[List[str]] = None, # 分隔符列表
        keep_separator: bool = True, # 注意: 原始 langchain 默认为 True
        is_separator_regex: bool = False, # 注意: 原始 langchain 默认为 False
        **kwargs: Any,
    ) -> None:
        """创建一个新的 RecursiveCharacterTextSplitter。"""
        super().__init__(keep_separator=keep_separator, **kwargs)
        self._separators = separators or ['\n\n', '\n', ' ', '']
        self._is_separator_regex = is_separator_regex
        if self._is_separator_regex and self._keep_separator not in [False, 'start', 'end']:
             logger.warning(f"keep_separator='{self._keep_separator}' 在正则表达式分隔符下未明确支持。行为可能不符合预期。")


    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """递归地分割传入的文本并返回块。"""
        final_chunks = []
        separator = separators[-1]
        next_separators = []
        for i, sep in enumerate(separators):
            effective_sep = sep if self._is_separator_regex else re.escape(sep)
            if sep == '' or re.search(effective_sep, text):
                separator = sep
                next_separators = separators[i + 1:]
                break
        effective_separator_regex = separator if self._is_separator_regex else re.escape(separator)
        splits = _split_text_with_regex(text, effective_separator_regex, self._keep_separator)
        _separator_join = '' if self._keep_separator else separator
        good_splits: List[str] = []
        for s in splits:
            if self._length_function(s) < self._chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    merged = self._merge_splits(good_splits, _separator_join)
                    final_chunks.extend(merged)
                    good_splits = []
                if next_separators:
                    recursive_chunks = self._split_text(s, next_separators)
                    final_chunks.extend(recursive_chunks)
                else:
                    final_chunks.append(s)
        if good_splits:
            merged = self._merge_splits(good_splits, _separator_join)
            final_chunks.extend(merged)
        return final_chunks

    def split_text(self, text: str) -> List[str]:
        """使用定义的分隔符递归地分割文本。"""
        return self._split_text(text, self._separators)

class ChineseRecursiveTextSplitter(RecursiveCharacterTextSplitter):
    """适用于中文文本的递归文本分割器。

    使用适合中文标点和句子结构的正则表达式分隔符。
    修改自 https://github.com/chatchat-space/Langchain-Chatchat/blob/master/text_splitter/chinese_recursive_text_splitter.py
    """
    def __init__(
        self,
        separators: Optional[List[str]] = None,
        keep_separator: bool = True,
        is_separator_regex: bool = True, # 对中文分隔符默认为 True
        **kwargs: Any,
    ) -> None:
        """创建一个新的 ChineseRecursiveTextSplitter。"""
        _separators = separators or [
            "\n\n",  # 段落
            "\n",    # 行
            "。|！|？",  # 中文句末标点
            "\.\s|\!\s|\?\s", # 英文句末标点加空格
            "；|;\s",  # 分号
            "，|,\s"   # 逗号
        ]
        super().__init__(separators=_separators, keep_separator=keep_separator, is_separator_regex=is_separator_regex, **kwargs)

class MarkdownTextRefSplitter(RecursiveCharacterTextSplitter):
    """尝试沿 Markdown 格式的标题和其他元素分割文本。"""

    def __init__(self, **kwargs: Any) -> None:
        """初始化一个 MarkdownTextRefSplitter。"""
        separators = [
            # 按标题分割 (任何级别)
            "\n#{1,6} ",
            # 代码块围栏
            "\n```\n",
            # 水平线
            "\n\\*\\*\\*+\n",
            "\n---+\n",
            "\n___+\n",
            # 段落和行
            "\n\n",
            "\n",
            " ",
            "",
        ]
        super().__init__(separators=separators, is_separator_regex=True, **kwargs)

class MarkdownHeaderTextSplitter:
    """基于指定的标题分割 Markdown 文件。"""

    def __init__(
        self,
        headers_to_split_on: List[Tuple[str, str]] = [
            ("#", "h1"),  # 使用 h1, h2 等作为键名
            ("##", "h2"),
            ("###", "h3"),
            ("####", "h4"),
            ("#####", "h5"),
            ("######", "h6"),
        ],
        strip_headers: bool = False, # 默认更改为 False 以在内容中保留标题
    ):
        """创建一个新的 MarkdownHeaderTextSplitter。"""
        self.headers_to_split_on = sorted(
            headers_to_split_on, key=lambda split: len(split[0]), reverse=True
        )
        self.strip_headers = strip_headers

    def _aggregate_lines_to_chunks(self, lines: List[LineType],
                                   base_meta: dict) -> List[Chunk]:
        """将具有共同元数据的行合并成块。"""
        aggregated_chunks: List[LineType] = []

        for line in lines:
            if aggregated_chunks and aggregated_chunks[-1]["metadata"] == line["metadata"]:
                aggregated_chunks[-1]["content"] += "\n" + line["content"]
            else:
                aggregated_chunks.append(line)

        final_chunks = []
        for chunk_data in aggregated_chunks:
            final_metadata = base_meta.copy()
            final_metadata.update(chunk_data['metadata'])
            final_chunks.append(
                Chunk(content=chunk_data["content"].strip(),
                      metadata=final_metadata)
            )
        return final_chunks

    def split_text(self, text: str, metadata: Optional[dict] = None) -> List[Chunk]:
        """基于标题分割 Markdown 文本。"""
        base_metadata = metadata or {}
        lines = text.split("\n")
        lines_with_metadata: List[LineType] = []
        current_content: List[str] = []
        current_metadata: Dict[str, str] = {}
        header_stack: List[HeaderType] = []

        in_code_block = False
        opening_fence = ""

        for line_num, line in enumerate(lines):
            stripped_line = line.strip()

            if stripped_line.startswith("```") and stripped_line.count("```") == 1:
                in_code_block = not in_code_block
                opening_fence = "```" if in_code_block else ""
            elif stripped_line.startswith("~~~") and stripped_line.count("~~~") == 1:
                 in_code_block = not in_code_block
                 opening_fence = "~~~" if in_code_block else ""
            elif in_code_block and stripped_line.startswith(opening_fence):
                 in_code_block = False
                 opening_fence = ""

            if in_code_block:
                current_content.append(line)
                continue

            found_header = False
            if not in_code_block:
                for sep, name in self.headers_to_split_on:
                    if stripped_line.startswith(sep) and (
                        len(stripped_line) == len(sep) or stripped_line[len(sep)] == " "
                    ):
                        found_header = True
                        header_level = sep.count("#")
                        header_data = stripped_line[len(sep):].strip()

                        if current_content:
                            lines_with_metadata.append({
                                "content": "\n".join(current_content),
                                "metadata": current_metadata.copy(),
                            })
                            current_content = []

                        while header_stack and header_stack[-1]["level"] >= header_level:
                            header_stack.pop()

                        new_header: HeaderType = {"level": header_level, "name": name, "data": header_data}
                        header_stack.append(new_header)

                        # 使用 name (现在是 h1, h2 等) 作为键
                        current_metadata = {h["name"]: h["data"] for h in header_stack}

                        if not self.strip_headers:
                            current_content.append(line)

                        break

            if not found_header and not in_code_block:
                 if line.strip() or current_content:
                    current_content.append(line)

        if current_content:
            lines_with_metadata.append({
                "content": "\n".join(current_content),
                "metadata": current_metadata.copy(),
            })

        return self._aggregate_lines_to_chunks(lines_with_metadata, base_meta=base_metadata)

    def create_chunks(self, text: str, metadata: Optional[dict] = None) -> List[Chunk]:
         """split_text 的别名，用于向后兼容。"""
         logger.warning("MarkdownHeaderTextSplitter 的 `create_chunks` 已弃用，请改用 `split_text`。")
         return self.split_text(text, metadata)

def clean_md(text: str) -> str:
    """移除或简化 Markdown 文本的某些部分，如代码块、链接等。"""
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'`.*?`', '', text)
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'^\s*[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    text = text.lower()
    text = text.strip()
    return text

# --- 主要执行 / 测试块 ---
if __name__ == '__main__':
    # 测试代码块 (保持原文的测试逻辑)
    try: # 增加简单的错误处理
        with open("article.md", "r", encoding="utf-8") as f:
            text = f.read()
        # 现在 splitter 将使用 h1, h2 等作为元数据键
        splitter = MarkdownHeaderTextSplitter() # 保持原文默认初始化
        chunks = splitter.split_text(text)
        for chunk in chunks:
            print("--- Chunk ---")
            print(f"Length: {len(chunk.content)}")
            print(f"Metadata: {chunk.metadata}")
            print("\n--- Markdown (Content Only) ---")
            print(chunk.to_markdown())
            print("\n--- Markdown (With Metadata) ---")
            print(chunk.to_markdown(return_all=True))
            print("====" * 40)
    except FileNotFoundError:
        print("Error: article.md not found. Please create the file for testing.")
    except Exception as e:
        print(f"An error occurred during testing: {e}")