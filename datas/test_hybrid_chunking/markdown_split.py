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

import yaml # 导入 yaml 库
from typing import (Dict, List, Optional, Tuple, TypedDict)
from dataclasses import dataclass, field


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
            final_metadata = base_meta.copy() # 注意: 这里原本使用了 copy.deepcopy，但 base_meta 通常是浅层字典，copy() 应该足够
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

            # 处理代码块
            if stripped_line.startswith("```") and stripped_line.count("```") == 1:
                in_code_block = not in_code_block
                opening_fence = "```" if in_code_block else ""
            # 检查代码块结束标记
            elif in_code_block and stripped_line.startswith(opening_fence):
                 in_code_block = False
                 opening_fence = ""

            # 如果在代码块内，直接添加行内容并继续
            if in_code_block:
                current_content.append(line)
                continue

            found_header = False
            # 如果不在代码块内，检查是否为标题行
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
                                "metadata": current_metadata.copy(), # 使用 copy()
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
            # 如果不是标题行且不在代码块内
            if not found_header and not in_code_block:
                 if line.strip() or current_content: # 只有当行不为空或当前内容不为空时才添加
                    current_content.append(line)

        if current_content:
            lines_with_metadata.append({
                "content": "\n".join(current_content),
                "metadata": current_metadata.copy(), # 使用 copy()
            })

        return self._aggregate_lines_to_chunks(lines_with_metadata, base_meta=base_metadata)

# --- 主要执行 / 测试块 ---
if __name__ == '__main__':
    # 测试代码块 
    try: # 增加简单的错误处理
        # 假设 article.md 文件存在于脚本同目录下
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