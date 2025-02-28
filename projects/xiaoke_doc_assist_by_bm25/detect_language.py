#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：筱可
# 2024年05月06日
"""
#### 使用说明：
1.  导入 `split_text_detect_language` 或 `tokenizer_detect_language` 函数。
2.  调用函数，传入待检测的文本字符串。
3.  根据返回值判断文本的主要语言。

#### 主要功能：
1.  `split_text_detect_language`: 通过比较中文和英文单词的数量来检测文本的主要语言。
2.  `tokenizer_detect_language`: 通过检测是否存在中文字符来判断文本是否包含中文。

#### 参数说明：

**`split_text_detect_language(text: str) -> str`**
*   `text` (str):  需要检测的文本。
*   返回值 (str):  `'zh'` 表示中文占主导，`'en'` 表示英文占主导。

**`tokenizer_detect_language(text: str) -> str`**
*   `text` (str):  需要检测的文本。
*   返回值 (str):  `'zh'` 表示存在中文字符，`'en'` 表示不存在中文字符。

#### 注意事项：
*   `split_text_detect_language` 依赖于正则表达式。
*   `tokenizer_detect_language` 适用于需要快速判断文本是否包含中文的场景。
"""
from typing import List, Dict
import re


def split_text_detect_language(text: str) -> str:
    """
    功能描述: 检测文本主要语言（中文或英文）

    参数:
        text (str): 输入文本

    返回值:
        str: 'zh' 表示中文占主导，'en' 表示英文占主导
    """
    chinese_chars: int = len(re.findall(r'[\u4e00-\u9fff]', text))
    # 简单统计字母单词（不含数字和符号）
    english_words: int = len(re.findall(r'\b[a-zA-Z]+\b', text))

    # 如果中文字符数量明显多于英文单词，按中文处理
    return 'zh' if chinese_chars > english_words * 2 else 'en'


def tokenizer_detect_language(text: str) -> str:
    """
    功能描述: 检测文本主要语言（中文或英文），
    因为jieba分词能兼容分割英文（空格），
    所以这里只检测中文字符存在即可

    参数:
        text (str): 输入文本

    返回值:
        str: 'zh' 表示存在中文字符，'en' 表示不存在中文字符
    """
    chinese_chars: int = len(re.findall(r'[\u4e00-\u9fff]', text))

    # 如果存在中文字符，按中文处理
    return 'zh' if chinese_chars > 0 else 'en'
