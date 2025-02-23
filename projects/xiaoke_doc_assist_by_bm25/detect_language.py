from typing import List, Dict
import re


def detect_language(text: str) -> str:
    """
    功能描述: 检测文本主要语言（中文或英文）

    参数:
        text (str): 输入文本

    返回值:
        str: 'zh' 表示中文占主导，'en' 表示英文占主导
    """
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    # 简单统计字母单词（不含数字和符号）
    english_words = len(re.findall(r'\b[a-zA-Z]+\b', text))

    # 如果中文字符数量明显多于英文单词，按中文处理
    return 'zh' if chinese_chars > english_words * 2 else 'en'
