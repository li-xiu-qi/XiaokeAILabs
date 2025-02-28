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
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    # 简单统计字母单词（不含数字和符号）
    english_words = len(re.findall(r'\b[a-zA-Z]+\b', text))

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
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))

    # 如果存在中文字符，按中文处理
    return 'zh' if chinese_chars > 0 else 'en'
