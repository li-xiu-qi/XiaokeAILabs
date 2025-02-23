from typing import List

"""
# KMP算法原理说明：
# KMP（Knuth-Morris-Pratt）算法是一种高效的字符串匹配算法，用于在主字符串（text）中查找模式串（pattern）
# 核心思想：
# 1. 利用已匹配部分的信息，避免不必要的回溯
# 2. 通过预处理模式串，构建最长前后缀数组（LPS - Longest Proper Prefix which is also Suffix）
# 3. 当发生失配时，根据LPS数组进行快速跳转，而不是简单地回退
# 算法优势：
# - 时间复杂度：O(n + m)，其中n是文本长度，m是模式串长度
# - 空间复杂度：O(m)，用于存储LPS数组
# - 相比朴素算法O(n*m)，避免了大量重复比较

"""

class KMPSearch:
    """
    KMP字符串匹配算法类
    用于高效地查找文本中的关键词
    """

    def __init__(self):
        """初始化方法"""
        self._pattern: str = ""  # 模式串（关键词）
        self._text: str = ""  # 目标文本
        self._lps: List[int] = []  # 最长前后缀数组

    def _compute_lps_array(self) -> None:
        """
        计算最长前后缀数组（LPS）
        用于KMP算法中的快速跳转
        """
        length: int = 0  # 当前最长前后缀的长度
        self._lps = [0] * len(self._pattern)  # 初始化LPS数组
        i: int = 1  # 从第二个字符开始

        # 遍历模式串计算LPS数组
        while i < len(self._pattern):
            if self._pattern[i] == self._pattern[length]:
                length += 1
                self._lps[i] = length
                i += 1
            else:
                if length != 0:
                    # 回退到上一个可能的前缀
                    length = self._lps[length - 1]
                else:
                    self._lps[i] = 0
                    i += 1

    def search(self, text: str, pattern: str) -> List[int]:
        """
        在文本中搜索关键词

        Args:
            text (str): 目标文本
            pattern (str): 要搜索的关键词

        Returns:
            List[int]: 所有匹配位置的起始索引列表
        """
        # 参数验证
        if not text or not pattern:
            return []

        # 初始化类变量
        self._text = text
        self._pattern = pattern

        # 计算LPS数组
        self._compute_lps_array()

        matches: List[int] = []  # 存储所有匹配位置
        i: int = 0  # 文本索引
        j: int = 0  # 模式串索引

        # 主搜索循环
        while i < len(self._text):
            if self._pattern[j] == self._text[i]:
                i += 1
                j += 1

            if j == len(self._pattern):
                # 找到一个匹配
                matches.append(i - j)
                j = self._lps[j - 1]
            elif i < len(self._text) and self._pattern[j] != self._text[i]:
                if j != 0:
                    j = self._lps[j - 1]
                else:
                    i += 1

        return matches