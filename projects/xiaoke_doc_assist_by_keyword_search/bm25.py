"""
BM25检索算法实现

BM25公式:
score(Q, d) = Σ IDF(qi) * [tf(qi,d) * (k1 + 1)] / [tf(qi,d) + k1 * (1 - b + b * |d|/avgdl)]
其中：
- Q: 查询，包含词 qi
- d: 文档
- IDF(qi) = log((N - df(qi) + 0.5) / (df(qi) + 0.5) + 1)  # 逆文档频率
- tf(qi,d): 词 qi 在文档 d 中的词频
- |d|: 文档 d 的长度
- avgdl: 平均文档长度
- N: 文档总数
- df(qi): 包含词 qi 的文档数
- k1: 控制词频饱和度的参数(通常1.2-2.0)
- b: 控制文档长度归一化的参数(通常0.75)

原理：
BM25是一种基于概率的检索模型，改进自TF-IDF，主要特点：
1. 词频饱和：通过k1参数控制词频增加时的得分增长，使高频词的影响达到上限
2. 文档长度归一化：通过b参数调节文档长度的影响，长文档不会无限制占优
3. IDF加权：给予稀有词更高的权重，同时避免负值
4. 平滑处理：在IDF计算中加入0.5，避免除零和极端值

这个实现提供了一个基础的BM25检索功能，可用于文本相关性排序。
"""
from abc import ABC, abstractmethod
from typing import List, Dict
import math
import jieba

# 抽象基类
class AbstractBM25(ABC):
    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        """
        抽象基类，定义BM25的核心功能
        
        Args:
            corpus: 文档集合，每个元素是一个文档字符串
            k1: 控制词频饱和度的参数
            b: 控制文档长度归一化的参数
        """
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.doc_count = len(corpus)

        # 分词后的文档集合，由子类实现
        self.tokenized_corpus = self._tokenize_corpus()

        # 计算每个文档的长度（词数）
        self.doc_lengths = [len(tokens) for tokens in self.tokenized_corpus]

        # 计算平均文档长度
        self.avg_doc_length = sum(self.doc_lengths) / self.doc_count if self.doc_count > 0 else 0

        # 词频和文档频率
        self.df = {}  # 文档频率
        self.tf = []  # 词频矩阵
        self._build_index()

    @abstractmethod
    def _tokenize(self, text: str) -> List[str]:
        """抽象方法：对文本进行分词"""
        pass

    def _tokenize_corpus(self) -> List[List[str]]:
        """对整个文档集合进行分词"""
        return [self._tokenize(doc) for doc in self.corpus]



    def _score(self, query_tokens: List[str], doc_id: int) -> float:
        """
        计算查询与文档的BM25得分
        
        Args:
            query_tokens: 分词后的查询词列表
            doc_id: 文档ID
        Returns:
            BM25得分
        """
        score = 0.0
        doc_len = self.doc_lengths[doc_id]

        for term in query_tokens:
            if term not in self.df:
                continue

            # 计算IDF
            idf = math.log((self.doc_count - self.df[term] + 0.5) /
                          (self.df[term] + 0.5) + 1.0)

            # 获取当前文档中该词的词频
            term_freq = self.tf[doc_id].get(term, 0)

            # 计算BM25的TF部分
            tf_part = term_freq * (self.k1 + 1) / \
                     (term_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length))

            # 累加得分
            score += idf * tf_part

        return score

    def search(self, query: str, top_k: int = 5) -> List[tuple]:
        """
        执行搜索并返回排序后的结果
        
        Args:
            query: 查询字符串
            top_k: 返回前k个结果
        Returns:
            列表，每个元素是(文档ID, 得分)的元组
        """
        query_tokens = self._tokenize(query)
        scores = [(doc_id, self._score(query_tokens, doc_id))
                 for doc_id in range(self.doc_count)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    
    def _build_index(self):
        for doc_id, tokens in enumerate(self.tokenized_corpus):
            term_freq = {}
            for term in tokens:
                term_freq[term] = term_freq.get(term, 0) + 1
            self.tf.append(term_freq)
            for term in set(tokens):
                self.df[term] = self.df.get(term, 0) + 1

# 英文BM25实现
class EnglishBM25(AbstractBM25):
    def _tokenize(self, text: str) -> List[str]:
        """英文分词：按空格分割并转为小写"""
        return text.lower().split()



# 中文BM25实现
class ChineseBM25(AbstractBM25):
    def _tokenize(self, text: str) -> List[str]:
        """中文分词：使用jieba"""
        return list(jieba.cut(text))


