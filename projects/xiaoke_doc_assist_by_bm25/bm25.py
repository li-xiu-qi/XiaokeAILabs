from abc import ABC, abstractmethod
from typing import List, Tuple, Dict
import math
import jieba
import Stemmer  # PyStemmer 库，用于英文词干提取
import re  # 用于英文文本预处理
import json  # 用于保存和加载 JSON 格式
import pickle  # 用于保存和加载 Pickle 格式
from stopwords import (
    STOPWORDS_EN_PLUS,
    STOPWORDS_CHINESE,
)
from detect_language import tokenizer_detect_language


# 抽象基类
class AbstractBM25(ABC):
    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75, stopwords: tuple = ()):
        """
        抽象基类，定义BM25的核心功能

        Args:
            corpus: 文档集合，每个元素是一个文档字符串
            k1: 控制词频饱和度的参数
            b: 控制文档长度归一化的参数
            stopwords: 停用词元组
        Raises:
            ValueError: 如果corpus为空
        """
        if not corpus:
            raise ValueError("Corpus cannot be empty")
        self.corpus: List[str] = corpus
        self.k1: float = k1
        self.b: float = b
        self.stopwords: set = set(stopwords)  # 转换为set以提高查找效率
        self.doc_count: int = len(corpus)

        # 分词后的文档集合，由子类实现
        self.tokenized_corpus: List[List[str]] = self._tokenize_corpus()

        # 计算每个文档的长度（词数）
        self.doc_lengths: List[int] = [len(tokens) for tokens in self.tokenized_corpus]

        # 计算平均文档长度
        self.avg_doc_length: float = sum(self.doc_lengths) / self.doc_count if self.doc_count > 0 else 0

        # 词频和文档频率
        self.df: Dict[str, int] = {}  # 文档频率
        self.tf: List[Dict[str, int]] = []  # 词频矩阵
        self._build_index()

    @abstractmethod
    def _tokenize(self, text: str) -> List[str]:
        """抽象方法：对文本进行分词"""
        pass

    def _tokenize_corpus(self) -> List[List[str]]:
        """对整个文档集合进行分词"""
        return [self._tokenize(doc) for doc in self.corpus]

    def _build_index(self):
        """构建词频和文档频率索引"""
        for doc_id, tokens in enumerate(self.tokenized_corpus):
            term_freq = {}
            for term in tokens:
                term_freq[term] = term_freq.get(term, 0) + 1
            self.tf.append(term_freq)
            for term in set(tokens):
                self.df[term] = self.df.get(term, 0) + 1

    def _score(self, query_tokens: List[str], doc_id: int) -> float:
        """
        计算查询与文档的BM25得分
        """
        score = 0.0
        doc_len = self.doc_lengths[doc_id]

        for term in query_tokens:
            if term not in self.df:
                continue

            idf = math.log((self.doc_count - self.df[term] + 0.5) /
                           (self.df[term] + 0.5) + 1.0)

            term_freq = self.tf[doc_id].get(term, 0)
            tf_part = term_freq * (self.k1 + 1) / \
                      (term_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length))

            score += idf * tf_part

        return score

    def search(self, query: str, top_k: int = 5) -> List[tuple]:
        """
        执行搜索并返回排序后的结果
        """
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        query_tokens = self._tokenize(query)
        scores = [(doc_id, self._score(query_tokens, doc_id))
                  for doc_id in range(self.doc_count)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def save(self, filepath: str):
        """
        将BM25索引保存到文件（支持 JSON 和 Pickle 格式）

        Args:
            filepath: 保存文件的路径（.json 或 .pkl）
        Raises:
            ValueError: 如果文件扩展名不支持
        """
        data = {
            'df': self.df,
            'tf': self.tf,
            'k1': self.k1,
            'b': self.b,
            'language': 'english' if isinstance(self, EnglishBM25) else 'chinese',
            'stopwords': list(self.stopwords)
        }
        if filepath.endswith('.json'):
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        elif filepath.endswith('.pkl'):
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
        else:
            raise ValueError("Unsupported file extension. Use .json or .pkl.")

    @classmethod
    def load(cls, filepath: str, corpus: List[str]):
        """
        从文件加载BM25索引（支持 JSON 和 Pickle 格式）

        Args:
            filepath: 索引文件的路径（.json 或 .pkl）
            corpus: 原始文档集合，用于初始化
        Returns:
            EnglishBM25 或 ChineseBM25 实例
        Raises:
            ValueError: 如果文件扩展名或语言不支持
        """
        if filepath.endswith('.json'):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif filepath.endswith('.pkl'):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
        else:
            raise ValueError("Unsupported file extension. Use .json or .pkl.")

        language = data['language']
        if language == 'english':
            bm25_cls = EnglishBM25
        elif language == 'chinese':
            bm25_cls = ChineseBM25
        else:
            raise ValueError("Unsupported language in saved data.")

        stopwords = tuple(data['stopwords'])
        bm25 = bm25_cls(corpus, data['k1'], data['b'], stopwords)
        bm25.df = data['df']
        bm25.tf = data['tf']
        bm25.doc_lengths = [sum(tf_doc.values()) for tf_doc in bm25.tf]
        bm25.avg_doc_length = sum(bm25.doc_lengths) / len(bm25.doc_lengths) if bm25.doc_lengths else 0
        return bm25


# 英文BM25实现（使用 PyStemmer 和停用词）
class EnglishBM25(AbstractBM25):
    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75, stopwords: tuple = STOPWORDS_EN_PLUS):
        """
        英文BM25实现，使用PyStemmer进行词干提取和停用词过滤
        """
        self.stemmer = Stemmer.Stemmer('english')  # 初始化英文词干提取器
        super().__init__(corpus, k1, b, stopwords)

    def _tokenize(self, text: str) -> List[str]:
        """英文分词：使用正则表达式预处理 + PyStemmer + 停用词过滤"""
        text = text.lower()
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)
        tokens = text.split()
        return [self.stemmer.stemWord(token) for token in tokens if token and token not in self.stopwords]


# 中文BM25实现
class ChineseBM25(AbstractBM25):
    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75, stopwords: tuple = STOPWORDS_CHINESE):
        """
        中文BM25实现，使用jieba分词和停用词过滤
        """
        super().__init__(corpus, k1, b, stopwords)

    def _tokenize(self, text: str) -> List[str]:
        """中文分词：使用jieba并过滤停用词"""
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z]', '', text)
        tokens = jieba.cut(text)
        return [token for token in tokens if token and token not in self.stopwords]


class MixedLanguageBM25(AbstractBM25):
    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75, stopwords_en: tuple = STOPWORDS_EN_PLUS,
                 stopwords_cn: tuple = STOPWORDS_CHINESE):
        """
        混合语言BM25实现，自动检测语言并使用相应的分词器和停用词过滤
        """
        self.english_bm25 = EnglishBM25(corpus, k1, b, stopwords_en)
        self.chinese_bm25 = ChineseBM25(corpus, k1, b, stopwords_cn)
        super().__init__(corpus, k1, b, stopwords_en + stopwords_cn)

    def _tokenize(self, text: str) -> List[str]:
        """根据检测到的语言选择相应的分词器"""
        language = tokenizer_detect_language(text)
        if language == 'en':
            return self.english_bm25._tokenize(text)
        else:
            return self.chinese_bm25._tokenize(text)


# 工厂函数
def create_bm25(corpus: List[str],
                language: str = 'mixed',
                k1: float = 1.5,
                b: float = 0.75,
                stopwords: tuple = None):
    """
    创建BM25实例的工厂函数

    Args:
        corpus: 文档集合
        language: 语言类型 ('english' 或 'chinese')
        k1: 控制词频饱和度的参数
        b: 控制文档长度归一化的参数
        stopwords: 自定义停用词元组（可选）
    """
    language = language.lower()
    if language in ['english', 'en']:
        stopwords = stopwords if stopwords is not None else STOPWORDS_EN_PLUS
        return EnglishBM25(corpus, k1, b, stopwords)
    elif language in ['chinese', 'cn']:
        stopwords = stopwords if stopwords is not None else STOPWORDS_CHINESE
        return ChineseBM25(corpus, k1, b, stopwords)
    elif language in ['mixed']:
        stopwords_en = stopwords if stopwords is not None else STOPWORDS_EN_PLUS
        stopwords_cn = stopwords if stopwords is not None else STOPWORDS_CHINESE
        return MixedLanguageBM25(corpus, k1, b, stopwords_en, stopwords_cn)
    else:
        raise ValueError("Unsupported language. Please choose 'english/en', 'chinese/cn', or 'mixed'.")


def load_bm25(filepath: str, corpus: List[str]):
    """
    从文件加载BM25实例

    Args:
        filepath: 索引文件的路径（.json 或 .pkl）
        corpus: 原始文档集合，用于初始化
    Returns:
        BM25实例
    """
    return AbstractBM25.load(filepath, corpus)


# 通用的搜索函数
def bm25_search(corpus: List[str], query: str,
                language: str = 'mixed', top_k: int = 5,
                k1: float = 1.5,
                b: float = 0.75,
                stopwords: tuple = None) -> List[Tuple[int, float, str]]:
    """
    执行BM25搜索
    """
    bm25 = create_bm25(corpus, language, k1, b, stopwords)
    results = bm25.search(query, top_k)
    return [(doc_id, score, corpus[doc_id]) for doc_id, score in results]
