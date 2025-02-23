import os
import sys

def append_src2syspath():
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)
    while os.path.basename(current_dir) == "projects":
        current_dir = os.path.dirname(current_dir)
    sys.path.append(current_dir)

# append_src2syspath()

from projects.xiaoke_doc_assist_by_keyword_search.bm25 import EnglishBM25, ChineseBM25


# 使用示例
if __name__ == "__main__":
    # 英文测试
    english_corpus = [
        "this is a sample document about machine learning",
        "machine learning is fascinating and useful",
        "this document discusses deep learning techniques",
        "another sample about artificial intelligence"
    ]
    english_bm25 = EnglishBM25(english_corpus)
    english_query = "machine learning"
    english_results = english_bm25.search(english_query, top_k=3)

    print("英文查询:", english_query)
    for doc_id, score in english_results:
        print(f"文档ID: {doc_id}, 得分: {score:.4f}, 文本: {english_corpus[doc_id]}")

    print("\n")

    # 中文测试
    chinese_corpus = [
        "这是一个关于机器学习的样本文档",
        "机器学习既迷人又实用",
        "本文档讨论深度学习技术",
        "另一个关于人工智能的样本"
    ]
    chinese_bm25 = ChineseBM25(chinese_corpus)
    chinese_query = "这里是机器的学习"
    chinese_results = chinese_bm25.search(chinese_query, top_k=3)

    print("中文查询:", chinese_query)
    for doc_id, score in chinese_results:
        print(f"文档ID: {doc_id}, 得分: {score:.4f}, 文本: {chinese_corpus[doc_id]}")
        
