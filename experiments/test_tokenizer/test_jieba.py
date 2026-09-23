import jieba  # 导入结巴分词库

# 测试句子
sentence1 = "你好，世界！"
sentence2 = "清华大学是中国顶尖的学府之一。"
sentence3 = "自然语言处理（NLP）是人工智能的重要分支。"

# 对句子进行分词（精确模式）
words1 = jieba.cut(sentence1, cut_all=False)
words2 = jieba.cut(sentence2, cut_all=False)
words3 = jieba.cut(sentence3, cut_all=False)

# 将生成器转换为列表以便打印
words1_list = list(words1)
words2_list = list(words2)
words3_list = list(words3)

print(f"Original: {sentence1}")
print(f"Tokenized: {' / '.join(words1_list)}")
print()

print(f"Original: {sentence2}")
print(f"Tokenized: {' / '.join(words2_list)}")
print()

print(f"Original: {sentence3}")
print(f"Tokenized: {' / '.join(words3_list)}")

# 全模式示例 (cut_all=True)
print("\nFull Mode:")
words1_full = jieba.cut(sentence1, cut_all=True)
words2_full = jieba.cut(sentence2, cut_all=True)
words3_full = jieba.cut(sentence3, cut_all=True)
print(f"Full mode (sentence1): {' / '.join(list(words1_full))}")
print(f"Full mode (sentence2): {' / '.join(list(words2_full))}")
print(f"Full mode (sentence3): {' / '.join(list(words3_full))}")

# 搜索引擎模式
print("\nSearch Engine Mode:")
words1_search = jieba.cut_for_search(sentence1)
words2_search = jieba.cut_for_search(sentence2)
words3_search = jieba.cut_for_search(sentence3)
print(f"Search engine mode (sentence1): {' / '.join(list(words1_search))}")
print(f"Search engine mode (sentence2): {' / '.join(list(words2_search))}")
print(f"Search engine mode (sentence3): {' / '.join(list(words3_search))}")

"""
Original: 你好，世界！
Tokenized: 你好 / ， / 世界 / ！

Original: 清华大学是中国顶尖的学府之一。
Tokenized: 清华大学 / 是 / 中国 / 顶尖 / 的 / 学府 / 之一 / 。

Original: 自然语言处理（NLP）是人工智能的重要分支。
Tokenized: 自然语言 / 处理 / （ / NLP / ） / 是 / 人工智能 / 的 / 重要 / 分支 / 。

Full Mode:
Full mode (sentence1): 你好 / ， / 世界 / ！
Full mode (sentence2): 清华 / 清华大学 / 华大 / 大学 / 是 / 中国 / 顶尖 / 的 / 学府 / 之一 / 。
Full mode (sentence3): 自然 / 自然语言 / 语言 / 处理 / （ / NLP / ） / 是 / 人工 / 人工智能 / 智能 / 的 / 重要 / 分支 / 。

Search Engine Mode:
Search engine mode (sentence1): 你好 / ， / 世界 / ！
Search engine mode (sentence2): 清华 / 华大 / 大学 / 清华大学 / 是 / 中国 / 顶尖 / 的 / 学府 / 之一 / 。
Search engine mode (sentence3): 自然 / 语言 / 自然语言 / 处理 / （ / NLP / ） / 是 / 人工 / 智能 / 人工智能 / 的 / 重要 / 分支 / 。
"""