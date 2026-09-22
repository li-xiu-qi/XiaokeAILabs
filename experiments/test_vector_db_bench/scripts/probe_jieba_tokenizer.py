"""jieba 分词器组成环节的实证探测。

读 jieba 0.42.1 源码可知分词由五步组成：字符切块 -> 前缀词典 -> DAG 构建 ->
最大概率路径 -> HMM 未登录词识别。本脚本把这些内部状态打印出来做实证：
  1. 词典规模（FREQ 表键数、词频总和、高频词）
  2. 单句的 DAG（每个起点的所有候选终点）
  3. 三种切分模式（精确 / 无 HMM / 全模式 / 搜索模式）的输出
  4. HMM 对未登录词的处理

用法：
    cd experiments/test_vector_db_bench
    ./.venv/Scripts/python.exe scripts/probe_jieba_tokenizer.py
"""

import jieba
import jieba.finalseg
import os
import time
import warnings

warnings.filterwarnings("ignore")


def main():
    # 词典规模
    t = time.time()
    jieba.initialize()
    load = time.time() - t
    freq = jieba.dt.FREQ
    dict_file = jieba.get_dict_file().name
    real_words = sum(1 for w, f in freq.items() if f > 0)
    print("=== 词典 ===")
    print("dict_file      :", dict_file, os.path.getsize(dict_file), "bytes")
    print("load_time      : %.3fs" % load)
    print("FREQ entries   : %d (含词频为 0 的前缀节点)" % len(freq))
    print("real words     : %d (词频 > 0)" % real_words)
    print("total freq     : %d" % jieba.dt.total)
    top = sorted(freq.items(), key=lambda x: -x[1])[:10]
    print("top words      :", top)

    # DAG 与三种模式
    s = "向量数据库的索引构建"
    print("\n=== 句子 %r 的 DAG ===" % s)
    dag = jieba.dt.get_DAG(s)
    print("DAG            :", dag)
    for i, ch in enumerate(s):
        ends = dag[i]
        cand = [s[i:e + 1] for e in ends]
        print("  位置 %d '%s' 候选: %s" % (i, ch, cand))

    print("\n=== 切分模式 ===")
    print("exact (HMM)    :", list(jieba.cut(s)))
    print("no HMM         :", list(jieba.cut(s, HMM=False)))
    print("cut_all        :", list(jieba.cut(s, cut_all=True)))
    print("cut_for_search :", list(jieba.cut_for_search(s)))
    print("tokenize search:", list(jieba.tokenize(s, mode="search")))

    # HMM 未登录词
    print("\n=== HMM 未登录词 ===")
    unk = "砼浇筑燊淼"
    print("unknown %r     :" % unk, list(jieba.cut(unk)))

    # 词典干预
    print("\n=== 词典干预 ===")
    print("suggest_freq('向量数据库'):", jieba.suggest_freq("向量数据库"))


if __name__ == "__main__":
    main()
