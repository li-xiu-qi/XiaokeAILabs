# 使用Sentence Transformers进行评估

在本教程中，我们将介绍如何使用Sentence Transformers库进行评估。

## 0. 安装

```python
%pip install -U sentence-transformers
```

```python
from sentence_transformers import SentenceTransformer

# 加载模型
model = SentenceTransformer('all-MiniLM-L6-v2')
```

## 1. 检索

让我们选择检索作为第一个任务

```python
import random

from sentence_transformers.evaluation import InformationRetrievalEvaluator

from datasets import load_dataset
```

BeIR是一个著名的检索基准。让我们使用xxx数据集进行评估。

```python
# 加载Quora IR数据集 (https://huggingface.co/datasets/BeIR/quora, https://huggingface.co/datasets/BeIR/quora-qrels)
corpus = load_dataset("BeIR/quora", "corpus", split="corpus")
queries = load_dataset("BeIR/quora", "queries", split="queries")
relevant_docs_data = load_dataset("BeIR/quora-qrels", split="validation")
```

```python
# 大幅缩小语料库规模，只保留相关文档 + 10,000个随机文档
required_corpus_ids = list(map(str, relevant_docs_data["corpus-id"]))
required_corpus_ids += random.sample(corpus["_id"], k=10_000)
corpus = corpus.filter(lambda x: x["_id"] in required_corpus_ids)

# 将数据集转换为字典
corpus = dict(zip(corpus["_id"], corpus["text"]))  # 我们的语料库 (cid => document)
queries = dict(zip(queries["_id"], queries["text"]))  # 我们的查询 (qid => question)
relevant_docs = {}  # 查询ID到相关文档的映射 (qid => set([relevant_cids])
for qid, corpus_ids in zip(relevant_docs_data["query-id"], relevant_docs_data["corpus-id"]):
    qid = str(qid)
    corpus_ids = str(corpus_ids)
    if qid not in relevant_docs:
        relevant_docs[qid] = set()
    relevant_docs[qid].add(corpus_ids)
```

最后，我们准备好进行评估。

```python
# 给定查询、语料库和相关文档的映射，InformationRetrievalEvaluator计算不同的IR指标
ir_evaluator = InformationRetrievalEvaluator(
    queries=queries,
    corpus=corpus,
    relevant_docs=relevant_docs,
    name="BeIR-quora-dev",
)

results = ir_evaluator(model)
```

