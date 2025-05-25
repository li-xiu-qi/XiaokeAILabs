# 评估

评估是所有机器学习任务中的关键部分。在本教程中，我们将完整地介绍在[MS Marco](https://microsoft.github.io/msmarco/)上评估嵌入模型性能的整个流程，并使用三种指标来展示其性能。

## 步骤0：设置

在环境中安装依赖。

```python
%pip install -U FlagEmbedding faiss-cpu
```

## 步骤1：加载数据集

首先，从Huggingface数据集中下载查询和MS Marco

```python
from datasets import load_dataset
import numpy as np

data = load_dataset("namespace-Pt/msmarco", split="dev")
```

考虑到时间成本，我们将在本教程中使用截断的数据集。`queries`包含数据集中的前100个查询。`corpus`由前5,000个查询的正向样本组成。

```python
queries = np.array(data[:100]["query"])
corpus = sum(data[:5000]["positive"], [])
```

如果你有GPU并且想尝试对MS Marco进行完整评估，请取消下面单元格的注释并运行：

```python
# data = load_dataset("namespace-Pt/msmarco", split="dev")
# queries = np.array(data["query"])

# corpus = load_dataset("namespace-PT/msmarco-corpus", split="train")
```

## 步骤2：嵌入

选择我们想要评估的嵌入模型，并将语料库编码为嵌入。

```python
from FlagEmbedding import FlagModel

# get the BGE embedding model
model = FlagModel('BAAI/bge-base-en-v1.5',
                  query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                  use_fp16=True)

# get the embedding of the corpus
corpus_embeddings = model.encode(corpus)

print("shape of the corpus embeddings:", corpus_embeddings.shape)
print("data type of the embeddings: ", corpus_embeddings.dtype)
```

输出:

```
Inference Embeddings: 100%|██████████| 21/21 [02:10<00:00,  6.22s/it]
```

输出:

```
shape of the corpus embeddings: (5331, 768)
data type of the embeddings:  float32

```

输出:

```


```

## 步骤3：索引

我们使用index_factory()函数创建我们想要的Faiss索引：

- 第一个参数`dim`是向量空间的维度，在这个例子中如果你使用的是bge-base-en-v1.5，则是768。

- 第二个参数`'Flat'`使索引进行穷举搜索。

- 第三个参数`faiss.METRIC_INNER_PRODUCT`告诉索引使用内积作为距离度量。

```python
import faiss

# get the length of our embedding vectors, vectors by bge-base-en-v1.5 have length 768
dim = corpus_embeddings.shape[-1]

# create the faiss index and store the corpus embeddings into the vector space
index = faiss.index_factory(dim, 'Flat', faiss.METRIC_INNER_PRODUCT)
corpus_embeddings = corpus_embeddings.astype(np.float32)
# train and add the embeddings to the index
index.train(corpus_embeddings)
index.add(corpus_embeddings)

print(f"total number of vectors: {index.ntotal}")
```

输出:

```
total number of vectors: 5331

```

由于嵌入过程很耗时，保存索引以便复现或进行其他实验是一个不错的选择。

取消下面几行的注释以保存索引。

```python
# path = "./index.bin"
# faiss.write_index(index, path)
```

如果你已经在本地目录中存储了索引，可以通过以下方式加载它：

```python
# index = faiss.read_index("./index.bin")
```

## 步骤4：检索

获取所有查询的嵌入，并获取它们对应的真实答案用于评估。

```python
query_embeddings = model.encode_queries(queries)
ground_truths = [d["positive"] for d in data]
corpus = np.asarray(corpus)
```

使用faiss索引搜索每个查询的前$k$个答案。

```python
from tqdm import tqdm

res_scores, res_ids, res_text = [], [], []
query_size = len(query_embeddings)
batch_size = 256
# The cutoffs we will use during evaluation, and set k to be the maximum of the cutoffs.
cut_offs = [1, 10]
k = max(cut_offs)

for i in tqdm(range(0, query_size, batch_size), desc="Searching"):
    q_embedding = query_embeddings[i: min(i+batch_size, query_size)].astype(np.float32)
    # search the top k answers for each of the queries
    score, idx = index.search(q_embedding, k=k)
    res_scores += list(score)
    res_ids += list(idx)
    res_text += list(corpus[idx])
```

输出:

```
Searching: 100%|██████████| 1/1 [00:00<00:00, 20.91it/s]

```

## 步骤5：评估

### 5.1 召回率

召回率表示模型从数据集中所有实际正样本中正确预测正样本的能力。

$$\textbf{Recall}=\frac{\text{真正例}}{\text{真正例}+\text{假负例}}$$

当假负例成本较高时，召回率很有用。换句话说，我们试图找到正类的所有对象，即使这会导致一些假正例。这一特性使召回率成为文本检索任务的有用指标。

```python
def calc_recall(preds, truths, cutoffs):
    recalls = np.zeros(len(cutoffs))
    for text, truth in zip(preds, truths):
        for i, c in enumerate(cutoffs):
            recall = np.intersect1d(truth, text[:c])
            recalls[i] += len(recall) / max(min(c, len(truth)), 1)
    recalls /= len(preds)
    return recalls

recalls = calc_recall(res_text, ground_truths, cut_offs)
for i, c in enumerate(cut_offs):
    print(f"recall@{c}: {recalls[i]}")
```

输出:

```
recall@1: 0.97
recall@10: 1.0

```

### 5.2 MRR

平均倒数排名（[MRR](https://en.wikipedia.org/wiki/Mean_reciprocal_rank)）是信息检索中广泛使用的一种指标，用于评估系统的有效性。它衡量的是搜索结果列表中第一个相关结果的排名位置。

$$MRR=\frac{1}{|Q|}\sum_{i=1}^{|Q|}\frac{1}{rank_i}$$

其中：
- $|Q|$ 是查询的总数。
- $rank_i$ 是第i个查询的第一个相关文档的排名位置。

```python
def MRR(preds, truth, cutoffs):
    mrr = [0 for _ in range(len(cutoffs))]
    for pred, t in zip(preds, truth):
        for i, c in enumerate(cutoffs):
            for j, p in enumerate(pred):
                if j < c and p in t:
                    mrr[i] += 1/(j+1)
                    break
    mrr = [k/len(preds) for k in mrr]
    return mrr
```

```python
mrr = MRR(res_text, ground_truths, cut_offs)
for i, c in enumerate(cut_offs):
    print(f"MRR@{c}: {mrr[i]}")
```

输出:

```
MRR@1: 0.97
MRR@10: 0.9825

```

### 5.3 nDCG

归一化折扣累积增益（nDCG）通过考虑相关文档的位置及其分级相关性分数，来衡量搜索结果排名列表的质量。nDCG的计算涉及两个主要步骤：

1. 折扣累积增益（DCG）衡量检索任务中的排名质量。

$$DCG_p=\sum_{i=1}^p\frac{2^{rel_i}-1}{\log_2(i+1)}$$

2. 通过理想DCG进行归一化，使其在不同查询之间具有可比性。
$$nDCG_p=\frac{DCG_p}{IDCG_p}$$
其中$IDCG$是给定文档集合的最大可能DCG，假设它们按相关性完美排序。

```python
pred_hard_encodings = []
for pred, label in zip(res_text, ground_truths):
    pred_hard_encoding = list(np.isin(pred, label).astype(int))
    pred_hard_encodings.append(pred_hard_encoding)
```

```python
from sklearn.metrics import ndcg_score

for i, c in enumerate(cut_offs):
    nDCG = ndcg_score(pred_hard_encodings, res_scores, k=c)
    print(f"nDCG@{c}: {nDCG}")
```

输出:

```
nDCG@1: 0.97
nDCG@10: 0.9869253606521631

```

恭喜！你已经完成了评估嵌入模型的完整流程。请随意尝试不同的数据集和模型！

