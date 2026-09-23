# 评估微调模型

在前面的部分中，我们准备了数据集并对模型进行了微调。在本教程中，我们将介绍如何使用我们构建的测试数据集来评估模型。

## 0. 安装

```python
% pip install -U datasets pytrec_eval FlagEmbedding
```

## 1. 加载数据

我们首先从处理过的文件中加载数据。

```python
from datasets import load_dataset

queries = load_dataset("json", data_files="ft_data/test_queries.jsonl")["train"]
corpus = load_dataset("json", data_files="ft_data/corpus.jsonl")["train"]
qrels = load_dataset("json", data_files="ft_data/test_qrels.jsonl")["train"]

queries_text = queries["text"]
corpus_text = [text for sub in corpus["text"] for text in sub]
```

```python
qrels_dict = {}
for line in qrels:
    if line['qid'] not in qrels_dict:
        qrels_dict[line['qid']] = {}
    qrels_dict[line['qid']][line['docid']] = line['relevance']
```

## 2. 搜索

然后我们准备一个函数，将文本编码为嵌入向量并搜索结果：

```python
import faiss
import numpy as np
from tqdm import tqdm


def search(model, queries_text, corpus_text):
    
    queries_embeddings = model.encode_queries(queries_text)
    corpus_embeddings = model.encode_corpus(corpus_text)
    
    # 创建 Faiss 索引并存储嵌入向量
    dim = corpus_embeddings.shape[-1]
    index = faiss.index_factory(dim, 'Flat', faiss.METRIC_INNER_PRODUCT)
    corpus_embeddings = corpus_embeddings.astype(np.float32)
    index.train(corpus_embeddings)
    index.add(corpus_embeddings)
    
    query_size = len(queries_embeddings)

    all_scores = []
    all_indices = []

    # 为所有查询搜索前 100 个答案
    for i in tqdm(range(0, query_size, 32), desc="Searching"):
        j = min(i + 32, query_size)
        query_embedding = queries_embeddings[i: j]
        score, indice = index.search(query_embedding.astype(np.float32), k=100)
        all_scores.append(score)
        all_indices.append(indice)

    all_scores = np.concatenate(all_scores, axis=0)
    all_indices = np.concatenate(all_indices, axis=0)
    
    # 将结果存储为评估格式
    results = {}
    for idx, (scores, indices) in enumerate(zip(all_scores, all_indices)):
        results[queries["id"][idx]] = {}
        for score, index in zip(scores, indices):
            if index != -1:
                results[queries["id"][idx]][corpus["id"][index]] = float(score)
                
    return results
```

## 3. 评估

```python
from FlagEmbedding.abc.evaluation.utils import evaluate_metrics, evaluate_mrr
from FlagEmbedding import FlagModel

k_values = [10,100]

raw_name = "BAAI/bge-large-en-v1.5"
finetuned_path = "test_encoder_only_base_bge-large-en-v1.5"
```

原始模型的结果：

```python
raw_model = FlagModel(
    raw_name, 
    query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
    devices=[0],
    use_fp16=False
)

results = search(raw_model, queries_text, corpus_text)

eval_res = evaluate_metrics(qrels_dict, results, k_values)
mrr = evaluate_mrr(qrels_dict, results, k_values)

for res in eval_res:
    print(res)
print(mrr)
```

输出:

```
pre tokenize: 100%|██████████| 3/3 [00:00<00:00, 129.75it/s]
You're using a BertTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
Inference Embeddings: 100%|██████████| 3/3 [00:00<00:00, 11.08it/s]
pre tokenize: 100%|██████████| 28/28 [00:00<00:00, 164.29it/s]
Inference Embeddings: 100%|██████████| 28/28 [00:04<00:00,  6.09it/s]
Searching: 100%|██████████| 22/22 [00:08<00:00,  2.56it/s]

```

输出:

```
defaultdict(<class 'list'>, {'NDCG@10': 0.70405, 'NDCG@100': 0.73528})
defaultdict(<class 'list'>, {'MAP@10': 0.666, 'MAP@100': 0.67213})
defaultdict(<class 'list'>, {'Recall@10': 0.82286, 'Recall@100': 0.97286})
defaultdict(<class 'list'>, {'P@10': 0.08229, 'P@100': 0.00973})
defaultdict(<class 'list'>, {'MRR@10': 0.666, 'MRR@100': 0.67213})

```

然后是微调后模型的结果：

```python
ft_model = FlagModel(
    finetuned_path, 
    query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
    devices=[0],
    use_fp16=False
)

results = search(ft_model, queries_text, corpus_text)

eval_res = evaluate_metrics(qrels_dict, results, k_values)
mrr = evaluate_mrr(qrels_dict, results, k_values)

for res in eval_res:
    print(res)
print(mrr)
```

输出:

```
pre tokenize: 100%|██████████| 3/3 [00:00<00:00, 164.72it/s]
You're using a BertTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
Inference Embeddings: 100%|██████████| 3/3 [00:00<00:00,  9.45it/s]
pre tokenize: 100%|██████████| 28/28 [00:00<00:00, 160.19it/s]
Inference Embeddings: 100%|██████████| 28/28 [00:04<00:00,  6.06it/s]
Searching: 100%|██████████| 22/22 [00:07<00:00,  2.80it/s]

```

输出:

```
defaultdict(<class 'list'>, {'NDCG@10': 0.84392, 'NDCG@100': 0.85792})
defaultdict(<class 'list'>, {'MAP@10': 0.81562, 'MAP@100': 0.81875})
defaultdict(<class 'list'>, {'Recall@10': 0.93143, 'Recall@100': 0.99429})
defaultdict(<class 'list'>, {'P@10': 0.09314, 'P@100': 0.00994})
defaultdict(<class 'list'>, {'MRR@10': 0.81562, 'MRR@100': 0.81875})

```

我们可以看到所有指标都有明显提升。

