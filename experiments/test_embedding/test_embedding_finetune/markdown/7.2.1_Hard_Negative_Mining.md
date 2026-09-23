# 难负样本

难负样本是指那些模型特别难以与正样本区分开的负样本。它们通常靠近决策边界，或者表现出与正样本高度相似的特征。因此，难负样本挖掘被广泛应用于机器学习任务中，以使模型专注于相似实例之间的细微差异，从而提高判别能力。

在文本检索系统中，难负样本可能是与查询具有某些特征相似性但不真正满足查询意图的文档。在检索过程中，这些文档的排名可能高于真正的答案。因此，明确地在这些难负样本上训练模型是非常有价值的。

## 1. 准备

首先，加载一个嵌入模型：

```python
from FlagEmbedding import FlagModel

model = FlagModel('BAAI/bge-base-en-v1.5')
```

输出:

```
/share/project/xzy/Envs/ft/lib/python3.11/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
  from .autonotebook import tqdm as notebook_tqdm

```

然后，从数据集中加载查询和语料库：

```python
from datasets import load_dataset

corpus = load_dataset("BeIR/scifact", "corpus")["corpus"]
queries = load_dataset("BeIR/scifact", "queries")["queries"]

corpus_ids = corpus.select_columns(["_id"])["_id"]
corpus = corpus.select_columns(["text"])["text"]
```

我们创建一个字典，用于映射 FAISS 索引使用的自动生成的 ID（从 0 开始），以供后续使用。

```python
corpus_ids_map = {}
for i in range(len(corpus)):
    corpus_ids_map[i] = corpus_ids[i]
```

## 2. 索引

使用嵌入模型对查询和语料库进行编码：

```python
p_vecs = model.encode(corpus)
```

输出:

```
pre tokenize: 100%|██████████| 21/21 [00:00<00:00, 46.18it/s]
You're using a BertTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
Attempting to cast a BatchEncoding to type None. This is not supported.
/share/project/xzy/Envs/ft/lib/python3.11/site-packages/_distutils_hack/__init__.py:54: UserWarning: Reliance on distutils from stdlib is deprecated. Users must rely on setuptools to provide the distutils module. Avoid importing distutils or import setuptools first, and avoid setting SETUPTOOLS_USE_DISTUTILS=stdlib. Register concerns at https://github.com/pypa/setuptools/issues/new?template=distutils-deprecation.yml
  warnings.warn(
Inference Embeddings:   0%|          | 0/21 [00:00<?, ?it/s]Attempting to cast a BatchEncoding to type None. This is not supported.
Inference Embeddings:   5%|▍         | 1/21 [00:49<16:20, 49.00s/it]Attempting to cast a BatchEncoding to type None. This is not supported.
Inference Embeddings:  10%|▉         | 2/21 [01:36<15:10, 47.91s/it]Attempting to cast a BatchEncoding to type None. This is not supported.
Inference Embeddings:  14%|█▍        | 3/21 [02:16<13:23, 44.66s/it]Attempting to cast a BatchEncoding to type None. This is not supported.
Inference Embeddings:  19%|█▉        | 4/21 [02:52<11:39, 41.13s/it]Attempting to cast a BatchEncoding to type None. This is not supported.
Inference Embeddings:  24%|██▍       | 5/21 [03:23<09:58, 37.38s/it]Attempting to cast a BatchEncoding to type None. This is not supported.
Inference Embeddings:  29%|██▊       | 6/21 [03:55<08:51, 35.44s/it]Attempting to cast a BatchEncoding to type None. This is not supported.
Inference Embeddings:  33%|███▎      | 7/21 [04:24<07:47, 33.37s/it]Attempting to cast a BatchEncoding to type None. This is not supported.
Inference Embeddings:  38%|███▊      | 8/21 [04:51<06:49, 31.51s/it]Attempting to cast a BatchEncoding to type None. This is not supported.
Inference Embeddings:  43%|████▎     | 9/21 [05:16<05:52, 29.37s/it]Attempting to cast a BatchEncoding to type None. This is not supported.
Inference Embeddings:  48%|████▊     | 10/21 [05:42<05:13, 28.51s/it]Attempting to cast a BatchEncoding to type None. This is not supported.
Inference Embeddings:  52%|█████▏    | 11/21 [06:05<04:25, 26.59s/it]Attempting to cast a BatchEncoding to type None. This is not supported.
Inference Embeddings:  57%|█████▋    | 12/21 [06:26<03:43, 24.85s/it]Attempting to cast a BatchEncoding to type None. This is not supported.
Inference Embeddings:  62%|██████▏   | 13/21 [06:45<03:06, 23.35s/it]Attempting to cast a BatchEncoding to type None. This is not supported.
Inference Embeddings:  67%|██████▋   | 14/21 [07:04<02:33, 21.89s/it]Attempting to cast a BatchEncoding to type None. This is not supported.
Inference Embeddings:  71%|███████▏  | 15/21 [07:21<02:03, 20.54s/it]Attempting to cast a BatchEncoding to type None. This is not supported.
Inference Embeddings:  76%|███████▌  | 16/21 [07:38<01:36, 19.30s/it]Attempting to cast a BatchEncoding to type None. This is not supported.
Inference Embeddings:  81%|████████  | 17/21 [07:52<01:11, 17.87s/it]Attempting to cast a BatchEncoding to type None. This is not supported.
Inference Embeddings:  86%|████████▌ | 18/21 [08:06<00:49, 16.58s/it]Attempting to cast a BatchEncoding to type None. This is not supported.
Inference Embeddings:  90%|█████████ | 19/21 [08:18<00:30, 15.21s/it]Attempting to cast a BatchEncoding to type None. This is not supported.
Inference Embeddings:  95%|█████████▌| 20/21 [08:28<00:13, 13.56s/it]Attempting to cast a BatchEncoding to type None. This is not supported.
Inference Embeddings: 100%|██████████| 21/21 [08:29<00:00, 24.26s/it]

```

```python
p_vecs.shape
```

结果:

```
(5183, 768)
```

然后创建一个 FAISS 索引

```python
import torch, faiss
import numpy as np

# 创建一个维度与我们的嵌入匹配的基本平面索引
index = faiss.IndexFlatIP(len(p_vecs[0]))
# 确保嵌入是 float32 类型
p_vecs = np.asarray(p_vecs, dtype=np.float32)
# 使用 GPU 加速索引搜索
if torch.cuda.is_available():
    co = faiss.GpuMultipleClonerOptions()
    co.shard = True
    co.useFloat16 = True
    index = faiss.index_cpu_to_all_gpus(index, co=co)
# 将所有嵌入添加到索引中
index.add(p_vecs)
```

## 3. 搜索

为了更好地演示，让我们使用单个查询：

```python
query = queries[0]
query
```

结果:

```
{'_id': '0',
 'title': '',
 'text': '0-dimensional biomaterials lack inductive properties.'}
```

获取该查询的 ID 和内容，然后使用我们的嵌入模型获取其嵌入向量。

```python
q_id, q_text = query["_id"], query["text"]
# 使用 encode_queries() 函数对查询进行编码
q_vec = model.encode_queries(queries=q_text)
```

使用索引搜索最接近的结果：

```python
_, ids = index.search(np.expand_dims(q_vec, axis=0), k=15)
# 将自动生成的 ID 转换回原始数据集中的 ID
converted = [corpus_ids_map[id] for id in ids[0]]
converted
```

结果:

```
['4346436',
 '17388232',
 '14103509',
 '37437064',
 '29638116',
 '25435456',
 '32532238',
 '31715818',
 '23763738',
 '7583104',
 '21456232',
 '2121272',
 '35621259',
 '58050905',
 '196664003']
```

```python
qrels = load_dataset("BeIR/scifact-qrels")["train"]
pos_id = qrels[0]
pos_id
```

结果:

```
{'query-id': 0, 'corpus-id': 31715818, 'score': 1}
```

最后，我们使用 top-k shifted by N 的方法，即获取排名第五之后的前 10 个负样本。

```python
negatives = [id for id in converted[5:] if int(id) != pos_id["corpus-id"]]
negatives
```

结果:

```
['25435456',
 '32532238',
 '23763738',
 '7583104',
 '21456232',
 '2121272',
 '35621259',
 '58050905',
 '196664003']
```

现在我们已经为第一个查询选择了一组难负样本！

还有其他方法可以优化选择难负样本的过程。例如，我们 GitHub 仓库中的[实现](https://github.com/FlagOpen/FlagEmbedding/blob/master/scripts/hn_mine.py)获取的是排名第 10 到第 210 的样本，即 top 10-210。然后从这 200 个候选项中采样 15 个。原因是直接选择 top K 可能会将一些假负样本（与查询有些相关但并非查询的确切答案的段落）引入负样本集。这可能会影响模型的性能。

