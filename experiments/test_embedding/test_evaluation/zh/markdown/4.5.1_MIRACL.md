# 在MIRACL上评估

[MIRACL](https://project-miracl.github.io/)（跨语言连续体的多语言信息检索）是WSDM 2023 Cup挑战赛，专注于跨18种不同语言的搜索。他们发布了一个多语言检索数据集，包含16种"已知语言"的训练和开发集，以及2种"惊喜语言"的仅开发集。这些主题由各语言的母语使用者生成，他们也标注了主题与给定文档列表之间的相关性。您可以在HuggingFace上找到该数据集。

注意：我们强烈建议您在GPU上运行MIRACL的评估。作为参考，在8xA100 40G节点上，整个过程大约需要一小时。

## 0. 安装

首先安装我们将使用的库：

```python
% pip install FlagEmbedding pytrec_eval
```

## 1. 数据集

凭借18种语言中大量的段落和文章，MIRACL是用于训练或评估多语言模型的丰富数据集。数据可以从[Hugging Face](https://huggingface.co/datasets/miracl/miracl)下载。

| 语言              | 段落数量 | 文章数量 |
|:----------------|--------------:|--------------:|
| 阿拉伯语 (ar)      |     2,061,414 |       656,982 |
| 孟加拉语 (bn)      |       297,265 |        63,762 |
| 英语 (en)        |    32,893,221 |     5,758,285 |
| 西班牙语 (es)      |    10,373,953 |     1,669,181 |
| 波斯语 (fa)       |     2,207,172 |       857,827 |
| 芬兰语 (fi)       |     1,883,509 |       447,815 |
| 法语 (fr)        |    14,636,953 |     2,325,608 |
| 印地语 (hi)       |       506,264 |       148,107 |
| 印尼语 (id)       |     1,446,315 |       446,330 |
| 日语 (ja)        |     6,953,614 |     1,133,444 |
| 韩语 (ko)        |     1,486,752 |       437,373 |
| 俄语 (ru)        |     9,543,918 |     1,476,045 |
| 斯瓦希里语 (sw)    |       131,924 |        47,793 |
| 泰卢固语 (te)     |       518,079 |        66,353 |
| 泰语 (th)        |       542,166 |       128,179 |
| 中文 (zh)        |     4,934,368 |     1,246,389 |

```python
from datasets import load_dataset

lang = "en"
corpus = load_dataset("miracl/miracl-corpus", lang, trust_remote_code=True)['train']
```

语料库中的每个段落有三个部分：`docid`、`title`和`text`。在具有docid `x#y`的文档结构中，`x`表示维基百科文章的id，而`y`是该文章内段落的编号。标题是该段落所属的id为`x`的文章名称。文本是段落的正文内容。

```python
corpus[0]
```

结果:

```
{'docid': '56672809#4',
 'title': 'Glen Tomasetti',
 'text': '1967年，Tomasetti因拒绝支付六分之一的税款而被起诉，理由是联邦预算的六分之一用于资助澳大利亚在越南的军事存在。在法庭上，她辩称澳大利亚参与越南战争违反了其作为联合国成员国的国际法律义务。Joan Baez等公众人物在美国进行了类似的抗议，但据当时的新闻报道，Tomasetti的起诉"被认为是澳大利亚首例此类案件"。Tomasetti最终被命令支付未付的税款。'}
```

qrels的形式如下：

```python
dev = load_dataset('miracl/miracl', lang, trust_remote_code=True)['dev']
```

```python
dev[0]
```

结果:

```
{'query_id': '0',
 'query': '克里奥尔语是法语的洋泾浜语吗？',
 'positive_passages': [{'docid': '462221#4',
   'text': "二战结束后的1945年，朝鲜被分为北朝鲜和南朝鲜，北朝鲜（在苏联的协助下）于1946年成为共产主义政府，称为朝鲜民主主义人民共和国，随后南朝鲜成为大韩民国。中国在1949年成为共产主义的中华人民共和国。1950年，苏联支持北朝鲜，而美国支持南朝鲜，中国与苏联结盟，这成为冷战的第一次军事行动。",
   'title': '第八美国陆军'},
  {'docid': '29810#23',
   'text': '德克萨斯州的巨大面积和位于多个气候带交叉处的位置使该州气候多变。该州的平原地区比北德克萨斯州有更冷的冬季，而墨西哥湾沿岸则冬季温和。德克萨斯州降水模式差异很大。位于该州西端的埃尔帕索年均降雨量为，而东南部地区的年均降雨量高达。北中部地区的达拉斯年均降雨量较为适中，每年。',
   'title': '德克萨斯州'},
  {'docid': '3716905#0',
   'text': '法语克里奥尔语，或基于法语的克里奥尔语，是一种克里奥尔语言（有母语使用者的接触语言），其中法语是"词汇提供者"。这种词汇提供者通常不是现代法语，而是17世纪来自巴黎、法国大西洋港口和新兴法国殖民地的共通语。基于法语的克里奥尔语全球有数百万母语使用者，主要分布在美洲和印度洋各群岛。本文还包含关于法语洋泾浜语的信息，这是一种缺乏母语使用者的接触语言。',
   'title': '基于法语的克里奥尔语言'},
  {'docid': '22399755#18',
   'text': '关于海地克里奥尔语起源有许多假说。语言学家John Singler认为，它很可能在法国殖民时期出现，当时经济转向大量依赖糖生产。这导致更多被奴役的非洲人，他们与法国人的互动创造了方言从洋泾浜语演变为克里奥尔语的条件。他的研究和魁北克蒙特利尔大学的Claire Lefebvre的研究表明，尽管克里奥尔语90%的词汇来自法语，但它在句法上是丰语（Fon）的亲戚，丰语是贝宁使用的尼日尔-刚果语系的Gbe语言。在海地克里奥尔语出现时，海地50%的被奴役非洲人是Gbe语言使用者。',
   'title': '海地文学'}],
 'negative_passages': [{'docid': '1170520#2',
   'text': '路易斯安那克里奥尔语是一种接触语言，18世纪由说标准法语（词汇提供语言）和几种来自非洲的底层或附层语言的人之间的互动产生。在成为克里奥尔语之前，其前身被视为洋泾浜语。导致路易斯安那克里奥尔语产生的社会情况很独特，因为词汇提供语言是接触地点发现的语言。通常词汇提供者是抵达接触地点的语言，属于底层/附层语言。法国人、法裔加拿大人和非洲奴隶都不是该地区的本地人；这一事实将路易斯安那克里奥尔语归类为外来民族之间产生的接触语言。一旦这种洋泾浜语作为"通用语"（lingua franca）传给下一代（他们被视为新语法的第一批母语使用者），它就可以被归类为克里奥尔语。',
   'title': '路易斯安那克里奥尔语'},
  {'docid': '49823#1',
   'text': '确切的克里奥尔语言数量尚不清楚，特别是因为许多语言记录或文档不足。自1500年以来，约有一百种克里奥尔语言出现。由于欧洲大航海时代和随之而来的大西洋奴隶贸易，这些语言主要基于英语和法语等欧洲语言。随着造船和航海技术的改进，贸易商必须学会与世界各地的人交流，而最快的方法是开发一种洋泾浜语，或简化的语言适合这一目的；进而，完整的克里奥尔语从这些洋泾浜语中发展而来。除了以欧洲语言为基础的克里奥尔语外，还有例如基于阿拉伯语、汉语和马来语的克里奥尔语。使用人数最多的克里奥尔语是海地克里奥尔语，有近一千万母语使用者，其次是托克皮辛语（Tok Pisin），约有400万人使用，其中大部分是第二语言使用者。',
   'title': '克里奥尔语'},
  {'docid': '1651722#10',
   'text': '克里奥语（Krio）是一种基于英语的克里奥尔语，尼日利亚洋泾浜英语、喀麦隆洋泾浜英语和皮钦吉利语（Pichinglis）都源自它。它也与美洲使用的基于英语的克里奥尔语言相似，尤其是古拉语（Gullah）、牙买加巴托瓦语（Jamaican Patois，牙买加克里奥尔语）和巴巴多斯克里奥尔语（Bajan Creole），但它有自己独特的特点。它还与加勒比地区的非英语克里奥尔语言（如基于法语的克里奥尔语言）有一些语言相似性。',
   'title': '克里奥语'},
  {'docid': '540382#4',
   'text': '直到最近，克里奥尔语被认为是不值得关注的葡萄牙语"退化"方言。因此，关于其形成细节的文档很少。自20世纪以来，语言学家对克里奥尔语的研究增多，提出了几种理论。洋泾浜语的单源理论假设，15至18世纪在葡萄牙人在西非海岸建立的堡垒中，使用了某种基于葡萄牙语的洋泾浜语言——称为西非洋泾浜葡萄牙语。根据这一理论，这种变体可能是所有洋泾浜语和克里奥尔语言的起点。这在一定程度上可以解释为什么许多克里奥尔语中可以找到葡萄牙语词汇项，但更重要的是，它可以解释这些语言共享的众多语法相似性，例如介词"na"，意思是"在"和/或"在...上"，它来源于葡萄牙语的缩写"na"，意思是"在...中"（阴性单数）。',
   'title': '基于葡萄牙语的克里奥尔语言'},
  {'docid': '49823#7',
   'text': '其他学者，如Salikoko Mufwene，认为洋泾浜语和克里奥尔语在不同情况下独立产生，洋泾浜语不总是必须先于克里奥尔语，克里奥尔语也不一定从洋泾浜语演变而来。根据Mufwene的说法，洋泾浜语出现在贸易殖民地，在"保留其本地方言进行日常互动的用户之间"。与此同时，克里奥尔语在定居殖民地发展，在这些地方，欧洲语言的使用者，通常是契约仆人，他们的语言与标准语言相去甚远，与非欧洲奴隶广泛互动，吸收了奴隶非欧洲母语中的某些词汇和特征，导致原始语言的严重基层化版本。这些仆人和奴隶开始将克里奥尔语用作日常通用语，而不仅仅是在需要与上层语言使用者交流的情况下使用。',
   'title': '克里奥尔语'},
  {'docid': '11236157#2',
   'text': '虽然世界上许多克里奥尔语的词汇基于葡萄牙语以外的语言（如英语、法语、西班牙语、荷兰语），但有人假设这些克里奥尔语是通过重新词汇化从这种通用语衍生而来的，即一种洋泾浜语或克里奥尔语保留语法但从另一种语言大量吸收词汇的过程。有一些证据表明重新词汇化是一个真实的过程。Pieter Muysken和显示存在一些语言，其语法和词汇分别来自两种不同的语言，这可以通过重新词汇化假设轻松解释。此外，萨拉马坎语（Saramaccan）似乎是一种洋泾浜语，处于从葡萄牙语向英语重新词汇化的中间阶段。然而，在这种混合语言的情况下，混合语言的语法或词汇与他们归因的语言的语法或词汇之间从来没有一一对应关系。',
   'title': '洋泾浜语单源理论'},
  {'docid': '1612877#8',
   'text': '混合语言在非常基本的方面不同于洋泾浜语、克里奥尔语和代码转换。在大多数情况下，混合语言使用者是两种语言的流利、甚至是母语使用者；然而，Michif（一种N-V混合语言）的使用者很独特，因为许多人不流利掌握两种源语言。另一方面，洋泾浜语通常在两种（或更多）不同语言的使用者需要找到某种方式相互交流的情况下发展，通常是在贸易环境中。当洋泾浜语成为年轻使用者的第一语言时，克里奥尔语就发展起来了。虽然克里奥尔语往往具有大幅简化的形态，但混合语言通常保留一种或两种父语言的屈折复杂性。例如，Michif保留了其法语名词和克里语动词的复杂性。',
   'title': '混合语言'},
  {'docid': '9606120#4',
   'text': '虽然它被归类为洋泾浜语，但这是不准确的。使用者已经流利掌握英语和法语，因此在双方缺乏共同语言的情况下不会使用它。总的来说，Camfranglais与其他洋泾浜语和克里奥尔语的不同之处在于，它由一系列语言组成，其中至少有一种已经为使用者所知。例如，虽然它包含借用、代码转换和洋泾浜语的元素，但它不是一种接触语言，因为双方可以假定都会说法语，即词汇提供者。已经提出了许多其他分类，如'洋泾浜语'、'行话'、'青年语言'、'喀麦隆混合语'、'法语的本土化挪用'或'混合俚语'。然而，由于Camfranglais比俚语更发达，这也是不够的。Kießling提出将其归类为'城市青年型高度混合社会方言'，Stein-Kanjora也同意这一定义。',
   'title': 'Camfranglais'}]}
```

每个项目有四个部分：`query_id`、`query`、`positive_passages`和`negative_passages`。这里，`query_id`和`query`对应查询的id和文本内容。`positive_passages`和`negative_passages`是包含`docid`、`title`和`text`的段落列表。

这种结构在`train`、`dev`、`testA`和`testB`集合中是相同的。

然后我们处理查询和语料库的id和文本，并获取开发集的qrels。

```python
corpus_ids = corpus['docid']
corpus_text = []
for doc in corpus:
   corpus_text.append(f"{doc['title']} {doc['text']}".strip())

queries_ids = dev['query_id']
queries_text = dev['query']
```

## 2. 从零开始评估

### 2.1 嵌入

在这个演示中我们使用bge-base-en-v1.5，您可以随意更换为您喜欢的模型。

```python
import os 
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
os.environ['SETUPTOOLS_USE_DISTUTILS'] = ''
```

```python
from FlagEmbedding import FlagModel

# 获取BGE嵌入模型
model = FlagModel('BAAI/bge-base-en-v1.5')

# 获取查询和语料库的嵌入
queries_embeddings = model.encode_queries(queries_text)
corpus_embeddings = model.encode_corpus(corpus_text)

print("嵌入的形状:", corpus_embeddings.shape)
print("嵌入的数据类型: ", corpus_embeddings.dtype)
```

输出:

```
初始目标设备: 100%|██████████| 8/8 [00:29<00:00,  3.66s/it]
预标记化: 100%|██████████| 1/1 [00:00<00:00, 52.84it/s]
预标记化: 100%|██████████| 1/1 [00:00<00:00, 55.15it/s]
预标记化: 100%|██████████| 1/1 [00:00<00:00, 56.49it/s]
预标记化: 100%|██████████| 1/1 [00:00<00:00, 55.22it/s]
预标记化: 100%|██████████| 1/1 [00:00<00:00, 49.22it/s]
预标记化: 100%|██████████| 1/1 [00:00<00:00, 54.69it/s]
预标记化: 100%|██████████| 1/1 [00:00<00:00, 49.16it/s]
预标记化: 100%|██████████| 1/1 [00:00<00:00, 50.77it/s]
块: 100%|██████████| 8/8 [00:10<00:00,  1.27s/it]
预标记化: 100%|██████████| 16062/16062 [08:12<00:00, 32.58it/s]  
预标记化: 100%|██████████| 16062/16062 [08:44<00:00, 30.60it/s]68s/it]
预标记化: 100%|██████████| 16062/16062 [08:39<00:00, 30.90it/s]41s/it]
预标记化: 100%|██████████| 16062/16062 [09:04<00:00, 29.49it/s]43s/it]
预标记化: 100%|██████████| 16062/16062 [09:27<00:00, 28.29it/s]it/s]t]
预标记化: 100%|██████████| 16062/16062 [09:08<00:00, 29.30it/s]32s/it]
预标记化: 100%|██████████| 16062/16062 [08:59<00:00, 29.77it/s]it/s]t]
预标记化: 100%|██████████| 16062/16062 [09:04<00:00, 29.50it/s]29s/it]
Inference Embeddings: 100%|██████████| 16062/16062 [17:10<00:00, 15.59it/s] 
Inference Embeddings: 100%|██████████| 16062/16062 [17:04<00:00, 15.68it/s]]
Inference Embeddings: 100%|██████████| 16062/16062 [17:01<00:00, 15.72it/s]s]
Inference Embeddings: 100%|██████████| 16062/16062 [17:28<00:00, 15.32it/s]
Inference Embeddings: 100%|██████████| 16062/16062 [17:43<00:00, 15.10it/s]
Inference Embeddings: 100%|██████████| 16062/16062 [17:27<00:00, 15.34it/s]
Inference Embeddings: 100%|██████████| 16062/16062 [17:36<00:00, 15.20it/s]
Inference Embeddings: 100%|██████████| 16062/16062 [17:31<00:00, 15.28it/s]
Chunks: 100%|██████████| 8/8 [27:49<00:00, 208.64s/it]

```

输出:

```
shape of the embeddings: (32893221, 768)
data type of the embeddings:  float16

```

### 2.2 Indexing

Create a Faiss index to store the embeddings.

```python
import faiss
import numpy as np

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
total number of vectors: 32893221

```

### 2.3 Searching

Use the Faiss index to search for each query.

```python
from tqdm import tqdm

query_size = len(queries_embeddings)

all_scores = []
all_indices = []

for i in tqdm(range(0, query_size, 32), desc="Searching"):
    j = min(i + 32, query_size)
    query_embedding = queries_embeddings[i: j]
    score, indice = index.search(query_embedding.astype(np.float32), k=100)
    all_scores.append(score)
    all_indices.append(indice)

all_scores = np.concatenate(all_scores, axis=0)
all_indices = np.concatenate(all_indices, axis=0)
```

输出:

```
Searching: 100%|██████████| 25/25 [15:03<00:00, 36.15s/it]

```

Then map the search results back to the indices in the dataset.

```python
results = {}
for idx, (scores, indices) in enumerate(zip(all_scores, all_indices)):
    results[queries_ids[idx]] = {}
    for score, index in zip(scores, indices):
        if index != -1:
            results[queries_ids[idx]][corpus_ids[index]] = float(score)
```

### 2.4 Evaluating

Download the qrels file for evaluation:

```python
endpoint = os.getenv('HF_ENDPOINT', 'https://huggingface.co')
file_name = "qrels.miracl-v1.0-en-dev.tsv"
qrel_url = f"wget {endpoint}/datasets/miracl/miracl/resolve/main/miracl-v1.0-en/qrels/{file_name}"

os.system(qrel_url)
```

输出:

```
--2024-11-21 10:26:16--  https://hf-mirror.com/datasets/miracl/miracl/resolve/main/miracl-v1.0-en/qrels/qrels.miracl-v1.0-en-dev.tsv
Resolving hf-mirror.com (hf-mirror.com)... 153.121.57.40, 133.242.169.68, 160.16.199.204
Connecting to hf-mirror.com (hf-mirror.com)|153.121.57.40|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 167817 (164K) [text/plain]
Saving to: ‘qrels.miracl-v1.0-en-dev.tsv’

     0K .......... .......... .......... .......... .......... 30%  109K 1s
    50K .......... .......... .......... .......... .......... 61% 44.5K 1s
   100K .......... .......... .......... .......... .......... 91% 69.6K 0s
   150K .......... ...                                        100% 28.0K=2.8s

2024-11-21 10:26:20 (58.6 KB/s) - ‘qrels.miracl-v1.0-en-dev.tsv’ saved [167817/167817]


```

结果:

```
0
```

Read the qrels from the file:

```python
qrels_dict = {}
with open(file_name, "r", encoding="utf-8") as f:
    for line in f.readlines():
        qid, _, docid, rel = line.strip().split("\t")
        qid, docid, rel = str(qid), str(docid), int(rel)
        if qid not in qrels_dict:
            qrels_dict[qid] = {}
        qrels_dict[qid][docid] = rel
```

Finally, use [pytrec_eval](https://github.com/cvangysel/pytrec_eval) library to help us calculate the scores of selected metrics:

```python
import pytrec_eval
from collections import defaultdict

ndcg_string = "ndcg_cut." + ",".join([str(k) for k in [10,100]])
recall_string = "recall." + ",".join([str(k) for k in [10,100]])

evaluator = pytrec_eval.RelevanceEvaluator(
    qrels_dict, {ndcg_string, recall_string}
)
scores = evaluator.evaluate(results)

all_ndcgs, all_recalls = defaultdict(list), defaultdict(list)
for query_id in scores.keys():
    for k in [10,100]:
        all_ndcgs[f"NDCG@{k}"].append(scores[query_id]["ndcg_cut_" + str(k)])
        all_recalls[f"Recall@{k}"].append(scores[query_id]["recall_" + str(k)])

ndcg, recall = (
    all_ndcgs.copy(),
    all_recalls.copy(),
)

for k in [10,100]:
    ndcg[f"NDCG@{k}"] = round(sum(ndcg[f"NDCG@{k}"]) / len(scores), 5)
    recall[f"Recall@{k}"] = round(sum(recall[f"Recall@{k}"]) / len(scores), 5)

print(ndcg)
print(recall)
```

输出:

```
defaultdict(<class 'list'>, {'NDCG@10': 0.46073, 'NDCG@100': 0.54336})
defaultdict(<class 'list'>, {'Recall@10': 0.55972, 'Recall@100': 0.83827})

```

## 3. Evaluate using FlagEmbedding

We provide independent evaluation for popular datasets and benchmarks. Try the following code to run the evaluation, or run the shell script provided in [example](../../examples/evaluation/miracl/eval_miracl.sh) folder.

```python
import sys

arguments = """- \
    --eval_name miracl \
    --dataset_dir ./miracl/data \
    --dataset_names en \
    --splits dev \
    --corpus_embd_save_dir ./miracl/corpus_embd \
    --output_dir ./miracl/search_results \
    --search_top_k 100 \
    --cache_path ./cache/data \
    --overwrite True \
    --k_values 10 100 \
    --eval_output_method markdown \
    --eval_output_path ./miracl/miracl_eval_results.md \
    --eval_metrics ndcg_at_10 recall_at_100 \
    --embedder_name_or_path BAAI/bge-base-en-v1.5 \
    --devices cuda:0 cuda:1 \
    --embedder_batch_size 1024
""".replace('\n','')

sys.argv = arguments.split()
```

```python
from transformers import HfArgumentParser

from FlagEmbedding.evaluation.miracl import (
    MIRACLEvalArgs, MIRACLEvalModelArgs,
    MIRACLEvalRunner
)


parser = HfArgumentParser((
    MIRACLEvalArgs,
    MIRACLEvalModelArgs
))

eval_args, model_args = parser.parse_args_into_dataclasses()
eval_args: MIRACLEvalArgs
model_args: MIRACLEvalModelArgs

runner = MIRACLEvalRunner(
    eval_args=eval_args,
    model_args=model_args
)

runner.run()
```

输出:

```
/root/anaconda3/envs/dev/lib/python3.12/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
  from .autonotebook import tqdm as notebook_tqdm
initial target device: 100%|██████████| 2/2 [00:09<00:00,  4.98s/it]
pre tokenize: 100%|██████████| 16062/16062 [18:01<00:00, 14.85it/s]  
You're using a BertTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
/root/anaconda3/envs/dev/lib/python3.12/site-packages/_distutils_hack/__init__.py:54: UserWarning: Reliance on distutils from stdlib is deprecated. Users must rely on setuptools to provide the distutils module. Avoid importing distutils or import setuptools first, and avoid setting SETUPTOOLS_USE_DISTUTILS=stdlib. Register concerns at https://github.com/pypa/setuptools/issues/new?template=distutils-deprecation.yml
  warnings.warn(
pre tokenize: 100%|██████████| 16062/16062 [18:44<00:00, 14.29it/s]92s/it]
Inference Embeddings:   0%|          | 42/16062 [00:54<8:28:19,  1.90s/it]You're using a BertTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
Inference Embeddings:   0%|          | 43/16062 [00:56<8:22:03,  1.88s/it]/root/anaconda3/envs/dev/lib/python3.12/site-packages/_distutils_hack/__init__.py:54: UserWarning: Reliance on distutils from stdlib is deprecated. Users must rely on setuptools to provide the distutils module. Avoid importing distutils or import setuptools first, and avoid setting SETUPTOOLS_USE_DISTUTILS=stdlib. Register concerns at https://github.com/pypa/setuptools/issues/new?template=distutils-deprecation.yml
  warnings.warn(
Inference Embeddings: 100%|██████████| 16062/16062 [48:29<00:00,  5.52it/s] 
Inference Embeddings: 100%|██████████| 16062/16062 [48:55<00:00,  5.47it/s]
Chunks: 100%|██████████| 2/2 [1:10:57<00:00, 2128.54s/it]  
pre tokenize: 100%|██████████| 1/1 [00:11<00:00, 11.06s/it]
pre tokenize: 100%|██████████| 1/1 [00:12<00:00, 12.72s/it]
Inference Embeddings: 100%|██████████| 1/1 [00:00<00:00, 32.15it/s]
Inference Embeddings: 100%|██████████| 1/1 [00:00<00:00, 39.80it/s]
Chunks: 100%|██████████| 2/2 [00:31<00:00, 15.79s/it]
Searching: 100%|██████████| 25/25 [00:00<00:00, 26.24it/s]
Qrels not found in ./miracl/data/en/dev_qrels.jsonl. Trying to download the qrels from the remote and save it to ./miracl/data/en.
--2024-11-20 13:00:40--  https://hf-mirror.com/datasets/miracl/miracl/resolve/main/miracl-v1.0-en/qrels/qrels.miracl-v1.0-en-dev.tsv
Resolving hf-mirror.com (hf-mirror.com)... 133.242.169.68, 153.121.57.40, 160.16.199.204
Connecting to hf-mirror.com (hf-mirror.com)|133.242.169.68|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 167817 (164K) [text/plain]
Saving to: ‘./cache/data/miracl/qrels.miracl-v1.0-en-dev.tsv’

     0K .......... .......... .......... .......... .......... 30%  336K 0s
    50K .......... .......... .......... .......... .......... 61%  678K 0s
   100K .......... .......... .......... .......... .......... 91%  362K 0s
   150K .......... ...                                        100% 39.8K=0.7s

2024-11-20 13:00:42 (231 KB/s) - ‘./cache/data/miracl/qrels.miracl-v1.0-en-dev.tsv’ saved [167817/167817]

Loading and Saving qrels: 100%|██████████| 8350/8350 [00:00<00:00, 184554.95it/s]

```

```python
with open('miracl/search_results/bge-base-en-v1.5/NoReranker/EVAL/eval_results.json', 'r') as content_file:
    print(content_file.read())
```

输出:

```
{
    "en-dev": {
        "ndcg_at_10": 0.46053,
        "ndcg_at_100": 0.54313,
        "map_at_10": 0.35928,
        "map_at_100": 0.38726,
        "recall_at_10": 0.55972,
        "recall_at_100": 0.83809,
        "precision_at_10": 0.14018,
        "precision_at_100": 0.02347,
        "mrr_at_10": 0.54328,
        "mrr_at_100": 0.54929
    }
}

```

