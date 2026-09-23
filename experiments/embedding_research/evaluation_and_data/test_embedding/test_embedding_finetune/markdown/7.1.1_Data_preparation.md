# 微调数据准备

在本教程中，我们将展示微调第一步的示例：数据集准备。

## 0. 安装

```python
% pip install -U datasets
```

假设我们希望针对财务任务微调模型。我们找到了一个可能有用的开源数据集：[financial-qa-10k](https://huggingface.co/datasets/virattt/financial-qa-10K)。让我们看看如何正确准备用于微调的数据集。

原始数据集具有以下结构：

- 5 列：'question'、'answer'、'context'、'ticker' 和 'filing'。
- 7000 行。

```python
from datasets import load_dataset

ds = load_dataset("virattt/financial-qa-10K", split="train")
ds
```

结果:

```text
Dataset({
    features: ['question', 'answer', 'context', 'ticker', 'filing'],
    num_rows: 7000
})
```

## 1. 微调数据

将数据集构建为以下格式：

```python
{"query": str, "pos": List[str], "neg":List[str], "pos_scores": List[int], "neg_scores": List[int], "prompt": str, "type": str}
```

`query` 是查询，`pos` 是正向文本列表，`neg` 是负向文本列表。`pos_scores` 是对应于查询和正向文本的分数列表，`neg_scores` 是对应于查询和负向文本的分数列表，如果不使用知识蒸馏，则可以忽略它们。`prompt` 是用于查询的提示，它将覆盖 query_instruction_for_retrieval。`type` 用于 bge-en-icl，它包括 `normal`、`symmetric_class`、`symmetric_clustering` 等。如果查询没有负向文本，可以从整个语料库中随机抽样一些作为负向文本。

我们选择“question”和“context”列作为查询和答案（正向），并重命名这些列。然后添加“id”列以供以后评估使用。

```python
ds = ds.select_columns(column_names=["question", "context"])
ds = ds.rename_column("question", "query")
ds = ds.rename_column("context", "pos")
ds = ds.add_column("id", [str(i) for i in range(len(ds))])
ds[0]
```

结果:

```text
{'query': 'NVIDIA 在扩展到其他计算密集型领域之前最初专注于哪个领域？',
 'pos': '自从我们最初专注于 PC 图形以来，我们已经扩展到其他几个大型且重要的计算密集型领域。',
 'id': '0'}
```

在嵌入模型训练期间，负样本非常重要。我们的初始数据集不包含负向文本。因此，我们直接从整个语料库中抽样一些。

```python
import numpy as np

np.random.seed(520)
neg_num = 10

def str_to_lst(data):
    data["pos"] = [data["pos"]]
    return data

# 抽样负向文本
new_col = []
for i in range(len(ds)):
    ids = np.random.randint(0, len(ds), size=neg_num)
    while i in ids:
        ids = np.random.randint(0, len(ds), size=neg_num)
    neg = [ds[i.item()]["pos"] for i in ids]
    new_col.append(neg)
ds = ds.add_column("neg", new_col)

# 将 'pos' 的键更改为列表
ds = ds.map(str_to_lst)
```

输出:

```text
Map: 100%|██████████| 7000/7000 [00:00<00:00, 22336.83 examples/s]

```

最后，我们添加用于查询的提示。它将是推理期间的 `query_instruction_for_retrieval`。

```python
instruction = "Represent this sentence for searching relevant passages: "
ds = ds.add_column("prompt", [instruction]*len(ds))
```

现在数据集的单行是：

```python
ds[0]
```

结果:

```text
{'query': 'NVIDIA 在扩展到其他计算密集型领域之前最初专注于哪个领域？',
 'pos': ['自从我们最初专注于 PC 图形以来，我们已经扩展到其他几个大型且重要的计算密集型领域。'],
 'id': '0',
 'neg': ['Kroger 预计其价值创造模型将随着时间的推移实现 8% 至 11% 的目标股东总回报率。',
  'CSB 在 2023 年购买了 29 亿美元的第一抵押贷款。',
  '有关某些存在或有事项的法律程序的更多信息，请参阅我们合并财务报表的附注 13。',
  '2022 财年的稀释每股收益为 16.69 美元，而 2021 财年为 15.53 美元。',
  '截至 2023 年 12 月 31 日的年度，总净销售额和收入主要由于以下原因增加：(1) 净批发量增加，主要原因是跨界车和全尺寸皮卡销量增加，部分被中型皮卡销量下降所抵消；(2) 由于经销商库存水平低和对我们产品的强劲需求导致有利的价格；(3) 由于全尺寸皮卡和全尺寸 SUV 销量增加以及厢式货车、乘用车和中型皮卡销量下降导致有利的组合，部分被跨界车销量增加所抵消；以及 (4) 由于零部件和配件销量增加导致有利的其他因素。',
  '截至 2023 年 12 月 31 日，我们拥有 3,157 名全职员工。',
  '第 3 项。法律程序。本 10-K 表格第 8 项中包含的附注 18“承诺和或有事项”中包含的信息通过引用并入本文。',
  '根据修订后的 2019 年担保融资协议，到期日设定为 2026 年 7 月 20 日。',
  '截至 2023 年 12 月 31 日，拉斯维加斯金沙集团的应收账款总额为 6.85 亿美元，信贷损失准备金为 2.01 亿美元，净余额为 4.84 亿美元。',
  '与上一财年相比，2023 财年运营费用占分部净销售额的百分比下降了 25 个基点，这主要是由于强劲的销售增长和较低的增量 COVID-19 相关成本，部分被工资成本增加所抵消。'],
 'prompt': 'Represent this sentence for searching relevant passages: '}
```

然后我们将数据集拆分为训练集和测试集。

```python
split = ds.train_test_split(test_size=0.1, shuffle=True, seed=520)
train = split["train"]
test = split["test"]
```

现在我们准备存储数据以供以后微调：

```python
train.to_json("ft_data/training.json")
```

输出:

```text
Creating json from Arrow format: 100%|██████████| 7/7 [00:00<00:00, 39.73ba/s]

```

结果:

```text
16583481
```

## 2. 用于评估的测试数据

最后一步是构建用于评估的测试数据集。

```python
test
```

结果:

```text
Dataset({
    features: ['query', 'pos', 'id', 'neg', 'prompt'],
    num_rows: 700
})
```

首先选择查询的列：

```python
queries = test.select_columns(column_names=["id", "query"])
queries = queries.rename_column("query", "text")
queries[0]
```

结果:

```text
{'id': '1289',
 'text': '星巴克如何在财务报表中确认与所得税事宜相关的利息和罚款？'}
```

然后选择语料库的列：

```python
corpus = ds.select_columns(column_names=["id", "pos"])
corpus = corpus.rename_column("pos", "text")
```

最后，创建指示查询和相应语料库之间关系的 qrels：

```python
qrels = test.select_columns(["id"])
qrels = qrels.rename_column("id", "qid")
qrels = qrels.add_column("docid", list(test["id"]))
qrels = qrels.add_column("relevance", [1]*len(test))
qrels[0]
```

输出:

```text
Flattening the indices: 100%|██████████| 700/700 [00:00<00:00, 180956.10 examples/s]

```

结果:

```text
{'qid': '1289', 'docid': '1289', 'relevance': 1}
```

存储训练集

```python
queries.to_json("ft_data/test_queries.jsonl")
corpus.to_json("ft_data/corpus.jsonl")
qrels.to_json("ft_data/test_qrels.jsonl")
```

输出:

```text
Creating json from Arrow format: 100%|██████████| 1/1 [00:00<00:00, 210.42ba/s]
Creating json from Arrow format: 100%|██████████| 7/7 [00:00<00:00, 261.19ba/s]
Creating json from Arrow format: 100%|██████████| 1/1 [00:00<00:00, 591.08ba/s]

```

结果:

```text
30574
```

