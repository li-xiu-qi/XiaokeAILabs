# 在BEIR上评估

[BEIR](https://github.com/beir-cellar/beir)（信息检索基准测试）是一个用于信息检索的异构评估基准。
它旨在评估基于NLP的检索模型的性能，并被现代嵌入模型研究广泛使用。

## 0. 安装

首先安装我们将使用的库：

```python
% pip install beir FlagEmbedding
```

## 1. 使用BEIR进行评估

BEIR包含18个数据集，可以从[链接](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/)下载，其中4个是需要适当许可证的私有数据集。如果你想访问这4个数据集，请查看他们的[wiki](https://github.com/beir-cellar/beir/wiki/Datasets-available)获取更多信息。以下信息和代码来源于BEIR的GitHub[仓库](https://github.com/beir-cellar/beir)。

| 数据集名称 | 类型     |  查询数  | 文档数 | 平均文档/查询 | 是否公开 | 
| ---------| :-----------: | ---------| --------- | ------| :------------:| 
| ``msmarco`` | `训练` `开发` `测试` | 6,980   |  8.84M     |    1.1 | 是 |  
| ``trec-covid``| `测试` | 50|  171K| 493.5 | 是 | 
| ``nfcorpus``  | `训练` `开发` `测试` |  323     |  3.6K     |  38.2 | 是 |
| ``bioasq``| `训练` `测试` |    500    |  14.91M    |  8.05 | 否 | 
| ``nq``| `训练` `测试`   |  3,452   |  2.68M  |  1.2 | 是 | 
| ``hotpotqa``| `训练` `开发` `测试`   |  7,405   |  5.23M  |  2.0 | 是 |
| ``fiqa``    | `训练` `开发` `测试`     |  648     |  57K    |  2.6 | 是 | 
| ``signal1m`` | `测试`     |   97   |  2.86M  |  19.6 | 否 |
| ``trec-news``    | `测试`     |   57    |  595K    |  19.6 | 否 |
| ``arguana`` | `测试`       |  1,406     |  8.67K    |  1.0 | 是 |
| ``webis-touche2020``| `测试` |   49     |  382K    |  49.2 |  是 |
| ``cqadupstack``| `测试`      |   13,145 |  457K  |  1.4 |  是 |
| ``quora``| `开发` `测试`  |   10,000     |  523K    |  1.6 |  是 | 
| ``dbpedia-entity``| `开发` `测试` |   400    |  4.63M    |  38.2 |  是 | 
| ``scidocs``| `测试` |    1,000     |  25K    |  4.9 |  是 | 
| ``fever``| `训练` `开发` `测试`     |   6,666     |  5.42M    |  1.2|  是 | 
| ``climate-fever``| `测试` |  1,535     |  5.42M |  3.0 |  是 |
| ``scifact``| `训练` `测试` |  300     |  5K    |  1.1 |  是 |

### 1.1 加载数据集

首先设置日志。

```python
import logging
from beir import LoggingHandler

logging.basicConfig(format='%(message)s',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
```

在这个演示中，我们选择`arguana`数据集进行快速演示。

```python
import os
from beir import util

url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/arguana.zip"
out_dir = os.path.join(os.getcwd(), "data")
data_path = util.download_and_unzip(url, out_dir)
print(f"数据集存储在: {data_path}")
```

输出:

```
数据集下载到: /share/project/xzy/Projects/FlagEmbedding/Tutorials/4_Evaluation/data/arguana

```

```python
from beir.datasets.data_loader import GenericDataLoader

corpus, queries, qrels = GenericDataLoader("data/arguana").load(split="test")
```

输出:

```
2024-11-15 03:54:55,809 - 正在加载语料库...

```

输出:

```
100%|██████████| 8674/8674 [00:00<00:00, 158928.31it/s]
```

输出:

```
2024-11-15 03:54:55,891 - 已加载8674个测试文档。
2024-11-15 03:54:55,891 - 文档示例: {'text': "你不必成为素食主义者才能环保。许多特殊环境是由畜牧业创造的——例如英国的白垩质草原和许多国家的山地牧场。终止畜牧业会使这些地区恢复为森林，导致许多独特植物和动物的丧失。种植农作物也可能对地球非常不利，肥料和农药会污染河流、湖泊和海洋。现在大多数热带森林被砍伐是为了木材，或者为了种植油棕树，而不是为了肉类生产创造空间。  英国农民和前编辑西蒙·法雷尔也表示："许多素食者和素食主义者依赖联合国的一个数据，即畜牧业产生全球18%的碳排放，但这个数字包含基本错误。它将所有与牧场相关的森林砍伐都归因于牛，而不是砍伐或开发。它还混淆了森林砍伐的一次性排放和持续污染。"  他还反驳了肉类生产效率低下的说法："科学家们已经计算出，全球用于生产肉类的有用植物食品比例约为5:1。如果你只用人类可以吃的食物喂养动物——这在西方世界确实如此——那可能是对的。但动物也吃我们不能吃的食物，比如草。所以真正的转换数字是1.4:1。" [1] 同时，如果素食饮食不是可持续的或使用从世界各地空运的易腐水果和蔬菜，那可能并不比肉类饮食更环保。吃当地生产的食物可能与素食一样对环境有重大影响。[2]  [1] Tara Kelly, Simon Fairlie: 如何通过吃肉拯救世界，2010年10月12日  [2] Lucy Siegle, '是时候成为素食者了吗？'《观察家报》，2008年5月18日", 'title': '动物 环境 一般 健康 健康 一般 体重 哲学 伦理学'}
2024-11-15 03:54:55,891 - 正在加载查询...
2024-11-15 03:54:55,903 - 已加载1406个测试查询。
2024-11-15 03:54:55,903 - 查询示例: 素食有助于环境  成为素食主义者是环保的行为。现代农业是我们河流中主要污染源之一。牛肉养殖是森林砍伐的主要原因之一，只要人们继续购买数十亿快餐，就会有财政激励继续砍伐树木以为牛创造空间。因为我们想吃鱼，我们的河流和海洋正被耗尽鱼类，许多物种面临灭绝。肉类养殖比谷物、豆类等农业消耗更多能源资源。食用肉类和鱼类不仅对动物造成残忍，还对环境和生物多样性造成严重伤害。例如考虑肉类生产相关的污染和森林砍伐  在多伦多1992年皇家农业冬季博览会上，加拿大农业部展示了两个对比鲜明的统计数据："需要四个足球场大小的土地（约1.6公顷）来养活每个加拿大人"和"一棵苹果树产生足够的水果制作320个馅饼。"想想看——几棵苹果树和几排小麦仅在不到一公顷的土地上就能为一个人提供足够食物！[1]  2006年联合国粮食及农业组织（FAO）的报告得出结论，全球畜牧业产生地球18%的温室气体排放——相比之下，全球所有汽车、火车、飞机和船只加起来只占温室气体排放的13%。[2]  由于上述观点，生产肉类破坏环境。对肉的需求导致森林砍伐。巴西联邦公共检察办公室的Daniel Cesar Avelino说："我们知道亚马逊森林砍伐的最大驱动力是牛。"据估计，为农业而清除亚马逊等热带雨林产生了世界17%的温室气体排放。[3] 不仅如此，肉类生产消耗的能量远远超过它最终提供给我们的能量，鸡肉生产的能量与蛋白质输出比为4:1；牛肉生产需要54:1的能量投入与蛋白质输出比例。  水的使用也是如此，由于生产相同重量的肉需要大量谷物，生产需要大量水。水是另一种我们很快就会在全球各地不足的资源。谷物饲养的牛肉生产每公斤食物需要100,000升水。养肉鸡每公斤肉需要3,500升水。相比之下，大豆生产每公斤食品使用2,000升；水稻1,912升；小麦900升；土豆500升。[4] 而全球有些地区正遭受严重水资源短缺。农业用水量是家庭用水（烹饪和洗涤）的70倍。世界三分之一的人口已经遭受水资源短缺。[5] 全球地下水位正在下降，河流开始干涸。中国黄河这样的大河已经不再流入大海。[6]  随着人口增加，成为素食主义者是唯一负责任的饮食方式。  [1] Stephen Leckie，'以肉为中心的饮食模式如何影响粮食安全和环境'，国际开发研究中心  [2] Bryan Walsh, 肉类：使全球变暖更糟糕，《时代》杂志，2008年9月10日.  [3] David Adam，超市供应商'帮助摧毁亚马逊雨林'，《卫报》，2009年6月21日。  [4] Roger Segelken，美国可以用喂给牲畜的谷物养活8亿人口，康奈尔科学新闻，1997年8月7日。  [5] Fiona Harvey，水资源短缺影响三分之一人口，FT.com，2003年8月21日  [6] Rupert Wingfield-Hayes，黄河'干涸'，BBC新闻，2004年7月29日

```

输出:

```


```

### 1.2 评估

然后我们从huggingface加载`bge-base-en-v1.5`并评估其在arguana上的性能。

```python
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval import models
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES


# 使用Sentence Transformers加载bge模型
model = DRES(models.SentenceBERT("BAAI/bge-base-en-v1.5"), batch_size=128)
retriever = EvaluateRetrieval(model, score_function="cos_sim")

# 获取搜索结果
results = retriever.retrieve(corpus, queries)
```

输出:

```
2024-11-15 04:00:45,253 - 使用pytorch设备名称: cuda
2024-11-15 04:00:45,254 - 加载预训练SentenceTransformer: BAAI/bge-base-en-v1.5
2024-11-15 04:00:48,750 - 正在编码查询...

```

输出:

```
批次: 100%|██████████| 11/11 [00:01<00:00,  8.27it/s]

```

输出:

```
2024-11-15 04:00:50,177 - 根据文档长度排序语料库（最长优先）...
2024-11-15 04:00:50,183 - 批量编码语料库... 警告：这可能需要一段时间！
2024-11-15 04:00:50,183 - 评分函数: 余弦相似度 (cos_sim)
2024-11-15 04:00:50,184 - 正在编码批次 1/1...

```

输出:

```
批次: 100%|██████████| 68/68 [00:07<00:00,  9.43it/s]

```

```python
logging.info("检索器评估 k 值: {}".format(retriever.k_values))
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
```

输出:

```
2024-11-15 04:00:58,514 - 检索器评估 k 值: [1, 3, 5, 10, 100, 1000]
2024-11-15 04:00:58,514 - 为进行评估，我们忽略相同的查询和文档ID（默认），如果想要忽略此设置，请明确设置``ignore_identical_ids=False``。
2024-11-15 04:00:59,184 - 

2024-11-15 04:00:59,188 - NDCG@1: 0.4075
2024-11-15 04:00:59,188 - NDCG@3: 0.5572
2024-11-15 04:00:59,188 - NDCG@5: 0.5946
2024-11-15 04:00:59,188 - NDCG@10: 0.6361
2024-11-15 04:00:59,188 - NDCG@100: 0.6606
2024-11-15 04:00:59,188 - NDCG@1000: 0.6613
2024-11-15 04:00:59,188 - 

2024-11-15 04:00:59,188 - MAP@1: 0.4075
2024-11-15 04:00:59,188 - MAP@3: 0.5193
2024-11-15 04:00:59,188 - MAP@5: 0.5402
2024-11-15 04:00:59,188 - MAP@10: 0.5577
2024-11-15 04:00:59,188 - MAP@100: 0.5634
2024-11-15 04:00:59,188 - MAP@1000: 0.5635
2024-11-15 04:00:59,188 - 

2024-11-15 04:00:59,188 - Recall@1: 0.4075
2024-11-15 04:00:59,188 - Recall@3: 0.6671
2024-11-15 04:00:59,188 - Recall@5: 0.7575
2024-11-15 04:00:59,188 - Recall@10: 0.8841
2024-11-15 04:00:59,188 - Recall@100: 0.9915
2024-11-15 04:00:59,189 - Recall@1000: 0.9964
2024-11-15 04:00:59,189 - 

2024-11-15 04:00:59,189 - P@1: 0.4075
2024-11-15 04:00:59,189 - P@3: 0.2224
2024-11-15 04:00:59,189 - P@5: 0.1515
2024-11-15 04:00:59,189 - P@10: 0.0884
2024-11-15 04:00:59,189 - P@100: 0.0099
2024-11-15 04:00:59,189 - P@1000: 0.0010

```

## 2. 使用FlagEmbedding进行评估

我们为流行的数据集和基准测试提供独立的评估。尝试以下代码运行评估，或运行[example](../../examples/evaluation/beir/eval_beir.sh)文件夹中提供的shell脚本。

加载参数：

```python
import sys

arguments = """-
    --eval_name beir 
    --dataset_dir ./beir/data 
    --dataset_names arguana
    --splits test dev 
    --corpus_embd_save_dir ./beir/corpus_embd 
    --output_dir ./beir/search_results 
    --search_top_k 1000 
    --rerank_top_k 100 
    --cache_path /root/.cache/huggingface/hub 
    --overwrite True 
    --k_values 10 100 
    --eval_output_method markdown 
    --eval_output_path ./beir/beir_eval_results.md 
    --eval_metrics ndcg_at_10 recall_at_100 
    --ignore_identical_ids True 
    --embedder_name_or_path BAAI/bge-base-en-v1.5 
    --embedder_batch_size 1024
    --devices cuda:4
""".replace('\n','')

sys.argv = arguments.split()
```

然后将参数传递给HFArgumentParser并运行评估。

```python
from transformers import HfArgumentParser

from FlagEmbedding.evaluation.beir import (
    BEIREvalArgs, BEIREvalModelArgs,
    BEIREvalRunner
)


parser = HfArgumentParser((
    BEIREvalArgs,
    BEIREvalModelArgs
))

eval_args, model_args = parser.parse_args_into_dataclasses()
eval_args: BEIREvalArgs
model_args: BEIREvalModelArgs

runner = BEIREvalRunner(
    eval_args=eval_args,
    model_args=model_args
)

runner.run()
```

输出:

```
在数据集中未找到Split 'dev'。从列表中移除。
ignore_identical_ids设置为True。这意味着搜索结果不会包含相同的ID。注意：MIRACL等数据集不应该将此设置为True。
预标记化: 100%|██████████| 9/9 [00:00<00:00, 16.19it/s]
你正在使用BertTokenizerFast分词器。请注意，使用快速分词器时，使用`__call__`方法比使用编码文本的方法后跟调用`pad`方法获取填充编码更快。
推理嵌入: 100%|██████████| 9/9 [00:11<00:00,  1.27s/it]
预标记化: 100%|██████████| 2/2 [00:00<00:00, 19.54it/s]
推理嵌入: 100%|██████████| 2/2 [00:02<00:00,  1.29s/it]
搜索: 100%|██████████| 44/44 [00:00<00:00, 208.73it/s]

```

看看结果并选择你喜欢的方式！

```python
with open('beir/search_results/bge-base-en-v1.5/NoReranker/EVAL/eval_results.json', 'r') as content_file:
    print(content_file.read())
```

输出:

```
{
    "arguana-test": {
        "ndcg_at_10": 0.63668,
        "ndcg_at_100": 0.66075,
        "map_at_10": 0.55801,
        "map_at_100": 0.56358,
        "recall_at_10": 0.88549,
        "recall_at_100": 0.99147,
        "precision_at_10": 0.08855,
        "precision_at_100": 0.00991,
        "mrr_at_10": 0.55809,
        "mrr_at_100": 0.56366
    }
}

```

