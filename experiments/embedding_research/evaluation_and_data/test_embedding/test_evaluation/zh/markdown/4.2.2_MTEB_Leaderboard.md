# MTEB排行榜

在上一个教程中，我们展示了如何在MTEB支持的数据集上评估嵌入模型。在本教程中，我们将介绍如何进行完整评估，并将结果与MTEB英语排行榜进行比较。

注意：即使使用GPU，在完整的英文MTEB上进行评估也非常耗时。因此，我们建议您通过本教程来了解具体流程。当您有足够的计算资源和时间时再运行实验。

## 0. 安装

在您的环境中安装我们将使用的包：

```python
%%capture
%pip install sentence_transformers mteb
```

## 1. 运行评估

MTEB英语排行榜包含7个任务的56个数据集：
1. **分类**：使用嵌入在训练集上训练逻辑回归，并在测试集上进行评分。F1是主要指标。
2. **聚类**：使用批量大小为32的小批量k-means模型训练，k等于不同标签的数量。然后使用v-measure进行评分。
3. **对分类**：提供一对文本输入和需要被分配的二元变量标签。主要指标是平均精度分数。
4. **重排序**：根据查询对相关和无关参考文本列表进行排序。指标是平均MRR@k和MAP。
5. **检索**：每个数据集包括语料库、查询和将每个查询链接到语料库中相关文档的映射。目标是为每个查询检索相关文档。主要指标是nDCG@k。MTEB直接采用BEIR进行检索任务。
6. **语义文本相似度(STS)**：确定每对句子之间的相似度。基于余弦相似度的斯皮尔曼相关系数是主要指标。
7. **摘要**：该任务只使用1个数据集。通过计算机器生成摘要与人工编写摘要的嵌入距离来评分。主要指标也是基于余弦相似度的斯皮尔曼相关系数。

该基准测试已被研究人员和工程师广泛接受，用来公平评估和比较他们训练的模型的表现。现在让我们来看看整个评估流程

导入`MTEB_MAIN_EN`以查看所有56个数据集。

```python
import mteb
from mteb.benchmarks import MTEB_MAIN_EN

print(MTEB_MAIN_EN.tasks)
```

输出:

```
['AmazonCounterfactualClassification', 'AmazonPolarityClassification', 'AmazonReviewsClassification', 'ArguAna', 'ArxivClusteringP2P', 'ArxivClusteringS2S', 'AskUbuntuDupQuestions', 'BIOSSES', 'Banking77Classification', 'BiorxivClusteringP2P', 'BiorxivClusteringS2S', 'CQADupstackAndroidRetrieval', 'CQADupstackEnglishRetrieval', 'CQADupstackGamingRetrieval', 'CQADupstackGisRetrieval', 'CQADupstackMathematicaRetrieval', 'CQADupstackPhysicsRetrieval', 'CQADupstackProgrammersRetrieval', 'CQADupstackStatsRetrieval', 'CQADupstackTexRetrieval', 'CQADupstackUnixRetrieval', 'CQADupstackWebmastersRetrieval', 'CQADupstackWordpressRetrieval', 'ClimateFEVER', 'DBPedia', 'EmotionClassification', 'FEVER', 'FiQA2018', 'HotpotQA', 'ImdbClassification', 'MSMARCO', 'MTOPDomainClassification', 'MTOPIntentClassification', 'MassiveIntentClassification', 'MassiveScenarioClassification', 'MedrxivClusteringP2P', 'MedrxivClusteringS2S', 'MindSmallReranking', 'NFCorpus', 'NQ', 'QuoraRetrieval', 'RedditClustering', 'RedditClusteringP2P', 'SCIDOCS', 'SICK-R', 'STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STS17', 'STS22', 'STSBenchmark', 'SciDocsRR', 'SciFact', 'SprintDuplicateQuestions', 'StackExchangeClustering', 'StackExchangeClusteringP2P', 'StackOverflowDupQuestions', 'SummEval', 'TRECCOVID', 'Touche2020', 'ToxicConversationsClassification', 'TweetSentimentExtractionClassification', 'TwentyNewsgroupsClustering', 'TwitterSemEval2015', 'TwitterURLCorpus']

```

加载我们想要评估的模型：

```python
from sentence_transformers import SentenceTransformer

model_name = "BAAI/bge-base-en-v1.5"
model = SentenceTransformer(model_name)
```

或者，MTEB提供了排行榜上的流行模型，以便复现它们的结果。

```python
model_name = "BAAI/bge-base-en-v1.5"
model = mteb.get_model(model_name)
```

然后开始在每个数据集上评估：

```python
for task in MTEB_MAIN_EN.tasks:
    # get the test set to evaluate on
    eval_splits = ["dev"] if task == "MSMARCO" else ["test"]
    evaluation = mteb.MTEB(
        tasks=[task], task_langs=["en"]
    )  # Remove "en" to run all available languages
    evaluation.run(
        model, output_folder="results", eval_splits=eval_splits
    )
```

## 2. 提交到MTEB排行榜

评估完成后，所有评估结果都应该存储在`results/{model_name}/{model_revision}`中。

然后运行以下shell命令创建model_card.md。将{model_name}和{model_revision}更改为您的路径。

```python
!mteb create_meta --results_folder results/{model_name}/{model_revision} --output_path model_card.md
```

如果该模型的readme已经存在：

```python
# !mteb create_meta --results_folder results/{model_name}/{model_revision} --output_path model_card.md --from_existing your_existing_readme.md 
```

将model_card.md的内容复制并粘贴到您的模型在HF Hub上的README.md的顶部。现在放松一下，等待排行榜的每日刷新。您的模型很快就会出现！

## 3. 部分评估

请注意，您不需要完成所有任务就能进入排行榜。

例如，您微调了模型在聚类上的能力。您只关心模型在聚类方面的表现，而不是其他任务。那么您可以只测试它在MTEB的聚类任务上的表现，并提交到排行榜。

```python
TASK_LIST_CLUSTERING = [
    "ArxivClusteringP2P",
    "ArxivClusteringS2S",
    "BiorxivClusteringP2P",
    "BiorxivClusteringS2S",
    "MedrxivClusteringP2P",
    "MedrxivClusteringS2S",
    "RedditClustering",
    "RedditClusteringP2P",
    "StackExchangeClustering",
    "StackExchangeClusteringP2P",
    "TwentyNewsgroupsClustering",
]
```

只运行聚类任务的评估：

```python
evaluation = mteb.MTEB(tasks=TASK_LIST_CLUSTERING)

results = evaluation.run(model, output_folder="results")
```

然后重复步骤2提交您的模型。排行榜刷新后，您可以在排行榜的"聚类"部分找到您的模型。

## 4. 未来工作

MTEB正在开发新版本的英语基准测试。它包含更新和简洁的任务，将使评估过程更快。

请查看他们的[GitHub](https://github.com/embeddings-benchmark/mteb)页面，了解未来更新和发布。

