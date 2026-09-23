# C-MTEB

C-MTEB是中文文本嵌入的最大基准测试，类似于MTEB。在本教程中，我们将介绍如何评估嵌入模型在C-MTEB中文任务上的能力。

## 0. 安装

首先安装依赖包：

```python
%pip install FlagEmbedding mteb
```

## 1. 数据集

C-MTEB使用与英文MTEB类似的任务分类和指标。它包含6个不同任务中的35个数据集：分类、聚类、对分类、重排序、检索和语义文本相似度（STS）。

1. **分类**：使用嵌入在训练集上训练逻辑回归，并在测试集上进行评分。F1是主要指标。
2. **聚类**：使用批量大小为32的小批量k-means模型训练，k等于不同标签的数量。然后使用v-measure进行评分。
3. **对分类**：提供一对文本输入和需要被分配的二元变量标签。主要指标是平均精度分数。
4. **重排序**：根据查询对相关和无关参考文本列表进行排序。指标是平均MRR@k和MAP。
5. **检索**：每个数据集包括语料库、查询和将每个查询链接到语料库中相关文档的映射。目标是为每个查询检索相关文档。主要指标是nDCG@k。MTEB直接采用BEIR进行检索任务。
6. **语义文本相似度(STS)**：确定每对句子之间的相似度。基于余弦相似度的斯皮尔曼相关系数是主要指标。


查看[HF页面](https://huggingface.co/C-MTEB)了解每个数据集的详情。

```python
ChineseTaskList = [
    'TNews', 'IFlyTek', 'MultilingualSentiment', 'JDReview', 'OnlineShopping', 'Waimai',
    'CLSClusteringS2S.v2', 'CLSClusteringP2P.v2', 'ThuNewsClusteringS2S.v2', 'ThuNewsClusteringP2P.v2',
    'Ocnli', 'Cmnli',
    'T2Reranking', 'MMarcoReranking', 'CMedQAv1-reranking', 'CMedQAv2-reranking',
    'T2Retrieval', 'MMarcoRetrieval', 'DuRetrieval', 'CovidRetrieval', 'CmedqaRetrieval', 'EcomRetrieval', 'MedicalRetrieval', 'VideoRetrieval',
    'ATEC', 'BQ', 'LCQMC', 'PAWSX', 'STSB', 'AFQMC', 'QBQTC'
]
```

## 2. 模型

首先，加载要评估的模型。注意，这里的指令用于检索任务。

```python
from ...C_MTEB.flag_dres_model import FlagDRESModel

instruction = "为这个句子生成表示以用于检索相关文章："
model_name = "BAAI/bge-base-zh-v1.5"

model = FlagDRESModel(model_name_or_path="BAAI/bge-base-zh-v1.5",
                      query_instruction_for_retrieval=instruction,
                      pooling_method="cls")
```

或者，您可以使用sentence_transformers加载模型：

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("PATH_TO_MODEL")
```

或者实现一个遵循以下结构的类：

```python
class MyModel():
    def __init__(self):
        """初始化分词器和模型"""
        pass

    def encode(self, sentences, batch_size=32, **kwargs):
        """ 返回给定句子的嵌入列表
        参数:
            sentences (`List[str]`): 要编码的句子列表
            batch_size (`int`): 编码的批大小

        返回:
            `List[np.ndarray]` 或 `List[tensor]`: 给定句子的嵌入列表
        """
        pass

model = MyModel()
```

## 3. 评估

在我们准备好数据集和模型后，我们可以开始评估。为了提高时间效率，我们强烈建议使用GPU进行评估。

```python
import mteb
from mteb import MTEB

tasks = mteb.get_tasks(ChineseTaskList)

for task in tasks:
    evaluation = MTEB(tasks=[task])
    evaluation.run(model, output_folder=f"zh_results/{model_name.split('/')[-1]}")
```

## 4. 提交到MTEB排行榜

评估完成后，所有评估结果应存储在`zh_results/{model_name}/`中。

然后运行以下shell命令创建model_card.md。将{model_name}及其后面的内容更改为您的路径。

```python
!!mteb create_meta --results_folder results/{model_name}/ --output_path model_card.md
```

将model_card.md的内容复制并粘贴到您在HF Hub上的模型README.md的顶部。然后前往[MTEB排行榜](https://huggingface.co/spaces/mteb/leaderboard)并选择中文排行榜来找到您的模型！它将在网站每日刷新后很快出现。

