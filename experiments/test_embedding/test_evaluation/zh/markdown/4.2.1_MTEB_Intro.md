# MTEB

对于嵌入模型的评估，MTEB是最著名的基准之一。在本教程中，我们将介绍MTEB、它的基本用法，并评估您的模型在MTEB排行榜上的表现。

## 0. 安装

在您的环境中安装我们将使用的包：

```python
%%capture
%pip install sentence_transformers mteb
```

## 1. 简介

[大规模文本嵌入基准测试(MTEB)](https://github.com/embeddings-benchmark/mteb)是一个大规模评估框架，旨在评估文本嵌入模型在各种自然语言处理(NLP)任务中的表现。MTEB的引入是为了标准化和改进文本嵌入的评估，对于评估这些模型在各种真实应用中的泛化能力至关重要。它包含了八个主要NLP任务和不同语言的广泛数据集，并提供了简单的评估流程。

MTEB也以MTEB排行榜而闻名，该排行榜包含了最新一流嵌入模型的排名。我们将在下一个教程中介绍这一点。现在让我们看看如何使用MTEB进行简单的评估。

```python
import mteb
from sentence_transformers import SentenceTransformer
```

现在我们来看看如何使用MTEB进行快速评估。

首先，我们加载要评估的模型：

```python
model_name = "BAAI/bge-base-en-v1.5"
model = SentenceTransformer(model_name)
```

以下是MTEB英文排行榜使用的检索数据集列表。

MTEB在其检索部分直接使用了开源基准BEIR，其中包含15个数据集（注意CQADupstack有12个子集）。

```python
retrieval_tasks = [
    "ArguAna",
    "ClimateFEVER",
    "CQADupstackAndroidRetrieval",
    "CQADupstackEnglishRetrieval",
    "CQADupstackGamingRetrieval",
    "CQADupstackGisRetrieval",
    "CQADupstackMathematicaRetrieval",
    "CQADupstackPhysicsRetrieval",
    "CQADupstackProgrammersRetrieval",
    "CQADupstackStatsRetrieval",
    "CQADupstackTexRetrieval",
    "CQADupstackUnixRetrieval",
    "CQADupstackWebmastersRetrieval",
    "CQADupstackWordpressRetrieval",
    "DBPedia",
    "FEVER",
    "FiQA2018",
    "HotpotQA",
    "MSMARCO",
    "NFCorpus",
    "NQ",
    "QuoraRetrieval",
    "SCIDOCS",
    "SciFact",
    "Touche2020",
    "TRECCOVID",
]
```

为了演示，我们只运行第一个："ArguAna"。

有关MTEB支持的任务和语言的完整列表，请查看[此页面](https://github.com/embeddings-benchmark/mteb/blob/18662380f0f476db3d170d0926892045aa9f74ee/docs/tasks.md)。

```python
tasks = mteb.get_tasks(tasks=retrieval_tasks[:1])
```

然后，使用我们选择的任务创建并初始化一个MTEB实例，并运行评估流程。

```python
# use the tasks we chose to initialize the MTEB instance
evaluation = mteb.MTEB(tasks=tasks)

# call run() with the model and output_folder
results = evaluation.run(model, output_folder="results")
```

结果:

```
[38;5;235m───────────────────────────────────────────────── [0m[1mSelected tasks [0m[38;5;235m ─────────────────────────────────────────────────[0m

```

结果:

```
[1mRetrieval[0m

```

结果:

```
    - ArguAna, [3;38;5;241ms2p[0m

```

结果:

```



```

输出:

```
Batches: 100%|██████████| 44/44 [00:41<00:00,  1.06it/s]
Batches: 100%|██████████| 272/272 [03:36<00:00,  1.26it/s]

```

结果应该存储在`{output_folder}/{model_name}/{model_revision}/{task_name}.json`中。

打开json文件，您应该会看到如下内容，这是使用不同指标在1到1000的截断点上对"ArguAna"进行评估的结果。

```python
{
  "dataset_revision": "c22ab2a51041ffd869aaddef7af8d8215647e41a",
  "evaluation_time": 260.14976954460144,
  "kg_co2_emissions": null,
  "mteb_version": "1.14.17",
  "scores": {
    "test": [
      {
        "hf_subset": "default",
        "languages": [
          "eng-Latn"
        ],
        "main_score": 0.63616,
        "map_at_1": 0.40754,
        "map_at_10": 0.55773,
        "map_at_100": 0.56344,
        "map_at_1000": 0.56347,
        "map_at_20": 0.56202,
        "map_at_3": 0.51932,
        "map_at_5": 0.54023,
        "mrr_at_1": 0.4139402560455192,
        "mrr_at_10": 0.5603739077423295,
        "mrr_at_100": 0.5660817425350153,
        "mrr_at_1000": 0.5661121884705748,
        "mrr_at_20": 0.564661930998293,
        "mrr_at_3": 0.5208629682313899,
        "mrr_at_5": 0.5429113323850182,
        "nauc_map_at_1000_diff1": 0.15930478114759905,
        "nauc_map_at_1000_max": -0.06396189194646361,
        "nauc_map_at_1000_std": -0.13168797291549253,
        "nauc_map_at_100_diff1": 0.15934819555197366,
        "nauc_map_at_100_max": -0.06389635013430676,
        "nauc_map_at_100_std": -0.13164524259533786,
        "nauc_map_at_10_diff1": 0.16057318234658585,
        "nauc_map_at_10_max": -0.060962623117325254,
        "nauc_map_at_10_std": -0.1300413865104607,
        "nauc_map_at_1_diff1": 0.17346152653542332,
        "nauc_map_at_1_max": -0.09705499215630589,
        "nauc_map_at_1_std": -0.14726476953035533,
        "nauc_map_at_20_diff1": 0.15956349246366208,
        "nauc_map_at_20_max": -0.06259296677860492,
        "nauc_map_at_20_std": -0.13097093150054095,
        "nauc_map_at_3_diff1": 0.15620049317363813,
        "nauc_map_at_3_max": -0.06690213479396273,
        "nauc_map_at_3_std": -0.13440904793529648,
        "nauc_map_at_5_diff1": 0.1557795701081579,
        "nauc_map_at_5_max": -0.06255283252590663,
        "nauc_map_at_5_std": -0.1355361594910923,
        "nauc_mrr_at_1000_diff1": 0.1378988612808882,
        "nauc_mrr_at_1000_max": -0.07507962333910836,
        "nauc_mrr_at_1000_std": -0.12969109830101241,
        "nauc_mrr_at_100_diff1": 0.13794450668758515,
        "nauc_mrr_at_100_max": -0.07501290390362861,
        "nauc_mrr_at_100_std": -0.12964855554504057,
        "nauc_mrr_at_10_diff1": 0.1396047981645623,
        "nauc_mrr_at_10_max": -0.07185174301688693,
        "nauc_mrr_at_10_std": -0.12807325096717753,
        "nauc_mrr_at_1_diff1": 0.15610387932529113,
        "nauc_mrr_at_1_max": -0.09824591983546396,
        "nauc_mrr_at_1_std": -0.13914318784294258,
        "nauc_mrr_at_20_diff1": 0.1382786098284509,
        "nauc_mrr_at_20_max": -0.07364476417961506,
        "nauc_mrr_at_20_std": -0.12898192060943495,
        "nauc_mrr_at_3_diff1": 0.13118224861025093,
        "nauc_mrr_at_3_max": -0.08164985279853691,
        "nauc_mrr_at_3_std": -0.13241573571401533,
        "nauc_mrr_at_5_diff1": 0.1346130730317385,
        "nauc_mrr_at_5_max": -0.07404093236468848,
        "nauc_mrr_at_5_std": -0.1340775377068567,
        "nauc_ndcg_at_1000_diff1": 0.15919987960292029,
        "nauc_ndcg_at_1000_max": -0.05457945565481172,
        "nauc_ndcg_at_1000_std": -0.12457339152558143,
        "nauc_ndcg_at_100_diff1": 0.1604091882521101,
        "nauc_ndcg_at_100_max": -0.05281549383775287,
        "nauc_ndcg_at_100_std": -0.12347288098914058,
        "nauc_ndcg_at_10_diff1": 0.1657018523692905,
        "nauc_ndcg_at_10_max": -0.036222943297402846,
        "nauc_ndcg_at_10_std": -0.11284619565817842,
        "nauc_ndcg_at_1_diff1": 0.17346152653542332,
        "nauc_ndcg_at_1_max": -0.09705499215630589,
        "nauc_ndcg_at_1_std": -0.14726476953035533,
        "nauc_ndcg_at_20_diff1": 0.16231721725673165,
        "nauc_ndcg_at_20_max": -0.04147115653921931,
        "nauc_ndcg_at_20_std": -0.11598700704312062,
        "nauc_ndcg_at_3_diff1": 0.15256475371124711,
        "nauc_ndcg_at_3_max": -0.05432154580979357,
        "nauc_ndcg_at_3_std": -0.12841084787822227,
        "nauc_ndcg_at_5_diff1": 0.15236205846534961,
        "nauc_ndcg_at_5_max": -0.04356123278888682,
        "nauc_ndcg_at_5_std": -0.12942556865700913,
        "nauc_precision_at_1000_diff1": -0.038790629929866066,
        "nauc_precision_at_1000_max": 0.3630826341915611,
        "nauc_precision_at_1000_std": 0.4772189839676386,
        "nauc_precision_at_100_diff1": 0.32118609204433185,
        "nauc_precision_at_100_max": 0.4740132817600036,
        "nauc_precision_at_100_std": 0.3456396169952022,
        "nauc_precision_at_10_diff1": 0.22279659689895104,
        "nauc_precision_at_10_max": 0.16823918613191954,
        "nauc_precision_at_10_std": 0.0377209694331257,
        "nauc_precision_at_1_diff1": 0.17346152653542332,
        "nauc_precision_at_1_max": -0.09705499215630589,
        "nauc_precision_at_1_std": -0.14726476953035533,
        "nauc_precision_at_20_diff1": 0.23025740175221762,
        "nauc_precision_at_20_max": 0.2892313928157665,
        "nauc_precision_at_20_std": 0.13522755012490692,
        "nauc_precision_at_3_diff1": 0.1410889527057097,
        "nauc_precision_at_3_max": -0.010771302313530132,
        "nauc_precision_at_3_std": -0.10744937823276193,
        "nauc_precision_at_5_diff1": 0.14012953903010988,
        "nauc_precision_at_5_max": 0.03977485677045894,
        "nauc_precision_at_5_std": -0.10292184602358977,
        "nauc_recall_at_1000_diff1": -0.03879062992990034,
        "nauc_recall_at_1000_max": 0.36308263419153386,
        "nauc_recall_at_1000_std": 0.47721898396760526,
        "nauc_recall_at_100_diff1": 0.3211860920443005,
        "nauc_recall_at_100_max": 0.4740132817599919,
        "nauc_recall_at_100_std": 0.345639616995194,
        "nauc_recall_at_10_diff1": 0.22279659689895054,
        "nauc_recall_at_10_max": 0.16823918613192046,
        "nauc_recall_at_10_std": 0.037720969433127145,
        "nauc_recall_at_1_diff1": 0.17346152653542332,
        "nauc_recall_at_1_max": -0.09705499215630589,
        "nauc_recall_at_1_std": -0.14726476953035533,
        "nauc_recall_at_20_diff1": 0.23025740175221865,
        "nauc_recall_at_20_max": 0.2892313928157675,
        "nauc_recall_at_20_std": 0.13522755012490456,
        "nauc_recall_at_3_diff1": 0.14108895270570979,
        "nauc_recall_at_3_max": -0.010771302313529425,
        "nauc_recall_at_3_std": -0.10744937823276134,
        "nauc_recall_at_5_diff1": 0.14012953903010958,
        "nauc_recall_at_5_max": 0.039774856770459645,
        "nauc_recall_at_5_std": -0.10292184602358935,
        "ndcg_at_1": 0.40754,
        "ndcg_at_10": 0.63616,
        "ndcg_at_100": 0.66063,
        "ndcg_at_1000": 0.6613,
        "ndcg_at_20": 0.65131,
        "ndcg_at_3": 0.55717,
        "ndcg_at_5": 0.59461,
        "precision_at_1": 0.40754,
        "precision_at_10": 0.08841,
        "precision_at_100": 0.00991,
        "precision_at_1000": 0.001,
        "precision_at_20": 0.04716,
        "precision_at_3": 0.22238,
        "precision_at_5": 0.15149,
        "recall_at_1": 0.40754,
        "recall_at_10": 0.88407,
        "recall_at_100": 0.99147,
        "recall_at_1000": 0.99644,
        "recall_at_20": 0.9431,
        "recall_at_3": 0.66714,
        "recall_at_5": 0.75747
      }
    ]
  },
  "task_name": "ArguAna"
}
```

现在我们已经成功地使用mteb进行了评估！在下一个教程中，我们将展示如何在英语MTEB的全部56个任务上评估您的模型，并与排行榜上的模型进行比较。

