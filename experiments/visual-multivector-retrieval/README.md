# 用 ColSmol 检索文档页面：多向量匹配实测

[English](README.en.md)

要从几百张报告、表格和图表页面中找到回答问题的一页，通常先要 OCR，再对识别文本建索引。视觉文档检索提供另一条路径：直接把页面作为图片编码，让检索模型同时利用文字、布局和图表信息。本实验使用约 256M 参数的 ColSmol，测量它在 100 个候选页面中定位已知相关页面的表现。

![文字问题和文档页面经过多向量匹配后排列](retrieval-flow.svg)

在 DGX Spark 的本次测试中，100 条问题的 **Recall@1 为 0.55，Recall@5 为 0.78，nDCG@5 为 0.6664**。即 55 条问题的已知配对页排在第一，78 条排在前五。这是小规模固定候选实测，不能当作完整 ViDoRe 榜单成绩。

## 模型怎样匹配问题和页面

普通单向量检索把整段内容压缩成一个向量；ColSmol 为问题和页面分别生成多个向量，采用 ColBERT 风格的延迟交互。每个问题向量与最相似的页面向量匹配，再将这些最大相似度相加：

$$s(q,d)=\sum_i\max_j \widehat{q_i}^{\mathsf T}\widehat{d_j}.$$

模型因此可以让问题中不同部分分别寻找页面上的对应线索。更多向量也意味着更高的存储和匹配成本，实际应用通常还需要索引或候选筛选；本实验没有测量全库索引或线上延迟。原理参见 [ColPali 论文](https://arxiv.org/abs/2407.01449)及 [ColSmol 模型卡](https://huggingface.co/vidore/colSmol-256M)。

## 数据、方法和结果

[ViDoRe DocVQA 测试子集](https://huggingface.co/datasets/vidore/docvqa_test_subsampled) 包含文档页图像及针对页面的问题，涵盖表格、报告和表单等内容。我们按数据顺序选前 100 个不同的 `image_filename`，各取首次出现的问题。每个问题对同一组 100 页排序，以其配对页作为唯一已知正例。

| 指标 | 结果 | 含义 |
|---|---:|---|
| Recall@1 | 0.55 | 配对页排在第 1 的问题比例 |
| Recall@5 | 0.78 | 配对页进入前 5 的问题比例 |
| nDCG@5 | 0.6664 | 配对页越靠前，得分越高 |

数据可能存在其他同样相关但未标注的页面，因此这些指标只针对已知配对页。候选范围只有选出的 100 页，不代表从大型文档库全量召回的效果。原模型直接推理，没有做本次微调。模型卡称其训练集与 ViDoRe 未使用相同的多页 PDF，但这不能排除基础模型预训练阶段的数据重叠。

[匿名逐查询排名](baseline100.json)记录 100 条查询的配对页名次，不包含原始图片、问题或数据集 ID。运行 `python recompute.py` 可以从名次重新计算三个汇总指标，但不能据此验证模型打分或页面标签。

## 运行基线

本次使用 Linux、NVIDIA GB10、BF16、PyTorch 2.14.0+cu130、Transformers 4.56.2、ColPali Engine 0.3.13、PEFT 0.17.1 和 Datasets 4.1.1。下面的模型下载需要 `huggingface_hub`，数据和权重自行从上游获取，不放在本仓库。

```bash
python -m pip install 'transformers==4.56.2' 'colpali-engine==0.3.13' 'peft==0.17.1' 'datasets==4.1.1' huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('vidore/colSmol-256M', revision='c79b633e17e060cc109b11aa2bf92a1517d3bd6f', local_dir='model')"
python -c "from huggingface_hub import snapshot_download; snapshot_download('vidore/ColSmolVLM-Instruct-256M-base', revision='99ca96f1f6b95b3a69e6abef74a2416cb738fed0', local_dir='base')"
python prepare_data.py --output data
python baseline.py --data data --base base --model model --out /tmp/visual-smoke.json --count 10
python baseline.py --data data --base base --model model --out /tmp/visual-baseline100.json --count 100
```

ColSmol 的发布权重是 adapter，需要先加载指定底模，再加载 adapter；底模本身不是可直接使用的检索器。测试耗时 78.12 秒，包含编码和两两打分，不含模型与数据加载；PyTorch 峰值分配约 2.81 GiB，不等于整机内存占用。脚本保留本次机器使用的 50 GiB 可用内存保护下限，并不是此模型的通用最低要求。

模型卡提示 ColPali Engine 当前提示格式与模型训练时使用的格式存在差异；本次用的是 0.3.13 实现，因此不与模型卡结果直接比较。不同软件版本、硬件和浮点实现也可能导致排名变化。10 页运行仅用于检查环境，100 页运行才对应这里的基线。

## 许可与后续

模型及底模仓库元数据标记 MIT，底模卡正文说明骨干为 Apache-2.0、adapter 为 MIT；模型卡许可正文另有 ColQwen2 文案，重新分发权重前应核对具体文件及上游许可。ViDoRe 数据集卡标记 MIT，仍应遵守原始 DocVQA 及源文档的适用条件。本目录不再分发数据集页面、问题文本或模型权重。

微调和蒸馏尚未开展。后续若做对照，需要先固定与测试文档来源隔离的训练集，比较原模型、同预算监督训练和教师分数蒸馏，再使用独立留出文档评估。
