# 跨模态检索机制研究：实验计划

状态：进行中（2026-09-26）
设备：DGX Spark（192.168.1.170），模型本机缓存离线加载
语料：见 datasets/，来源与许可见 scripts/fetch_crossmodal_datasets.py 的 SOURCES 清单

## 一、研究问题

文本内相似度的四因素（长度、内容相关性、语言、用词）已在 XiaokeAILabs 仓
test_sentence_length 实验里测完，结论是相似度分数由这四类因素共同推动，且幅度
与模型强相关。跨模态要多回答三件事。

第一，模态 gap。同一内容的图与文之间的相似度，比同模态的文与文低多少，这个
折扣是否与跨语言折扣同类。

第二，描述中介。查询命中一张图，走图的原生向量和走图的文字描述，两条路的分数
差多少，什么时候一条能救另一条。这一条对应素材检索里帧图像向量与帧描述向量
两条路由的实际取舍。

第三，细粒度组合。查询说「马在吃草」，图里是马在吃草，模型能否不与「草在吃马」
混淆。这类失效在文本内不存在，是跨模态特有的。

## 二、语料与用途

| 数据集 | 规模 | 用途 |
|---|---|---|
| Flickr30k 图像 | 31783 张 | 图文检索主基准 |
| Flickr30k 标注 CSV | 158915 条 caption | 每图 5 条人工描述 |
| ARO VG-Relation | 23937 条 | 关系硬负样本，true 与 false caption 只差两实体关系顺序 |
| ARO VG-Attribution | 28748 条 | 属性硬负样本，只差对象与属性的归因 |
| ARO Flickr30k-Order（原始标注） | 1000 图 × 5 caption | 词序硬负样本的底料，扰动需现场生成 |

注意 flickr30k_test.json 是 Flickr30k Karpathy 划分的原始标注，每图 5 条人工
caption，不是 ARO 预处理后的版本。ARO 官方在读取时才对每条 caption 现场生成
四种打乱。本实验自己实现打乱，且不依赖词性标注，因为 DGX 无外网，装不了
spaCy 与 nltk 数据。
| FSD50K（部分） | 约 3200 条 | 音频侧，多标签声音事件，音频二进制在 parquet 内 |

ARO 覆盖三类细粒度错误。VG-Relation 是关系颠倒，true 与 false 只交换两个实体
的关系顺序。VG-Attribution 是属性错配，只交换形容词与名词的组合。两者的
true 与 false caption 已由官方构造好，直接可用。Flickr30k-Order 是词序混乱，
需要本实验自己生成扰动。

已抽检确认：VG-Relation 的 false caption 语义上仍然成立（「手臂在腿左边」与
「腿在手臂左边」都通顺），测的是纯空间关系理解，不混入语义荒谬。VG-Attribution
的 false 有部分荒谬（「长草的大象」），这本身就是测试点，模型应当因为图里没有
这个内容而给低分。两类都不需要额外清洗。

## 三、实验分组

每组都配无关负对照。绝对相似度在跨模态下不可信，各向异性会把不同模态的向量
推向不同区域，必须报 gap 不报绝对分。

### A 组：模态 gap 与检索基线

用 Flickr30k 全部图文对。算三类相似度。

图对自身 caption、caption 对其他图的 caption、图对其他图的图。三者对照给出
跨模态折扣的量级，并给出同模态内的无关基线。

指标用标准检索指标 R@1、R@5、R@10，以及 margin（相关对减无关对的中位数）。
margin 与绝对分一起报，因为绝对分不可跨模型比较。

### B 组：ARO 硬负样本，隔离「换个说法」的跨模态代价

三个子集各取全部或大样本。对每条样本算两个差。

图侧 gap 是 sim(图, true_caption) 减 sim(图, false_caption)。文本侧 gap 是
sim(true_caption, false_caption) 与底噪的差。两者对照回答：模型在纯文本上
能分清新旧说法的程度，与在图侧能分的程度差多少。

这条直接连接文本内实验的结论。文本内已测到换个说法掉 0.1 到 0.5，跨模态的
预期是掉得更多，因为要同时跨过模态边界和表述差异两层。若实测不是这个方向，
说明 ARO 的失效不是表述问题而是结构理解问题，需要改归因。

按关系、属性、词序三类分别报，因为三者的失效机制可能不同。

### C 组：描述中介，两条路由的取舍

用 FSD50K 或 Flickr30k 的子集。对同一内容构造三条查询路径。

原生向量路径，查询直接对音频或图像向量。描述路径，查询对内容的文字描述。
标签路径，查询对多标签词（FSD50K 的标签如 Dog、Rain）。

三条路径的 margin 对照给出路由取舍的条件。这一组的预期结论是：描述路径在
查询用语与描述用语接近时强，原生路径在查询指向视觉或听觉细节时强，两者互补
而非替代。

### D 组：细粒度属性的单项控制

用 ARO 的 caption 结构做单向扰动。固定一张图，逐项改动 caption 的一个属性
（换颜色词、换数量词、换物体词、换关系词、换空间词），看 gap 的变化梯度。

这条给出哪类词最能推高或拉低跨模态分数，是把 B 组的结论拆到词级。

## 四、指标定义

- R@k：查询的正确目标进入前 k 的比例，标准检索指标。
- margin：相关对相似度减无关对相似度的中位数，成对报。
- gap_A/B：见 B 组定义。
- 全部指标按模型分别报，不跨模型比较绝对值。

## 五、模型

第一轮只用 jina-embeddings-v4（Qwen2.5-VL-3B 底座，text 与 image 统一空间，
2048 维，本机缓存可用）。加载需打 transformers 5.16.1 兼容补丁，见
scripts/mm_patch.py。

第二轮加 Qwen3-VL-Embedding-2B 与 CLIP 系做模型间对照，验证「幅度与模型强相关」
这条文本内已成立的结论在跨模态下是否也成立。

## 六、已知限制

- FSD50K 只取部分分片，不是全集，音频侧结论限于已取子集。
- Flickr30k 的 caption 是英文，中文查询的跨语言跨模态组合未测。
- jina-v4 的 task adapter 选 retrieval，其他 adapter 未对照。
- ARO 的 false caption 是人工扰动生成的，其中一部分在现实世界无意义，模型可能
  因为语义荒谬而给低分，而不是因为结构理解错误。这一点要在 B 组单独检查：
  对 false caption 做人工可读性抽样，把明显荒谬的样本剔出来重算。

## 七、与文本内实验的关系

文本内四因素实验的产物在 XiaokeAILabs 仓
experiments/embedding_research/behavior_tests/test_sentence_length/，报告在
results/REPORT.md。跨模态这一轮不重复测那四因素，只测跨模态特有的三项：模态
gap、描述中介、细粒度组合。两轮的共同结论是分数由输入的结构决定而非由「相关
与否」决定，跨模态只是把可变动的一项从语言换成了模态。

## 八、方法论来源（论文）

本实验的评测设计不是自创，来自以下工作。论文原文尚未下载到 pkm-hub-papers，
列在此处避免丢失，等实验结论出来后再统一入库，避免入库后因设计变动而返工。

| 论文 | 提供了什么 |
|---|---|
| When and Why Vision-Language Models Behave like Bags-of-Words, and What to Do about It?（ICLR 2023，Yuksekgonul 等） | ARO 基准本身。VG-Relation、VG-Attribution、COCO-Order、Flickr30k-Order 四个子集的构造方式与官方下载源 |
| Winoground（Thrush 等 2022） | 词序敏感性的评测范式，每样本两图两描述，描述仅词序不同。本实验未用，因为 ARO 的负样本更可控 |
| VALSE | 组合性评测的另一条路线，本实验未用 |
| Clotho: an Audio Captioning Dataset（ICASSP 2020，Drossos 等） | 音频 caption 数据集的标准形态，每音频 5 条句子级描述。Zenodo 不可达，未采用 |
| On Metric Learning for Audio-Text Cross-Modal Retrieval | 音频文本检索的度量学习目标对比，AudioCaps 与 Clotho 上的系统评测。作为音频侧方法论文待补 |
| Semantics at an Angle（arXiv 2504.16318） | 余弦失效的四类诊断框架，已在文本内实验用过，跨模态这轮的读数纪律沿用 |

音频侧最终选 FSD50K 而非 Clotho 或 AudioCaps，原因是后两者的音频源分别落在
Zenodo 与 YouTube，本机网络均不可达，而 FSD50K 在 HuggingFace 有含音频二进制的
镜像。代价是它是多标签而非自然语言 caption，「查询」的形态从句子退化为标签词，
这一点在 C 组的结论里必须标注，不能写成自然语言查询的结论。
