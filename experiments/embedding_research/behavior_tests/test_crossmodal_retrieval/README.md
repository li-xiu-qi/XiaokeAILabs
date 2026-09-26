# 跨模态检索机制研究：模态 gap、硬负样本与词袋

> 2026-09-26 建立。与 `test_sentence_length`（文本内四因素）同属 embedding 行为研究，靶子从同一编码器内部换成跨模态。文本内那批回答「分数由什么决定」，这批回答「跨模态时分数为什么更不可信」。

## 布局

| 位置 | 内容 |
|---|---|
| `PLAN.md` | 实验计划：研究问题、语料来源与许可、六组分组设计 |
| `README.md` | 本文件 |
| `code/` | 全部脚本，2026-09-26 在 DGX Spark 上跑通 |
| `code/mm_lab.py` | 编码引擎，封装双塔模型的图文编码，按模型判断是否打兼容补丁 |
| `code/mm_patch.py` | jina-v4 专用加载补丁，修 remote code 与模态白名单两个坑 |
| `code/dbg.py` | 补丁自检，验证 patch 是否真的生效（加载失败时先跑这个） |
| `code/fetch_crossmodal_datasets.py` | 数据集下载，含每个文件的来源、体积、许可 |
| `code/run_experiments.py` | A 组（检索基线）与 B 组（ARO 硬负样本）入口，`--model` 换模型 |
| `code/diag_cohesion.py` | 内聚性诊断，区分「模型特性」与「编码 bug」 |
| `code/collect_t2t.py` | 补采文本对文本分布，供画图用 |
| `results/<模型名>/` | 按模型分目录的读数与直方图汇总，全量数据在 DGX |
| `results/REPORT.md` | 实验报告，含两个模型的对照读数与解读 |

## 测什么

三个问题，对应 PLAN.md 的六组，目前跑完两组、两个模型。

| 问题 | 组 | 状态 |
|---|---|---|
| 模态 gap：同内容的图与文相似度比同模态低多少 | A 组 | 已完成，两个模型 |
| 换个说法在跨模态下的代价是否比文本内更大 | B 组 | 已完成，两个模型 |
| 折扣比例是否随模型变化 | A 组扩展 | 已完成，两个模型 |
| 描述中介：走图的原生向量与走图的文字描述差多少 | 后续组 | 未开始 |
| 抽象层级、细粒度属性、负向查询、图内文字 | 后续组 | 未开始 |

语料：Flickr30k 图像与五条人工标注，ARO 的关系与属性硬负样本，FSD50K 音频分片（字段未验证）。全部数据集只落 DGX，本机不留副本。

## 主结论（详见 results/REPORT.md）

跨模态判别 margin（相关样本减无关样本的中位数）显著低于同模态，但幅度由模型决定，不是常数。jina-embeddings-v4 的跨模态 margin 只有同模态的 34%，BGE-VL-base 是 67%。折扣比例差一倍，这个差距本身比任何单一模型的绝对值都更要紧：它意味着换模型必须重测基线，沿用上一个模型的阈值会失效。

更反直觉的一点是，这个差距没有单调规律。BGE-VL 的向量维度只有 jina-v4 的四分之一，跨模态表现反而更好，所以不能按参数量或维度推断哪个模型的跨模态对齐更强。

机制上，这不是模型变差，是对比训练的目标只优化排序、不优化跨模态距离。gap 是结构性的，初始化时就存在，训练会磨但闭合速率极慢。

B 组给出更硬的一层：6000 条硬负样本上，关系顺序上两个模型都与掷硬币无区别（52%），属性归因上 jina-v4 完全无信号（47.7%）、BGE-VL 略强于硬币（61.6%）。这一层与 gap 叠加，共同解释图检索文本的 R@1 为何只有个位数百分比。

## 踩过的坑

四个都记在 `code/` 的注释里，省得下次重踩。

remote code 的 ProcessorMixin 不传 tokenizer，直接加载会崩，`mm_patch.py` 打补丁；句柄库的模态白名单不认模型的 remote code，需要显式声明两种模态；ARO 的标注文件是原始划分不是预处理版，词序扰动要自己生成，而目标机无外网装不了词性标注库，改用不依赖词性的轻量实现。

第四个是加第二个模型时踩的，值得单独说：`mm_patch` 定位 remote code 目录原先靠「取最新修改的那个」，缓存里只有一个 remote code 模型时能跑，第二个模型一下载就被抢走，报找不到第一个模型的文件。已改为按模型名精确匹配，匹配不到直接报错不猜。另外补丁的 fallback 是「拿第一个模型的 processor 类手动构造」，对标准 CLIP 系模型会直接失败，所以补丁现在只在 jina-v4 上生效，其他模型走原生加载路径。

## 复现

DGX Spark 上，模型走本机缓存离线加载：

```bash
ssh ke@192.168.1.170
cd ~/crossmodal_lab/scripts

# 默认模型
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ~/xinfer-env/bin/python run_experiments.py --group ab --limit 3000

# 换模型，结果落 results/<模型名>/，不覆盖上一个
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ~/xinfer-env/bin/python run_experiments.py --group ab --limit 3000 --model BAAI/BGE-VL-base

# 换模型后要补同模态基线，否则算不出折扣比例
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ~/xinfer-env/bin/python collect_t2t.py 500 BAAI/BGE-VL-base
```

数据不在仓里，先跑 `code/fetch_crossmodal_datasets.py`。目标机无外网，需经本机代理隧道中转，隧道脚本在 `pkm-hub-runtime/scripts/start-dgx-tunnel.ps1`。
