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
| `code/route_compare.py` | C 组入口：图向量路由 对 描述向量路由 |
| `code/order_hardneg.py` | D 组入口：词序扰动硬负样本，两种扰动做对照 |
| `code/audio_baseline.py` | E 组入口：音频对文本基线，`--pool` 控制候选池大小 |
| `code/omni_image_audio.py` | F 组入口：同一模型内同时测图文与音频，三个读数共用同模态分母 |
| `code/clap_audio.py` | G 组入口：第二个音频模型（CLAP 双塔），测同一套以分离模态难与模型弱 |
| `code/diag_cohesion.py` | 内聚性诊断，区分「模型特性」与「编码 bug」 |
| `code/collect_t2t.py` | 补采文本对文本分布，供画图用 |
| `results/<模型名>/` | 按模型分目录的读数与直方图汇总，全量数据在 DGX |
| `logs/` | 执行日志，从 DGX 拉回并清理噪声，含与 Spark 原名的对应表 |
| `results/REPORT.md` | 实验报告，含两个模型的对照读数与解读 |

## 测什么

三个问题，对应 PLAN.md 的六组，目前跑完两组、两个模型。

| 问题 | 组 | 状态 |
|---|---|---|
| 模态 gap：同内容的图与文相似度比同模态低多少 | A 组 | 已完成，两个模型，口径已对齐 |
| 换个说法在跨模态下的代价是否比文本内更大 | B 组 | 已完成，两个模型 |
| 折扣比例是否随模型变化 | A 组扩展 | 已完成，两个模型 |
| 走图的原生向量与走图的文字描述，哪条路更好 | C 组 | 已完成，两个模型，优劣顺序反转 |
| 词序进不进入表示，语法崩坏会被惩罚吗 | D 组 | 已完成，两个模型，表现截然相反 |
| 音频对文本的 gap 有多厚，检索能不能用 | E 组 | 已完成，一个新模型，需换模型（图文模型不支持音频） |
| 音频与图文的跨模态损失差多少 | F 组 | 已完成，同一模型内，音频折扣是图文的 1/40 |
| 音频折扣低是模态难还是模型弱 | G 组 | 已完成，两者都成立：换模型 margin 翻 7.4 倍，改善后仍不可用 |
| 三类失效机制在本地是否存在 | 几何诊断 | 已完成，各向异性与 hubness 都是实锤，RRF 等权注水也被证伪 |
| 抽象层级、细粒度属性、负向查询、图内文字 | 后续组 | 未开始 |

语料：Flickr30k 图像与五条人工标注，ARO 的关系与属性硬负样本，FSD50K 音频分片（字段已验通：音频在 `audio.bytes` 的 WAV 二进制，44.1kHz 单声道，另有 `description` 人类可读描述）。全部数据集只落 DGX，本机不留副本。

模型：A 到 D 组跑 jina-embeddings-v4 与 BGE-VL-base（均为图文双塔，不支持音频编码）。E 与 F 组换 jina-embeddings-v5-omni-small，单塔统一架构，文本、图像、音频、视频共用一个向量空间，是首个能在同一模型内比较多模态的本地样本。E 组只测音频，绝对分不与图文互比；F 组在同一模型内同时测图文与音频，两侧可比（同模型同分母），但 F 组的读数不可与 A 到 D 组互比。G 组换 laion/larger_clap_general，双塔独立架构（HTSAT 音频编码器加 RoBERTa 文本编码器，端到端联合对比训练），与前两者的架构都不同，用来分离「模态难」与「单模型音频塔弱」。

## 主结论（详见 results/REPORT.md）

跨模态判别 margin（相关样本减无关样本的中位数）显著低于同模态，但幅度由模型决定，不是常数。jina-embeddings-v4 的跨模态 margin 只有同模态的 34%，BGE-VL-base 是 67%。同一批 500 张图上，图检索文本的 R@1 是 6% 对 72%，差 12 倍。折扣比例差一倍，这个差距本身比任何单一模型的绝对值都更要紧：它意味着换模型必须重测基线，沿用上一个模型的阈值会失效。

更反直觉的一点是，这个差距没有单调规律。BGE-VL 的向量维度只有 jina-v4 的四分之一，跨模态表现反而更好，所以不能按参数量或维度推断哪个模型的跨模态对齐更强。

**折扣幅度受两件事控制，模型与模态，而模态的量级差远大于模型。** 模型之间差一倍（jina-v4 的 34% 对 BGE-VL 的 67%），同一模型内换模态差 40 倍（v5-omni 上图对文 66.5%，音频对文 1.7%）。所以「跨模态损失有多大」的完整表述是：模型决定同一个模态内的折扣幅度，模态决定折扣的量级。做音频检索时不要以图文的可用度为参照。

这条在 G 组被补了一刀：同一模态内换模型能改善 7.4 倍（v5-omni 的音频 margin 0.0087 对 CLAP 的 0.0643），但改善之后 CLAP 的 R@1 仍只有 2%，够不上检索可用线。所以模态决定量级、模型决定量级内的位置、而当前音频模型的位置都低于可用线，三句缺一不可。

**多路检索融合前先看各路质量是否同量级。** 实测等权 RRF 在两路质量差 90 倍时，把强路从 R@1 90.2% 拖到 13.2%，比弱那路还差。融合不是折中，质量悬殊时是注水。

**各向异性程度可以预测 cos 失效率。** 同一模型内跨模态成立：图文侧有效秩占 17%、第一奇异值 15.7%，音频侧有效秩只占 1.5%、第一奇异值 65.5%、负相似度占比接近零。这一项裁决了库内两篇论文的相反预测，并把各向异性从名词变成可测指标。

**hubness 在音频侧是图文的两倍。** k=10 近邻次数的基尼系数图文 0.374、音频 0.74 至 0.75，音频侧最热的一条描述成了 115 个查询的前十近邻而平均只有 10 次。它与中位排名（图文 1、音频 195）方向一致，hubness 从透镜变成实测读数。

机制上，这不是模型变差，是对比训练的目标只优化排序、不优化跨模态距离。gap 是结构性的，初始化时就存在，训练会磨但闭合速率极慢。

B 组给出更硬的一层：6000 条硬负样本上，关系顺序上两个模型都与掷硬币无区别（52%），属性归因上 jina-v4 完全无信号（47.7%）、BGE-VL 略强于硬币（61.6%）。这一层与 gap 叠加，共同解释图检索文本的 R@1 为何只有个位数百分比。

C 组回答路由取舍：走图的原生向量还是走图的文字描述。两条路的优劣顺序随模型反转，jina-v4 上描述路由 R@1 是图向量路由的 2.5 倍，BGE-VL 上反过来。所以选路由也要按模型实测，没有普适答案。

D 组闭合 ARO 三类硬负样本，测词序进不进入表示。BGE-VL 对词序和语法敏感（语法崩坏时 95.3% 能排除），jina-v4 基本不敏感，而且对乱码有系统性负偏好：语法崩坏的句子拿到的图侧分数反而高于原句（gap 中位数负 0.0151）。解释是乱码句词集合不变、表示被推向更通用区域，而 jina-v4 的无关图文基线本身就高（0.363），通用文本因此拿到更高分数。

## 踩过的坑

六个都记在 `code/` 的注释里，省得下次重踩。

remote code 的 ProcessorMixin 不传 tokenizer，直接加载会崩，`mm_patch.py` 打补丁；句柄库的模态白名单不认模型的 remote code，需要显式声明两种模态；ARO 的标注文件是原始划分不是预处理版，词序扰动要自己生成，而目标机无外网装不了词性标注库，改用不依赖词性的轻量实现。

第四个是加第二个模型时踩的，值得单独说：`mm_patch` 定位 remote code 目录原先靠「取最新修改的那个」，缓存里只有一个 remote code 模型时能跑，第二个模型一下载就被抢走，报找不到第一个模型的文件。已改为按模型名精确匹配，匹配不到直接报错不猜。另外补丁的 fallback 是「拿第一个模型的 processor 类手动构造」，对标准 CLIP 系模型会直接失败，所以补丁现在只在 jina-v4 上生效，其他模型走原生加载路径。结果按模型分目录，否则第二个模型会静默覆盖第一个的读数。

第五个是 C 组自己踩的：第一版用同一条 caption 既作查询又作描述候选，排除自身之后这条查询就不存在正确候选，检索任务不成立，读出来 R@1 全是 0。改成查询取第二条 caption、候选取第一条，两条路才都有明确答案。取值顺序也错过一次：把对角线置成负无穷之后又去取它当相关值，得到一片 Infinity。

第六个是数据集侧的：FSD50K 的两个 parquet 分片都是残文件（一个是完整大小的四分之一，另一个一半），pyarrow 报 magic bytes not found。头四个字节是合法魔数但尾部不是，所以是下载中断而非格式错误。重下 train-00000 后字段才验通：音频在 `audio.bytes`（WAV），另有 `description` 字段是人类可读描述。

## 复现

DGX Spark 上，模型走本机缓存离线加载：

```bash
ssh ke@192.168.1.170
cd ~/crossmodal_lab/scripts

# A 组 + B 组
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ~/xinfer-env/bin/python run_experiments.py --group ab --limit 500

# 换模型，结果落 results/<模型名>/，不覆盖上一个
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ~/xinfer-env/bin/python run_experiments.py --group ab --limit 500 --model BAAI/BGE-VL-base

# 换模型后要补同模态基线，否则算不出折扣比例
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ~/xinfer-env/bin/python collect_t2t.py 500 BAAI/BGE-VL-base

# C 组：图向量路由 对 描述向量路由
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ~/xinfer-env/bin/python route_compare.py 500
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ~/xinfer-env/bin/python route_compare.py 500 BAAI/BGE-VL-base

# D 组：词序扰动硬负样本
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ~/xinfer-env/bin/python order_hardneg.py 1000
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ~/xinfer-env/bin/python order_hardneg.py 1000 BAAI/BGE-VL-base
```

A 组与 C 组必须用同一个 limit，否则 R@1 不能横向比。数据不在仓里，先跑 `code/fetch_crossmodal_datasets.py`。目标机无外网，需经本机代理隧道中转，隧道脚本在 `pkm-hub-runtime/scripts/start-dgx-tunnel.ps1`。

## 三项几何诊断：复现脚本

这三项（各向异性、hubness、RRF 融合）原先只有终端输出，撤到不可复现。现补了两个脚本，读数与口径都落盘到 `results/geometry_diag/`。

```bash
# hubness + RRF + 检索读数，只用已有 simmat，不需要 GPU，秒级
~/xinfer-env/bin/python scripts/diag_geometry_simmat.py --check   # --check 与 REPORT 历史读数逐项比对

# 各向异性，需要 embedding 矩阵，重编码（首次几分钟，之后走 results/diag_cache/ 缓存）
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ~/xinfer-env/bin/python scripts/diag_anisotropy_embed.py --n 500 --check
```

两个口径上的坑，都是反推定义时才发现、且会让数字静默偏移的：

`unrel_median` 是**每查询的中位数再取中位**，不是全局中位数。omni_a2t 全局 0.2398 对逐查询 0.2511，clap_a2t 0.3265 对 0.3353。REPORT 的 margin 一律用逐查询口径。

hubness 取 top-k 时必须**排除自匹配**，对角线置 -inf 后再排。不排除时每条候选必得自己那一票，分布被系统性拉平，omni_i2t 的 deg_max 会从 55 掉到 50、基尼从 0.374 掉到 0.339。

另有一个 transformers 5.x 的参数改名：`proc(audios=...)` 已废弃，必须写 `proc(audio=...)`，否则报 deprecated ValueError。
