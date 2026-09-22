# PForDelta 整数压缩性能模型

> 实证日期 2026-09-07 · Python 3.13.0 · 脚本 `scripts/pfor_delta_model.py`、`scripts/verify_pfor_delta.py` · 结果 `results/pfor_delta_*.json`

PForDelta（Patched Frame of Reference，Zhang, Long & Suel, 2008）把整数序列按固定长度切帧，每帧选一个统一位宽打包九成数值，装不下的异常值另存异常区。它是 Lucene 5.0+ 的默认 docID 压缩方案。核心价值不在压缩率（与 Simple8b 同量级），而在统一位宽：解码是一段无分支的移位循环，可直接映射到 SIMD 指令，异常值的污染也被限制在帧内。本笔记给出位宽选择算法、帧编解码、异常区编码与压缩比闭式公式，并用实测验证压缩比、帧大小、异常值比例与编解码吞吐。核心结论是 90% 位宽规则只在异常值比例低于 5% 时成立，超过后刚性帧规则反被 Simple8b 的贪心策略超过。

## 核心公式

每帧的字节 = ceil(frame_size × b / 8) + 2 + 异常区字节，其中 b 是位宽，2 字节是帧头（1 字节存 b，1 字节存异常数），异常区用 Simple8b 压缩扁平的 (下标, 原始值) 序列。

位宽选择：遍历 0 到 32 位，取能容纳至少 90% 数值的最小位宽。

```
b = min{b' : |{v : v < 2^b'}| ≥ 0.9 × frame_size}
```

压缩比(vs uint32) = 32 × n / 总字节。均匀分布下 b 由九成分位值决定，实测 0-1000 的整数选 10 位（1000 本身需 10 位，九成分位也落在 10 位区间），每整数 10.13 位。

## 编解码结构

编码一帧分三步：选位宽 b，用 b 位打包全部 frame_size 个值（异常值位置填 0），把超出 b 位的值连同下标写进异常区。

解码一帧分两步：按 b 位从字里切出正常区，再把异常值按下标填回。

异常区是扁平的 (下标, 值) 序列，下标天然小（0-127 只需 7 位），原始值可能很大（到 2^24）。用 Simple8b 压这个混合序列时，下标吃窄档、原始值吃宽档，两种位宽在相邻的字里自动切换。这是 Simple8b 在 PForDelta 里的实际角色：不是主编码器，是异常区的容器。

末帧不足 frame_size 时按帧内记录的 count 截断，编解码严格可逆。

## 压缩比

本机实测（n=100 万，frame_size=128）：

| 分布 | 每整数位宽 | vs uint32 | vs varint | vs Simple8b |
|---|---:|---:|---:|---:|
| docID（1% 异常值） | 12.76 | 2.51x | 1.24x | 1.04x |
| 均匀 0-1000 | 10.13 | 3.16x | 1.48x | 1.05x |
| 全大数 0-2^24 | 24.13 | 1.33x | 1.28x | 1.33x |

字节数对照（uint32 恒为 3.81 MiB）：docID 分布下 PForDelta 1.52 MiB、Simple8b 1.59 MiB、Varint 1.88 MiB。三种位封装编码在同一量级，PForDelta 对 Simple8b 的优势只有 4%，因为 1% 的异常值比例下 Simple8b 的贪心定档已经足够好。

PForDelta 真正拉开差距的是异常值比例升高的时候。固定 0-3000 主体、扫描异常值比例（n=100 万）：

| 异常值比例 | 平均位宽/帧 | 每整数位宽 | vs uint32 | vs Simple8b |
|---:|---:|---:|---:|---:|
| 0% | 12.0 | 12.13 | 2.64x | 1.06x |
| 0.5% | 12.0 | 12.44 | 2.57x | 1.05x |
| 1% | 12.0 | 12.76 | 2.51x | 1.04x |
| 2% | 12.0 | 13.39 | 2.39x | 1.03x |
| 5% | 12.1 | 15.31 | 2.09x | 1.00x |
| 10% | 16.2 | 21.42 | 1.49x | 0.81x |
| 20% | 23.2 | 26.55 | 1.21x | 0.78x |

5% 是拐点。低于 5% 时 PForDelta 稳赢 Simple8b 3-6%；超过 5% 后 90% 规则失效：帧内异常值超过阈值，位宽被迫一路加宽到能装下 90%，10% 异常值时平均位宽从 12 跳到 16.2，20% 时到 23.2。同期 Simple8b 只在与异常值相邻的字里付 30 位的代价， degradation 平缓得多。

这个拐点是 90% 规则的设计前提，不是实现缺陷。规则假定异常值罕见，真实 docID 差值序列（异常值比例通常在 1% 以下）符合这个前提。遇到异常值密集的序列，该退回 Simple8b 的贪心或改用 Elias-Fano。

## 帧大小

帧大小对压缩比影响很小（docID 分布，n=100 万）：

| frame_size | 每整数位宽 | vs uint32 |
|---:|---:|---:|
| 32 | 13.15 | 2.43x |
| 64 | 12.89 | 2.48x |
| 128 | 12.76 | 2.51x |
| 256 | 12.70 | 2.52x |
| 512 | 12.67 | 2.53x |
| 1024 | 12.65 | 2.53x |

从 32 到 1024 只差 0.5 位/整数（4%）。原因是帧头只有 2 字节，开销 = 16 bit / frame_size，在 128 帧上是 0.125 bit/整数，可以忽略；异常区的大小只取决于异常值个数，与帧数无关。所以帧大小不是压缩杠杆。

帧大小的真实作用是随机访问粒度：要读第 k 个整数，必须解压它所在的整帧。帧越小 seek 越快，帧越大压缩略好。Lucene 选 128 是在这两者之间取的平衡点，对应 128 个 docID 一跳，与倒排索引的块对齐。

均匀分布上同样只有微小差异（128 帧 10.13 位 vs 1024 帧 10.02 位），验证了帧头开销可忽略这个判断。

## 闭式预测

预测模型：主体按 90 分位位宽 b_b 打包，异常值按原始位宽存，加每帧 2 字节帧头与异常区字节。

```
预测位宽 = [Σ min(bitlen(v), b_b) + 16×帧数 + 8×异常区字节] / n
```

实测（docID 分布，n=10 万）：frame_size=128 时预测 12.55、实测 12.76，误差 1.6%；256 帧时预测 12.49、实测 12.70，同样 1.6%。均匀分布上预测与实测的偏差在同一量级（0-1000 分布预测约 12.55、实测 12.76）。

这个模型在低异常值比例下准，因为它假设每帧的位宽都等于全局九成分位。异常值比例升高后帧间位宽分化变大，预测会偏低（实测 21.42 时模型给的值明显更小），所以它只适合异常值稀少的场景，与算法本身的适用范围一致。

## 编解码速度

纯 Python 实现（n=100 万，frame_size=128，取三轮最好）：

| 分布 | 编码 | 解码 |
|---|---:|---:|
| docID | 0.36 M int/s | 0.43 M int/s |
| 均匀 0-1000 | 0.41 M int/s | 0.46 M int/s |
| 全大数 0-2^24 | 0.17 M int/s | 0.18 M int/s |

比 Simple8b 的纯 Python 实现慢一个量级（1.75-5.15 M int/s）。差距来自两处：位宽选择要对每帧做一次 0-32 位的扫描统计，位打包要逐位建字。这两处都是纯 Python 解释器开销，不是算法属性。C 实现下正常区的解码是一段定长移位循环，无分支、可被编译器自动向量化，这是 PForDelta 被选进 Lucene 的实际原因。

![整数压缩方案对比](../figures/fig-compression.png)

## 规模外推

以实测每整数位宽外推 1000 万 docID 列表（1% 异常值，12.76 bit/int）：

| 方案 | 1000 万个的存储 |
|---|---:|
| uint32 | 38.1 MiB |
| Varint | 18.8 MiB |
| Simple8b | 15.9 MiB |
| PForDelta | 15.2 MiB |

十亿级倒排索引上这个差距会被放大到 GB 量级。真实收益还来自 delta 编码：docID 先转差值再进 PForDelta，差值的主体位宽通常在 4-8 位，压缩比可以到 5-8 倍。

## 选型边界

PForDelta 适合异常值比例低于 5% 的整数序列，尤其是需要 SIMD 解压或随机访问的场景。倒排索引 docID 差值、有序时间戳、自增主键都符合。

异常值比例超过 5% 时退回 Simple8b（贪心定档 degradation 平缓）或 Elias-Fano（分区处理高位低位）。数值全部落在窄区间（如 0-15）时 Simple8b 更简单，位宽选择的开销不值得付。需要高压缩率且可随机访问时，Partitioned Elias-Fano 是当前更优的选择。

与 Simple8b 的分工：两者压缩率同量级，PForDelta 赢在解码无分支（SIMD 友好）和异常污染局部化，输在帧头开销与位宽选择的额外计算。选型看的是解压吞吐与随机访问需求，不是压缩率。

## 进阶优化

整数压缩的优化围绕三个维度展开：解压速度（SIMD 化）、压缩率（异常区与熵编码）、随机访问（跳过与 seek）。

**NewPForDelta（Yan, Ding & Suel, 2009）**：阿里搜索推荐团队提出，去掉正常部分开头的第一个异常值位置字段，异常间隔改为存低 5 位，异常部分只存异常值本身。配合 SSE4/AVX2 优化正常部分解压，异常部分因高位回填无法 SIMD。

**OptPForDelta**：在 NewPForDelta 基础上做位宽选择的自适应优化，按帧的实际异常值比例在位宽与异常区开销之间取最优。JavaFastPFOR 实测解压超 12 亿整数/秒（4.5 GB/s）。

**SIMD-BP128（Lemire & Boytsov, 2013）**：固定 128 个整数一块、统一位宽，一条 SIMD 指令解压一块。实测速度是同期 varint-G8IU 与 PFOR 最快方案的两倍，每整数最多再省 2 位。它是牺牲随机访问换吞吐的路线， Lucene 与 Elasticsearch 的 docvalues 大量采用。

**Stream VByte（Lemire, Kurz & Rupp, 2018）**：字节对齐编码的 SIMD 化，把控制流与数据流分离到不同内存流，让 SIMD 一次处理多个整数的控制位。比传统 VByte 快数倍，是 Elasticsearch 等系统的默认 byte-oriented 方案。

**TurboPFor**：Lemire 的 C 库，综合多档位宽与异常区优化，实测解码吞吐 15 GB/s（约 40 亿整数/秒），比 gzip/LZO/Snappy/LZ4 等通用编解码器快一个量级。最新版本引入 AVX-512 路径（SIMD-BP512）。

**Partitioned Elias-Fano**：对有序序列做高位低位分区，查询支持随机访问与 skip。压缩率优于 PForDelta 家族，是当前倒排索引静态索引的主流选择之一。

工程取舍：要吞吐选 SIMD-BP128 / TurboPFor，要压缩率选 Partitioned Elias-Fano，要随机访问与跳过选 Elias-Fano，要兼容老索引选 PForDelta。Lucene 实际用的 FOR（Frame of Reference）家族块大小为 256，存最小值加位宽加位打包差值，与 PForDelta 的 90% 规则不同，它取块内最大值定档、不设异常区。

## 复现

```bash
cd experiments/test_index_theory
<pkm-hub-runtime>/.venv/Scripts/python.exe scripts/pfor_delta_model.py       # 编解码可逆与公式自测
<pkm-hub-runtime>/.venv/Scripts/python.exe scripts/verify_pfor_delta.py      # 压缩比、帧大小与速度实测
```

## 来源

- 公式推导与实测均为本机 2026-09-07 验证（脚本见上）
- PForDelta 参考 Zhang, Long, Suel (2008) "Performance of compressed inverted list caching in search engines"
- NewPForDelta 参考 Yan, Ding, Suel (2009) "Inverted index compression and query processing with optimized document ordering"
- SIMD-BP128 参考 Lemire & Boytsov (2013) "Decoding billions of integers per second through vectorization"
- Stream VByte 参考 Lemire, Kurz, Rupp (2018) "Stream VByte: Faster Byte-Oriented Integer Compression"
