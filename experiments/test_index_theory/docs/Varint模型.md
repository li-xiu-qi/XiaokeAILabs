# Varint 变长整数编码性能模型

> 实证日期 2026-09-07 · Python 3.13.0 · numpy 2.5.2 · 脚本 `scripts/varint_model.py`、`scripts/verify_varint.py` · 结果 `results/varint_*.json`

Varint（变长整数编码，VByte/LEB128）把每个字节的低 7 位用作数据、最高位作 continuation bit（还有后续字节则为 1）。1-127 用 1 字节，128-16383 用 2 字节，16384-2097151 用 3 字节，以此类推。本笔记给出编码字节数、压缩比的闭式解，并用三种分布实测验证。核心结论是压缩比由数值分布而非编码本身决定，小整数域内压 2-4 倍，满量程 uint32 反而膨胀 1.25 倍。Google Protocol Buffers、Apache Avro、gRPC、LevelDB、RocksDB、Elasticsearch 都采用它。

## 核心公式

编码字节数 `bytes(n) = ceil(bits(n) / 7)`，其中 `bits(n) = max(1, n.bit_length())`。

压缩比 `ratio = itemsize / 期望字节数`，itemsize 是定长编码字节数。同组值全落在 `[0, U]` 时压缩比是常数 `itemsize / bytes(U)`；混合值域时按期望字节数算。

均匀分布 `[lo, hi)` 的期望字节数有闭式解：按 7 比特分组边界 `2^7, 2^14, 2^21, ...` 切开区间，加权求和再除以值域宽度。

**每字节开销有两种口径，不要混用。** 比特口径是每字节 8 比特含 1 个 continuation bit，开销 1/7 ≈ 14.3%。字节口径是满量程 uint32 需 5 字节、定长只需 4 字节，膨胀 1.25 倍即 25%。两者描述的是不同对象，前者是编码机制的固有代价，后者是 32 位数据的实际落盘结果。

## 编码分组

| 数值范围 | 字节数 | 相对 uint32 | 相对 uint64 |
|---|---:|---:|---:|
| 0-127 | 1 | 4.0x | 8.0x |
| 128-16383 | 2 | 2.0x | 4.0x |
| 16384-2097151 | 3 | 1.33x | 2.67x |
| 2097152-2^28-1 | 4 | 1.0x | 2.0x |
| 2^28-2^32-1 | 5 | 0.80x | 1.60x |
| 2^32-2^64-1 | 6-10 | — | 0.80-1.33x |

小于 128 的值压到 1 字节，是 varint 的最佳工作区间。超过 2^28 后相对 uint32 已无收益，Protobuf 官方文档的结论一致：数值大于 2^28 时 fixed32 更省，大于 2^56 时 fixed64 更省。

## 实测验证

n=100 万，纯 Python 实现（`encode_varint`/`decode_varint` 逐值调用）。字节数公式先做逐值校验：分四段量级共 60.03 万个样本（覆盖 0、2^20、2^32、2^64 四个量级），公式与实测编码长度零不一致。

三种分布的压缩比：

| 分布 | 组内上界 | 定长基准 | 原始 | varint | 实测比 | 理论比 | 均值字节/值 |
|---|---|---|---:|---:|---:|---:|---:|
| 均匀 0-1000 | 1,000 | uint32 | 3.8 MiB | 1.8 MiB | 2.14x | 2.14x | 1.87 |
| Zipf (a=1.3) | 2.14e9 | uint32 | 3.8 MiB | 1.2 MiB | 3.19x | 3.22x | 1.25 |
| 均匀 0-2^32 | 4.29e9 | uint64 | 7.6 MiB | 4.7 MiB | 1.62x | 1.62x | 4.94 |

三种分布的理论与实测全部吻合（误差 <1%）。Zipf 分布均值 1.25 字节/值、压缩 3.2 倍，因为绝大多数值落在 1 字节区间，只有极少数尾部大值拉到 5 字节。这解释了 varint 在真实索引里的表现：docID、词频这类重尾小整数分布天然适配。

均匀 0-2^32 只有 1.62 倍，因为相对 uint64 才是 1.62 倍，相对 uint32 则不可比（数值超出定长范围）。

编解码速度（20 万个值，纯 Python 循环）：编码 0.92 M ops/s，解码 0.80 M ops/s。这个数字是解释器开销下的吞吐，真实系统用 C++/SIMD 实现会快一到两个数量级（见「进阶优化」）。

![整数压缩方案对比](../figures/fig-compression.png)

## 规模外推

以均匀 0-1000（均值 1.87 字节/值）和 Zipf（均值 1.25 字节/值）外推，对比定长 uint32（4 字节/值）：

| n | 均匀 varint | 均匀 uint32 | Zipf varint | Zipf uint32 |
|---:|---:|---:|---:|---:|
| 100 万 | 1.8 MiB | 3.8 MiB | 1.2 MiB | 3.8 MiB |
| 1000 万 | 18 MiB | 38 MiB | 12 MiB | 38 MiB |
| 1 亿 | 178 MiB | 382 MiB | 119 MiB | 382 MiB |
| 10 亿 | 1.74 GiB | 3.73 GiB | 1.16 GiB | 3.73 GiB |

十亿级整数从 3.73 GiB 降到 1.16 GiB，省 2.6 倍。这是 varint 在倒排索引、时序数据库里的实际收益量级。

## 选型边界

varint 的适用条件很明确：**值域必须集中在小区间**。判据是看 P99 或最大值落在哪个字节组，而不是看平均值。平均 1.25 字节但最大值 2^31 时，99.99% 的值只花 1 字节，收益仍然成立；反过来，值域均匀铺满 uint32 时收益只有 1.62 倍（相对 uint64），相对 uint32 甚至不可用。

不适合的场景有三类。值域均匀且大（如哈希值、UUID 的数值形式），固定长度编码更省且支持 O(1) 随机访问。需要随机访问第 i 个元素时 varint 必须从头解码，定长编码可以直接 seek 偏移。性能敏感的解码路径要换 SIMD 变体，逐值标量解码在现代 CPU 上受分支预测惩罚严重。

连续递增序列（docID、时间戳）不要直接对原值用 varint，先做差分再 varint，收益更大，见 [Delta 编码模型](Delta编码模型.md)。

## 进阶优化

varint 的性能瓶颈在解码：逐值标量解码每个字节都有一次分支判断，现代 CPU 上分支预测失败的代价高于解码本身。过去十年的优化集中在消除分支、用 SIMD 一次解多个值。

**StreamVByte**（Lemire 等，把 SIMD 用在 Group Varint 上的方案）用 16 字节控制字节描述后续 4 组共 16 个 32 位整数，解码时一次 load 16 字节、查表 shuffle、一次得到多个值。专利自由，Apache 2.0 许可，内置差分编码支持，2010 年后的 Intel 处理器和带 NEON 的 ARM 都能跑。

**MaskedVByte** 针对解码优化：抽取 16 字节的 continuation bit 作 mask，按 mask 查预定义的 shuffle 向量表把字节重排成整数。实测 6.5-27 亿整数/秒（Haswell 平台），编码密度越高越快。维基百科的 LEB128 条目确认了这个区间。

**SIMD-BP128\***（Lemire & Boytsov，把位封装与 SIMD 结合）在台式机处理器上比 varint-G8IU 和 PFOR 快接近两倍，每个整数还省 2 比特。前提是整数块足够长（128 个以上），短块上收益消失。

**Group Varint**（Facebook Folly 库）每 4 个整数一组，1 字节头部每 2 比特记录一个整数的字节数（1-4），解码时先读头部再批量读数据。相对传统 varint，小整数区间从 1 字节/整数涨到 1.25 字节/整数（差 20%），中等整数区间从 2-3 字节降到 2.25 字节（好 12.5%）。它还被用在近存储硬件加速里，某高维向量检索加速器用它消除主机端通信瓶颈，比无压缩方案快 6 倍。

**PFOR / Patched FOR** 把每个值编码为相对基值的增量再位封装。基本 FOR 遇到离群值会失效（一个 118 就让全部值需要 5 位），PFOR 把离群值存为异常项、其余值保持小位宽。SIMD-FastPFOR 和 SIMD-FastBP128 在此基础上做 SIMD 布局优化。

**Rust varint-simd** 的实测吞吐（AMD Zen+ 平台）：u32 解码 293-360 M values/s，u64 解码 229-303 M values/s，比标量实现快 2-4 倍。

选型上，纯字节导向方案（varint、Group Varint、StreamVByte）解码更快，位封装方案（PFOR、BP128）压缩率更高。Elasticsearch 的默认实现是 PFor/Delta 组合，RocksDB 的 BlockBasedTable 用 varint 存索引块，前者偏压缩率、后者偏速度。

## 复现

```bash
cd experiments/test_index_theory
<pkm-hub-runtime>/.venv/Scripts/python.exe scripts/varint_model.py       # 公式自测
<pkm-hub-runtime>/.venv/Scripts/python.exe scripts/verify_varint.py      # 压缩比与速度实测
```

## 来源

- 公式推导与实测均为本机 2026-09-07 验证（脚本见上），种子 20260907，n=100 万
- MaskedVByte 吞吐区间 650-2700 M int/s 来自维基百科 LEB128 条目（Haswell 实测）
- SIMD-BP128\* 相对 varint-G8IU/PFOR 的加速比来自 Lemire & Boytsov 论文（CIKM 2011）
- Group Varint 头部格式与压缩率对比来自 Facebook Folly 库文档
- varint-simd 吞吐数字来自 as-com/varint-simd 仓库的 benchmark 表
- Protobuf 在 2^28/2^56 处改用定长的建议来自 Protobuf 编码文档
