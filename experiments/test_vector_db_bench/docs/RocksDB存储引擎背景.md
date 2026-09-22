# RocksDB 存储引擎背景

本笔记记录 RocksDB 是什么、LSM 树为什么写得快，以及它为何关系到本实验里 SurrealDB 的存储开销。它是背景概念，不是排错记录（坑和根因看 `调试与排错记录.md`），也不是性能结论（数字看 `reports/2026-09-04-真实数据-索引矩阵基准.md`）。

## RocksDB 是什么

Meta（原 Facebook）开源的**嵌入式持久化键值存储引擎**，C++ 库。2012 年从 Google LevelDB fork 出来、2013 年开源，针对 SSD 和服务器工作负载做了深度优化。它本身不是数据库服务，而是被链接进别的程序里提供「存键值对、落盘、读写快」的底层能力。（来源：PingCAP《RocksDB 概述》、腾讯云开发者文章、Wikipedia，2026-09-04 检索）

版本已到 10.x（我看到 10.0/10.2 的发布记录；确切最新版以官方仓库为准，来源之间对版本号有冲突，这条待复核）。

## 核心是 LSM 树，快在写

它写吞吐高，靠的是 LSM-Tree（日志结构合并树），思路和传统 B 树相反。

写入不直接改磁盘上的有序文件（那是随机写、慢），而是分两步：先追加写预写日志 WAL（防崩溃丢失），再写进内存里的有序跳表 MemTable；MemTable 写满后整体刷成磁盘上的 SST 文件，按层组织（默认最多 6 层，每层约是上一层的 10 倍）。后台再把相邻层的 SST 合并、压缩。本质是**把随机写转换成顺序写**，机械硬盘上能快约 100 倍、SSD 上也有数倍差距。读则靠每层文件的索引和 Bloom filter 定位。（来源：PingCAP《RocksDB 概述》、腾讯云文章）

这套「每条写入独立成 entry、靠后台压缩整理」的模型，就是下面那笔存储开销的根源。

## 为什么本实验会碰到它

SurrealDB 拿 RocksDB 当存储后端（`surreal start ... rocksdb:data/surreal.db`）。主报告把 SurrealDB 的 **7.47x 存储开销**归因于 RocksDB 的文档存储模型：它把每条记录的整个文档（含 `pk`、`emb` 数组、索引项）当作一个 value 存，11,293 条就有 11,293 个 RocksDB entry，每个都带 key 前缀和 WAL 记录。LSM 树为了写性能接受的正是这种「每条独立成 entry、后台压缩」的空间换时间。

需要说清：**这笔开销是通用文档数据库套在 RocksDB 上的固有代价，不是向量索引本身的开销。** 这是主报告的分析判断（工程上成立的归因，不是绝对事实）。对比之下 LanceDB 用列式存储把向量紧凑排列，总倍数只有 1.04x，是两个极端。

## 谁还在用 RocksDB

TiKV（TiDB 的存储层）、YugabyteDB 的 DocDB、Kafka Streams 的本地状态存储等都用 RocksDB 或它的改版。（来源：Wikipedia、db-engines.com，2026-09-04 检索）

## 来源

- RocksDB 概述（PingCAP 中文文档）：https://docs.pingcap.com/zh/tidbcloud/rocksdb-overview/
- LSM 树与 RocksDB 架构（腾讯云开发者）：https://cloud.tencent.com/developer/article/2565007
- RocksDB（Wikipedia）：https://en.wikipedia.org/wiki/RocksDB
- 均为 2026-09-04 检索；版本号与支持矩阵会随时间变化，引用前建议复核官方仓库。
