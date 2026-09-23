# 索引本质测量与理论性能模型

这个实验不测工具，测索引和存储引擎本身的数学性质。涵盖八个层次：

**通用索引与存储引擎**：B+树、LSM-tree、列式存储、哈希索引。它们的点查、写入、扫描、存储行为由数据结构和算法决定，这些关系式与用哪个产品无关。目标是建立可计算的性能模型（cost model），输入数据规模就能递推出延迟、IO、存储、写入放大。

**文本与查询引擎**：倒排索引、Join 算法、BM25 排序。倒排索引是文本搜索的核心结构，Join 是关系数据库查询的基础，BM25 是倒排索引上最主流的打分函数。

**基础算法**：跳表、布隆过滤器、LRU 缓存、外部排序。数据库系统的标准组件。

**概率数据结构**：布谷鸟过滤器、Count-Min Sketch、HyperLogLog。快速存在性判断、频率估计、基数估计。

**近似最近邻**：LSH、SimHash、MinHash。高维相似度搜索的另一条技术路线。

**前缀搜索与区间查询**：Trie、ART、Segment Tree、Fenwick Tree。字符串前缀搜索和区间聚合。

**哈希表优化**：Robin Hood、Swiss Table、Hopscotch。现代哈希表的三种高性能实现。

**Heavy Hitters**：Misra-Gries、Space-Saving、Lossy Counting。数据流中的高频项检测。

**压缩编码**：Varint、Delta、Simple8b、PForDelta。整数压缩的标准方案，倒排索引和列式存储的基础。

**动态哈希**：Cuckoo Hashing、Extendible Hashing。支持扩容的哈希表。

**基数估计扩展**：HyperLogLog++、FID-Sketch。大规模基数估计的改进版。

**数据格式**：xlsx、sqlite、json、jsonl。同一份数据在不同格式下的体积和速度。理论模型估算体积，实测 benchmark 验证。

**主流数据库架构映射**：PostgreSQL、MySQL InnoDB、SQLite、DuckDB、RocksDB、Elasticsearch、Redis。

## 全部算法索引（37 个）

每个算法都有两件套留在本仓：理论模型自测脚本（`scripts/<algorithm>_model.py`，纯公式计算器）+ 实测验证脚本（`scripts/verify_<algorithm>.py`，自制结构实测）。

**公式推导与成本模型结论属可迁移知识，2026-09 起分批迁往配套知识库**（计算机科学基础下的「索引与数据结构成本模型」系列笔记）。已迁走的算法在下表不再有 `docs/` 链接，其实测数据表随推导一起迁出；未迁的算法文档仍在 `docs/`，在对应小节注明。换掉机器、数据、版本仍成立的内容归知识库，本机实测的过程记录留本仓，边界见仓根 AGENTS.md。

### 通用索引与存储引擎（4 个，已迁）

| 算法 | 核心机制 |
|---|---|
| B+树 | 多路平衡树，根到叶路径 |
| LSM-tree | 内存写缓冲 + 多层有序文件 |
| 列式存储 | 按列存、压缩、向量化 |
| 哈希索引 | 桶数组 + 冲突处理 |

### 文本与查询引擎（3 个，文档在仓）

| 算法 | 核心机制 | 文档 |
|---|---|---|
| 倒排索引 | 词典 + 倒排列表，TF-IDF/BM25 | `docs/倒排索引模型.md` |
| Join 算法 | NLJ、Hash Join、Sort-Merge Join | `docs/Join算法模型.md` |
| BM25 排序 | IDF + TF 饱和 + 长度归一化 | `docs/BM25模型.md` |

### 基础算法（4 个，已迁）

| 算法 | 核心机制 |
|---|---|
| 跳表 | 多层链表，O(log n) 查找 |
| 布隆过滤器 | 位数组 + 哈希函数，概率判断 |
| LRU 缓存 | 双向链表 + 哈希表，淘汰最久未使用 |
| 外部排序 | 分块排序 + 多路归并 |

### 概率数据结构（4 个，已迁）

| 算法 | 核心机制 |
|---|---|
| 布谷鸟过滤器 | 布谷鸟哈希 + 指纹，支持删除 |
| Count-Min Sketch | 二维计数数组，频率估计 |
| HyperLogLog | 桶寄存器 + 前导零，基数估计 |
| HyperLogLog++ | 稀疏表示 + 64 位哈希，精度更高 |
| FID-Sketch | 4 位概率计数，Count-Min Sketch 的改进版 |

### 近似最近邻（3 个，文档在仓）

| 算法 | 核心机制 | 文档 |
|---|---|---|
| LSH | 局部敏感哈希，S-curve 筛选 | `docs/LSH模型.md` |
| SimHash | 随机超平面投影，海明距离 | `docs/SimHash模型.md` |
| MinHash | 最小哈希，Jaccard 相似度估计 | `docs/MinHash模型.md` |

### 前缀搜索与区间查询（4 个，已迁）

| 算法 | 核心机制 |
|---|---|
| Trie | 字典树，前缀 O(L) 查找 |
| ART | 自适应基数树，节点大小自适应 |
| Segment Tree | 线段树，O(log n) 区间查询 |
| Fenwick Tree | 树状数组，O(log n) 前缀和 |

### 哈希表优化（3 个，文档在仓）

| 算法 | 核心机制 | 文档 |
|---|---|---|
| Robin Hood Hashing | 开放寻址 + 探测距离均衡 | `docs/RobinHood模型.md` |
| Swiss Table | SIMD 分组比较，2-3x std::unordered_map | `docs/SwissTable模型.md` |
| Hopscotch Hashing | 邻域位图，负载因子 90%+ | `docs/Hopscotch模型.md` |

### Heavy Hitters（3 个，已迁）

| 算法 | 核心机制 |
|---|---|
| Misra-Gries | k 计数器，确定性，频率 > n/k 保证 |
| Space-Saving | m 计数器 + 最小堆，O(1) 更新 |
| Lossy Counting | 分桶删除，误差 ε×N |

### 压缩编码（4 个，文档在仓）

| 算法 | 核心机制 | 文档 |
|---|---|---|
| Varint | 每字节 7 位数据 + 1 位 continuation | `docs/Varint模型.md` |
| Delta Encoding | 差分 + Zigzag + Varint | `docs/Delta编码模型.md` |
| Simple8b | 64 位字，4 位选择器 + 60 位数据 | `docs/Simple8b模型.md` |
| PForDelta | 帧内统一位宽 + 异常区 | `docs/PForDelta模型.md` |

### 动态哈希（2 个，文档在仓）

| 算法 | 核心机制 | 文档 |
|---|---|---|
| Cuckoo Hashing | 两表 + 递归踢出，O(1) 最坏查找 | `docs/布谷鸟哈希模型.md` |
| Extendible Hashing | 目录 + 桶分裂，O(1) 查找 | `docs/可扩展哈希模型.md` |

### 向量索引（1 个统一模型，文档在仓）

| 算法 | 核心机制 | 文档 |
|---|---|---|
| 向量索引 | FLAT/SQ/PQ/IVF/HNSW 统一公式 | `docs/向量索引统一模型.md` |

向量索引详细文档：`docs/向量索引的存储与内存模型.md`、`docs/向量索引的查询延迟模型.md`、`docs/向量索引的召回特性.md`。

### 数据格式（1 个统一模型，文档在仓）

| 算法 | 核心机制 | 文档 |
|---|---|---|
| 数据格式 | xlsx/sqlite/json/jsonl 体积与速度 | `docs/数据格式模型.md` |

数据格式的详细实测文档见 `docs/数据格式体积与速度实测.md`（从 test_data_formats_bench 合并）。

### 架构映射（1 个，已迁）

7 个数据库（PostgreSQL、MySQL InnoDB、SQLite、DuckDB、RocksDB、Elasticsearch、Redis）的索引和存储设计映射已迁往配套知识库，本仓不再维护原文。

## 过程记录（留仓）

这些是与本实验绑定的过程文档，不迁：

- `docs/向量索引实验交接笔记.md`：三轮错误判断的修正链路、未完成清单
- `docs/faiss 图量化索引标定踩坑记录.md`：绑 faiss 版本的标定踩坑
- `docs/数据格式体积与速度实测.md`：数据格式的实测过程记录

## 实测状态

全部算法的模型自测通过，验证脚本实际运行成功，结果写入 `results/`。已迁算法的实测数据表随推导一起在知识库；未迁算法的实测数据表仍在 `docs/`。

## 复现

```bash
cd experiments/test_index_theory
# 跑单个算法的模型自测
<pkm-hub-runtime>/.venv/Scripts/python.exe scripts/<algorithm>_model.py
# 跑单个算法的实测验证
<pkm-hub-runtime>/.venv/Scripts/python.exe scripts/verify_<algorithm>.py
```
