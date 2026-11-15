# 数据结构与存储成本：从物理开销推导到工程系统选型

> **阅读导航**：[项目首页](../../../README.md) · [实验地图](../README.md) · [数据结构算法实测与复现](../../../experiments/data-structure-costs/README.md)

在 AI 系统工程、大规模搜索引擎、向量数据库内核以及高并发数据管道的设计中，绝大多数吞吐瓶颈与内存溢出，最终都可以追溯到**底层数据结构的理论复杂度模型与物理内存布局的失配**。

选择一个数据结构，核心是在**点查延迟、范围扫描能力、写入放大、内存带宽占用以及空间放大**这五大物理约束之间做严格的代数权衡。

本专题结合 CPU 缓存行、SIMD 向量化指令与顺序 I/O 等现代硬件特征，整理了 36 种经典数据结构的理论推导、位级压缩编码与物理布局，提供可计算的工程选型参考。

---

## 体系化核心专文导览

| 核心专题 | 涵盖算法与核心结构 | 要回答的底层工程问题 | 关键结论与物理机制 |
|---|---|---|---|
| **[概率数据结构与基数估计](概率数据结构与基数估计.md)** | Bloom Filter、Cuckoo Filter、Count-Min Sketch、HyperLogLog / HLL++ | 如何用极小常数空间（几 KB 到几 MB）完成海量数据的成员判定、频次统计与百亿级 UV 去重？ | 推导 Bloom Filter 最优位宽（1% 误判仅需 9.6 比特）；推导 Cuckoo Filter 指纹踢出与动态删除机制；推导 HLL 调和平均消除离群值；解析 HLL++ 稀疏/稠密表示切换 |
| **[读写放大与有序索引选型](读写放大与有序索引选型.md)** | B+ 树、LSM-Tree、跳表（Skip List）、自适应基数树（ART） | 磁盘块读优化、高吞吐顺序追加与纯内存并发环境下，存储引擎如何权衡读写放大？ | 剖析 B+ 树原地更新的写放大（10x~50x）；推导 LSM-Tree 分层合并（Compaction）与布隆过滤器剪枝优化；解析跳表基于概率高度的无锁并发；剖析 ART 动态节点自适应（Node4~256）与 SIMD 匹配 |
| **[哈希表演进与内存冲突解决](哈希表演进与内存冲突解决.md)** | 拉链法、Robin Hood、Hopscotch、Swiss Table（SIMD 并行探查）、可扩展哈希 | 现代 CPU 体系结构下，为什么单次指针跳转的延迟远高于连续内存比对？工业级哈希表如何突破装载因子瓶颈？ | 剖析拉链法指针膨胀与 Cache Miss 短板；推导 Robin Hood“劫富济贫”PSL 均化机制；深度拆解 Swiss Table 16 字节控制元数据与 128 位 SIMD 并行过滤；解析磁盘可扩展哈希的局部深度分裂 |
| **[列式存储与倒排索引压缩编码](列式存储与倒排索引压缩编码.md)** | Varint、Delta 编码、PFor-Delta、Simple8b、列式 RLE、倒排索引与 WAND 算法 | 海量有序整型倒排链表与大数据分析列如何实现 1~3 比特的高效压缩并维持 GB/s 解码吞吐？ | 剖析 Varint 延续位分支预测瓶颈；推导 PFor-Delta 异常值抽离与 SIMD 批量解包；解析 Simple8b 64 位无分支查表装箱；详解倒排索引跳表指针与 WAND（Weak AND）动态剪枝 |
| **[数据格式体积与速度实测](数据格式体积与速度实测.md)** | JSON 数组、JSONL、Excel（XLSX）、SQLite 数据库 | 同一份结构化数据，写成四种格式在真实文件体积、读写速度、追加单次耗时与索引膨胀上差多少？ | 对照记录中的文件体积、读写、追加和索引代价；XLSX 的分阶段耗时为两次测量相减，无法据此确定 XML 解析占比 |

---

## 实验代码、验证脚本与交互式工具

本专题不仅提供严密的数学推导与架构解析，还在实验区提供了完整的可复算代码与动态交互工具：

- **[实验代码与验证程序（experiments/data-structure-costs/）](../../../experiments/data-structure-costs/README.md)**：
  - 36 个 `scripts/*_model.py`：输入数据规模与硬件参数，精确输出理论空间与计算开销；
  - 37 个 `scripts/verify_*.py`：基于纯 Python 算法原型，实测小规模数据下的统计指标，与理论模型相互校验；
  - 数据格式实测脚本：`benchmark_json_xlsx.py`、`benchmark_xlsx_sqlite_size.py`、`benchmark_sqlite_pandas_queries.py`。
- **[交互式性能与成本计算器](../../../experiments/data-structure-costs/performance_calculator.html)**：
  - 无需安装任何环境，使用浏览器打开即可运行；
  - 动态拖拽数据量、误判率、键值长度与硬件带宽，实时渲染 Chart.js 空间与延迟曲线，供系统设计前期做快速容量规划（Capacity Planning）。
- **[数据结构机制拓扑图集](../../../experiments/data-structure-costs/figures/)**：
  - 收录 30+ 张高清晰度数据结构物理内存布局与访问路径示意图。
