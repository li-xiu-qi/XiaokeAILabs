# 向量数据库本地基准

在同一台机器、同一份数据上，对比 6 个向量库在**不同索引机制**下的查询延迟、召回率、索引构建时间和落盘占用。

**被测对象**：Milvus v3.0.0（Docker）/ Qdrant 1.19.0（Docker）/ SurrealDB 3.2.4（Windows 原生）/ LanceDB 0.38.0（嵌入式）/ ChromaDB 1.5.9（嵌入式）/ sqlite-vec 0.1.9（嵌入式）。

**测试数据**：20 Newsgroups 全部 11,293 篇训练集，all-MiniLM-L6-v2 生成 384 维向量；test 集分层采样 40 条做查询，召回率以暴力精确 top-k 为基准。

## 这个实验回答什么

不只是「SurrealDB vs Milvus 谁快」，而是三件更细的事：

- **索引类型维度**：每个库跑它实际支持的索引（FLAT / IVF / HNSW / PQ / DISKANN 等），同库不同索引、同索引跨库都能比。
- **索引构建时间**：`build_index()` 单独计时，含 Milvus 的异步就绪等待。
- **存储三段拆分**：把落盘拆成「向量 / 正文文本 / 索引」三部分分别计量，看清膨胀来自哪里。

## 快速复现

```bash
# 起服务（Milvus / Qdrant 走 Docker；SurrealDB 用 bin/surreal.exe）
docker compose -f config/milvus-standalone-compose.yml up -d
MSYS_NO_PATHCONV=1 docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
  -v "$(pwd)/data/qdrant:/qdrant/storage" qdrant/qdrant:latest

# 装依赖
./.venv/Scripts/python.exe -m pip install -r requirements.txt

# 索引矩阵基准（11293 x 384，20 组，约 12 分钟）
./.venv/Scripts/python.exe scripts/index_bench.py

# 文本存储成本（每库 A/B：向量+pk vs 向量+pk+文本，约 5 分钟）
./.venv/Scripts/python.exe scripts/index_bench.py --text-storage
```

结果落在 `results/`（`indexbench-*.json`、`textbench-*.json`）。

四份报告共内嵌 7 张图，全部由 `figures/make_figures.py` 从结果数据生成（主报告 4 张：召回-延迟帕累托、存储放大倍数、构建时间、写入吞吐；随机向量报告 1 张：SurrealDB/Milvus ef 异常；五库对比 1 张：LanceDB 调参根因；部署报告 1 张：部署能力矩阵）：

```bash
# 生成/更新全部 7 张图（需带 matplotlib 等绘图依赖的环境，本目录无 .venv）
python figures/make_figures.py
```

图与脚本都在 `figures/`。主报告 4 张读 `results/indexbench-*.json`；另 3 张数据来自对应报告里的表格（存档/调参根因，无对应 JSON）。改主报告数据后重跑 `index_bench.py` 再跑本命令即可刷新前 4 张。

## 文档导航

| 文档 | 内容 |
|---|---|
| [环境搭建.md](环境搭建.md) | 环境怎么搭、测试怎么设计、存储测量口径与踩坑 |
| [reports/2026-09-04-真实数据-索引矩阵基准.md](reports/2026-09-04-真实数据-索引矩阵基准.md) | **主报告**：索引矩阵 × 构建时间 × 存储三段拆分，内嵌 4 张对比图，唯一该引用的性能与存储结论 |
| [reports/2026-09-05-内存占用基准.md](reports/2026-09-05-内存占用基准.md) | 内存（RAM）占用基准：五库默认索引内存增量 + 索引级内存对比（LanceDB 四种 / Milvus 七种），回答「内存占用也和索引类型有关系吗」 |
| [reports/2026-09-06-混合检索BM25对比.md](reports/2026-09-06-混合检索BM25对比.md) | 混合检索（向量+BM25，RRF 融合）实测：双 ground truth 设计、两个导致结论错误的缺陷（查询文本与向量错位 / 阈值未校准）、修完重跑的数据与解释边界 |
| [reports/2026-09-04-部署方式与平台支持.md](reports/2026-09-04-部署方式与平台支持.md) | 各库部署形态、Windows 支持程度、选型建议 |
| [reports/2026-09-03-真实数据-五库对比.md](reports/2026-09-03-真实数据-五库对比.md) | 早期五库对比（真实 embedding，单一默认索引） |
| [reports/2026-09-03-随机向量专项测试.md](reports/2026-09-03-随机向量专项测试.md) | 合成随机向量探底，召回失真，仅留档 |
| [docs/调试与排错记录.md](docs/调试与排错记录.md) | 所有踩坑、被推翻的假设、测量误差的根因 |
| [docs/RocksDB存储引擎背景.md](docs/RocksDB存储引擎背景.md) | 背景概念：RocksDB 与 LSM 树是什么，为何关系到 SurrealDB 的存储开销 |
| [docs/向量搜索引擎底层实现.md](docs/向量搜索引擎底层实现.md) | 背景概念：五个库各自用什么做向量搜索（FAISS 封装 / Rust 自研 / 无 ANN） |
| [docs/六库背后的人物与公司背景.md](docs/六库背后的人物与公司背景.md) | 背景概念：六个库的公司、创始人、融资与维护模式 |
| [docs/六库索引类型支持清单.md](docs/六库索引类型支持清单.md) | 背景概念：六个库各自支持的向量索引类型总览，标注哪些在本次基准中实测过 |
| [docs/向量索引类型区别.md](docs/向量索引类型区别.md) | 背景概念：FLAT、IVF、HNSW、PQ、SQ、DISKANN、SCANN、ANNOY、RaBitQ 的原理、优缺点与选型 |
| [docs/主流分词器横向对比.md](docs/主流分词器横向对比.md) | 主流分词器分类与对比（jieba / HanLP / THULAC / IK / Lucene / ICU / tantivy / 子词分词器），含 9 个源码仓库与底层切分规则 |
| [docs/六库分词器实现与中文支持.md](docs/六库分词器实现与中文支持.md) | 六库 BM25 的分词器实现（jieba / ICU / whitespace / snowball）与中文支持实测：只有 Milvus 和 LanceDB 能用，Milvus 靠内置 jieba 和 language_identifier 自动识别中英文 |
| [docs/六库BM25支持度与接入难度.md](docs/六库BM25支持度与接入难度.md) | 六库 BM25 支持度实测+源码对比（Milvus/LanceDB/SurrealDB 实测通过，ChromaDB/Qdrant 源码确认，sqlite-vec 无），含接入难度分层；混合检索实测已拆到独立报告 |
| [docs/FAISS预计算表内存限制与Milvus分段.md](docs/FAISS预计算表内存限制与Milvus分段.md) | 背景概念：FAISS 的 IVFPQ 预计算表内存限制（1GB 说法的来源），Milvus 用分段绕开单索引规模上限 |
| [docs/各库体积与安装包数据.md](docs/各库体积与安装包数据.md) | 七个库在 clone 源码 / pip 包 / Docker 镜像 / 二进制安装包 / 运行时内存五个维度的体积对比，含 seekdb v1.3.0→v1.4.0 轻量化进展 |

每份报告开头都写了「这个文档是干嘛的」。性能与存储结论只看真实数据主报告。

## 一句话结论

LanceDB 存储最省（列式压缩），Milvus IVF_SQ8 是存储效率最高的 ANN 索引；Qdrant 落盘被固定页预分配主导、文本增量测不出（详见主报告）；SurrealDB 走 RocksDB 通用文档存储，向量开销最高。详细数字与选型建议看 `reports/2026-09-04-真实数据-索引矩阵基准.md`。

> `reports/2026-09-03-随机向量专项测试.md` 用的是 768 维均匀随机向量（ANN 最坏情况、召回失真），仅留档，不要引用它的数字。
