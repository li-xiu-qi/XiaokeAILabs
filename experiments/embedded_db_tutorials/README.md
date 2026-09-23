# 嵌入式数据库学习：SQLite 与 DuckDB 及其向量检索扩展

两个嵌入式数据库的使用教程，主题完全同构：先学库本身，再学它挂上向量检索扩展之后怎么用。SQLite 和 DuckDB 都不需要独立服务器，数据库就是一个文件（SQLite）或进程内引擎（DuckDB），适合本地应用、桌面工具、边缘部署，也都通过扩展支持 ANN 向量检索，是轻量 RAG 的常见底座。

## 子目录

| 目录 | 库 | 内容 |
|---|---|---|
| `sqlite/` | SQLite 3.x + sqlite-vec | 基础使用（建库建表、CRUD、SQL 查询）、sqlite-vec 向量扩展入门与进阶、KNN 检索脚本、官方文档副本（中英） |
| `duckdb/` | DuckDB + vss/fts 扩展 | 入门教程（建表、SQL、parquet 读取）、VSS 向量检索扩展（HNSW 索引、Sentence Transformers 实战）、FTS 全文检索扩展（BM25） |

## 两个库的向量扩展对照

同是在关系库里做向量检索，两条路子的设计不一样，这是本系列最值得对照着看的地方：

**SQLite 走 sqlite-vec 扩展**：用 `serialize_float32` 把向量序列化成 blob 存储，`vec0` 虚拟表建索引，`vec_distance_cosine` 等函数算距离，KNN 查询要手写 MATCH 语法。Windows 下需要扩展二进制（`sqlite/vec0.dll`），加载方式见 sqlite 子目录的 notebook。

**DuckDB 走 vss 扩展**：向量直接存 `ARRAY` 列，`CREATE INDEX ... USING HNSW` 建索引，查询还是普通 SQL 的 `ORDER BY 距离 LIMIT k`，扩展带优化器规则，能避免全表扫描。新版还引入了 HNSW_INDEX_JOIN 算子。

对照着读能看清「向量检索塞进 SQL 引擎」的两种集成策略：SQLite 用虚拟表另起一套语法，DuckDB 把向量当一等类型融进 SQL。

## 学习顺序

先 `sqlite/sqlite_tutorial.ipynb` 或 `duckdb/duckdb_tutorial.ipynb` 把库本身跑通（没学过 SQL 的话），再进各自的向量检索 notebook。熟悉其中一个之后，直接对照着看另一个的向量部分，差别比各自从头学更明显。

两个库的扩展文档副本（`docs/` 下）来自官方仓库，部分图片与相对链接指向未随附的上游结构，阅读时以 notebook 里的实测为准。
