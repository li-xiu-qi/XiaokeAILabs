# DuckDB 向量相似性搜索扩展的最新更新

> 原标题：What's New in the Vector Similarity Search Extension?

DuckDB 又向成为向量数据库迈进了一步！这篇文章介绍了自首次发布以来 DuckDB 向量相似性搜索（VSS）扩展的一些新特性和改进。

## 索引创建速度提升

DuckDB 在 HNSW 索引创建速度方面有了显著提升：

- **预知行数优势**：在已填充数据的表上创建 HNSW（Hierarchical Navigable Small Worlds）索引比先创建索引再插入数据更高效，因为已知总行数可以更准确地预测索引大小和优化线程分配
- **工作分配优化**：初始版本中，工作分配较为粗糙，仅为每个"行组"（默认约 120,000 行）安排一个额外工作者线程
- **缓冲机制改进**：新版本引入了额外的缓冲步骤，实现了更细粒度的工作分配和更智能的内存管理，有效减少线程间竞争
- **用户体验提升**：新增了索引创建过程的进度条，让用户能够直观了解创建进度

## 新增距离函数

为解决原有距离函数的一致性问题，VSS 扩展新增了更多实用的距离度量：

- **原有函数回顾**：初始版本支持三种距离函数：
  - `array_distance`（返回值接近 0 表示相似）
  - `array_cosine_similarity`（返回值为 1 表示相同）
  - `array_inner_product`

- **新增函数**：
  - `array_cosine_distance`：等同于 `1 - array_cosine_similarity`
  - `array_negative_inner_product`：等同于 `-array_inner_product`

- **一致性改进**：这些新函数可通过 HNSW 索引加速，使所有支持的度量在查询模式和排序上保持一致

- **扩展支持**：
  - 为动态大小的 `LIST` 数据类型添加了等效距离函数（前缀为 `list_`）
  - 将 `<=>` 二元运算符改为 `array_cosine_distance` 的别名，与 PostgreSQL 的 `pgvector` 扩展语义保持一致

## 索引加速的 "Top-K" 聚合

DuckDB 核心功能增强了聚合函数的能力：

- **函数增强**：新增了 `min_by` 和 `max_by` 聚合函数（及其别名 `arg_min` 和 `arg_max`）的重载版本

- **灵活参数**：这些新版本接受可选的第三参数 `n`，用于指定要保留的 top-k 元素数量，并将结果输出为排序的 `LIST`

- **示例用法**：

```python
import duckdb
import numpy as np
import pandas as pd

# 连接到DuckDB并加载VSS扩展
con = duckdb.connect(database=':memory:')
con.install_extension('vss')
con.load_extension('vss')

# 创建一个示例表
con.execute("""
CREATE OR REPLACE TABLE vecs AS
    SELECT
        row_number() OVER () AS id,
        [a, b, c]::FLOAT[3] AS vec
    FROM
        range(1,4) AS x(a), range(1,4) AS y(b), range(1,4) AS z(c);
""")

# 找到向量最接近 [2, 2, 2] 的前 3 行
result = con.execute("""
SELECT
    arg_min(vecs, array_distance(vec, [2, 2, 2]::FLOAT[3]), 3)
FROM
    vecs;
""").fetchall()
print(result[0][0])
```

查询结果：

```
[{'id': 14, 'vec': [2.0, 2.0, 2.0]}, {'id': 13, 'vec': [2.0, 1.0, 2.0]}, {'id': 11, 'vec': [1.0, 2.0, 2.0]}]
```

- **优化加速**：VSS 扩展现在包含优化器规则，可使用 HNSW 索引加速 top-k 聚合，避免对底层表进行全扫描和排序

## 索引加速的 `LATERAL` 连接

针对批量向量搜索的性能优化：

- **挑战分析**：尽管基于 USearch 库的 HNSW 索引查找速度很快，但在逐个搜索向量时，DuckDB 的延迟与其他解决方案相比较高

- **性能瓶颈**：
  - USearch 并非瓶颈（仅占运行时间约 2%）
  - DuckDB 的矢量化执行引擎未针对"点查询"优化，最小工作单元为 2,048 行
  - 在小工作集上，预先优化和缓冲区分配的开销变得不必要

- **解决思路**：充分发挥 DuckDB 处理大量数据的优势，专注于"N:M"查询而非"1:N"查询，通过 `LATERAL` 连接实现

- **LATERAL 连接示例**：

```python
import duckdb
import numpy as np
import random

# 连接到DuckDB并加载VSS扩展
con = duckdb.connect(database=':memory:')
con.install_extension('vss')
con.load_extension('vss')

# 设置随机种子以确保可重复性
random.seed(42)
con.execute("SELECT setseed(0.42)")

# 创建示例表
con.execute("""
CREATE TABLE queries AS
    SELECT
        i AS id,
        [random(), random(), random()]::FLOAT[3] AS embedding
    FROM generate_series(1, 10000) r(i);
""")

con.execute("""
CREATE TABLE items AS
    SELECT
        i AS id,
        [random(), random(), random()]::FLOAT[3] AS embedding
    FROM generate_series(1, 10000) r(i);
""")

# 收集每个查询向量最接近的 5 个项目
result = con.execute("""
SELECT queries.id AS id, list(inner_id) AS matches
    FROM queries, LATERAL (
        SELECT
            items.id AS inner_id,
            array_distance(queries.embedding, items.embedding) AS dist
        FROM items
        ORDER BY dist
        LIMIT 5
    )
GROUP BY queries.id;
""")

# 转换为Pandas DataFrame并显示前几行
df_result = result.df()
print(df_result.head())
```

- **性能提升**：新的 HNSW_INDEX_JOIN 运算符将查询执行时间从 10 秒缩短到约 0.15 秒，加速约 66 倍

- **查询计划优化**：预估基数从 5,000,000 提升到 50,000，极大简化了执行计划

## 升级提示

如果您已经安装了 DuckDB v1.1.2 的 VSS 扩展，可通过以下命令获取最新版本：

```python
import duckdb

# 连接到DuckDB
con = duckdb.connect(database='your_database.db')

# 更新VSS扩展
con.execute("UPDATE EXTENSIONS (vss)")

print("VSS扩展已更新到最新版本")
```

## 总结

本次更新为 DuckDB 向量相似性搜索扩展带来了多方面的改进：

- 更快的索引创建速度
- 语义更一致的距离函数
- 强大的 top-k 聚合优化
- 高效的批量向量搜索支持

虽然本次更新主要关注新功能和性能提升，但团队仍在努力解决之前提到的一些限制，包括自定义索引和基于索引的优化。

如有任何问题或反馈，欢迎通过 [duckdb-vss GitHub 仓库](https://github.com/duckdb/duckdb-vss)或 [DuckDB Discord](https://discord.gg/duckdb) 联系我们。
