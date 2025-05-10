import time
import duckdb
import random

# 创建三个独立的数据库连接
con_hnsw = duckdb.connect(':memory:')
con_lateral = duckdb.connect(':memory:') 
con_regular = duckdb.connect(':memory:')

# 加载VSS扩展
for con in [con_hnsw, con_lateral, con_regular]:
    con.install_extension('vss')
    con.load_extension('vss')
    
    # 设置随机种子以确保可重复性
    random.seed(42)
    con.execute("SELECT setseed(0.42)")
    
    # 创建示例表并生成随机向量数据
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

# 先为HNSW测试创建索引
con_hnsw.execute("CREATE INDEX hnsw_idx ON items USING HNSW(embedding);")
print("HNSW索引已创建")

# 确保所有测试都处理相同数量的查询
query_limit = 1000  # 设置一个合适的查询数量，根据数据规模调整

# 1. 测试LATERAL JOIN方式
start_time = time.time()
lateral_query = f"""
SELECT queries.id AS id, list(inner_id) AS matches
FROM (SELECT * FROM queries LIMIT {query_limit}) as queries, 
LATERAL (
    SELECT
        items.id AS inner_id,
        array_distance(queries.embedding, items.embedding) AS dist
    FROM items
    ORDER BY dist
    LIMIT 5
)
GROUP BY queries.id;
"""
lateral_result = con_lateral.execute(lateral_query)
lateral_time = time.time() - start_time
print(f"LATERAL JOIN 方式耗时: {lateral_time:.4f} 秒 (处理 {query_limit} 个查询)")

# 2. 测试HNSW_INDEX_JOIN方式
start_time = time.time()
hnsw_query = f"""
SELECT queries.id AS id, list(inner_id) AS matches
FROM (SELECT * FROM queries LIMIT {query_limit}) as queries, 
LATERAL (
    SELECT
        items.id AS inner_id,
        array_distance(queries.embedding, items.embedding) AS dist
    FROM items
    WHERE array_distance(queries.embedding, items.embedding) < 10
    ORDER BY dist
    LIMIT 5
)
GROUP BY queries.id;
"""
hnsw_result = con_hnsw.execute(hnsw_query)
hnsw_time = time.time() - start_time
print(f"HNSW_INDEX_JOIN 方式耗时: {hnsw_time:.4f} 秒 (处理 {query_limit} 个查询)")

# 转换为Pandas DataFrame并显示前几行
df_result = hnsw_result.df()
print(f"\nHNSW查询结果前5行:")
print(df_result.head())

# 3. 测试常规JOIN方式
start_time = time.time()
non_lateral_query = f"""
WITH sample_queries AS (
    SELECT * FROM queries LIMIT {query_limit}
),
distances AS (
    SELECT 
        q.id AS query_id,
        i.id AS item_id,
        array_distance(q.embedding, i.embedding) AS dist
    FROM 
        sample_queries q,
        items i
),
ranked AS (
    SELECT 
        query_id,
        item_id,
        ROW_NUMBER() OVER (PARTITION BY query_id ORDER BY dist) AS rank
    FROM 
        distances
)
SELECT 
    query_id AS id,
    list(item_id) AS matches
FROM 
    ranked
WHERE 
    rank <= 5
GROUP BY 
    query_id;
"""
non_lateral_result = con_regular.execute(non_lateral_query)
non_lateral_time = time.time() - start_time
print(f"\n常规JOIN方式耗时: {non_lateral_time:.4f} 秒 (处理 {query_limit} 个查询)")

# 计算HNSW索引加速比
speedup_hnsw = lateral_time / hnsw_time if hnsw_time > 0 else float('inf')
print(f"\nHNSW_INDEX_JOIN 相对普通LATERAL JOIN的速度提升: {speedup_hnsw:.2f}x")

# 计算LATERAL JOIN加速比
speedup_lateral = non_lateral_time / lateral_time if lateral_time > 0 else float('inf')
print(f"LATERAL JOIN 相对常规JOIN的速度提升: {speedup_lateral:.2f}x")

# 总计速度提升
total_speedup = non_lateral_time / hnsw_time if hnsw_time > 0 else float('inf')
print(f"HNSW_INDEX_JOIN 相对常规JOIN的速度提升: {total_speedup:.2f}x")

# 关闭连接
con_hnsw.close()
con_lateral.close()
con_regular.close()