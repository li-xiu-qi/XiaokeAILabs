import duckdb
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体以支持中文显示
plt.rcParams['axes.unicode_minus'] = False  # 显示负号

# 连接到 DuckDB (可以是内存数据库或文件数据库)
con = duckdb.connect(database=':memory:', read_only=False)

# 创建一个表并插入数据 (或者从现有数据加载)
con.execute("CREATE TABLE items (name VARCHAR, value INTEGER)")
con.execute("INSERT INTO items VALUES ('A', 10), ('B', 20), ('C', 15)")

# 查询数据并转换为 Pandas DataFrame
df = con.execute("SELECT name, value FROM items").fetchdf()

# 使用 Matplotlib 绘图
plt.figure(figsize=(8, 6))
plt.bar(df['name'], df['value'])
plt.xlabel("名称")
plt.ylabel("值")
plt.title("DuckDB 数据可视化")
plt.show()

# 关闭连接
con.close()